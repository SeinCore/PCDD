import os
import json
import threading
import argparse
import importlib.util
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL = os.environ.get("PCDD_DEEPSEEK_MODEL", "deepseek-v4-flash")
BASE_URL = os.environ.get("PCDD_DEEPSEEK_BASE_URL", "https://api.deepseek.com")
API_KEY = os.environ.get("PCDD_DEEPSEEK_API_KEY")
API_RETRIES = int(os.environ.get("PCDD_API_RETRIES", "5"))
API_RETRY_BASE_SECONDS = float(os.environ.get("PCDD_API_RETRY_BASE_SECONDS", "2"))

SYSTEM_PROMPT = """You are an experienced fact-checking analyst for short video news. You need to maintain a neutral and objective stance and focus on identifying cognitive forgery strategies reflected in the textual narrative. Given the video title, on-screen text, and audio transcript, produce three-stage outputs that serve as cognitive supervision signals: (i) sentence-level evidence salience scores indicating how strongly each sentence contributes to potential cognitive forgery strategies, (ii) a concise reasoning rationale that assesses, based on the salient sentences and overall video text, whether cognitive forgery strategies are present, and (iii) decision-level diagnostic probabilities over the cognitive forgery strategies: Commonsense Violation, Logical Fallacy, and Emotional Manipulation."""

USER_PROMPT_TEMPLATE = """Title: {title}
On-screen Text: {ocr}
Audio Transcript: {transcript}"""

lock = threading.Lock()
error_ids_file = os.path.join(tempfile.gettempdir(), "pcdd_json_parse_error_ids.txt")
REQUIRED_RESPONSE_KEYS = {"EvidenceSalience", "ReasoningRationale", "StrategyProbabilities"}
REQUIRED_SCORE_KEYS = {"CommonsenseViolation", "LogicalFallacy", "EmotionalManipulation"}


def is_valid_response(response):
    if not isinstance(response, dict):
        return False
    if set(response.keys()) != REQUIRED_RESPONSE_KEYS:
        return False
    if not isinstance(response.get("EvidenceSalience"), list):
        return False
    if not isinstance(response.get("ReasoningRationale"), str):
        return False
    scores = response.get("StrategyProbabilities")
    if not isinstance(scores, dict) or set(scores.keys()) != REQUIRED_SCORE_KEYS:
        return False
    for value in scores.values():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        if not 0 <= float(value) <= 1:
            return False
    return True


def load_config(config_path: str) -> dict:
    spec = importlib.util.spec_from_file_location("pcdd_runtime_config", config_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Cannot import config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.as_dict() if hasattr(module, "as_dict") else {
        "datasets": getattr(module, "DATASETS", {}),
    }
    for dataset_config in config.get("datasets", {}).values():
        for key in [
            "train_title_transcript", "val_title_transcript", "test_title_transcript",
            "train_analysis", "val_analysis", "test_analysis",
        ]:
            value = dataset_config.get(key)
            if value and not os.path.isabs(value):
                dataset_config[key] = str(REPO_ROOT / value)
    return config

def save_results(results, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(output_path)}.",
        suffix=".tmp",
        dir=output_dir or ".",
        text=True,
    )
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, output_path)

def save_error_id(video_id):
    with open(error_ids_file, "a", encoding="utf-8") as f:
        f.write(f"{video_id}\n")

def _process_video(item, index, total, system_prompt, user_prompt_template, results, output_path):
    video_id = item.get("id")
    title = item.get("title")
    transcript = item.get("transcript", "")
    ocr = item.get("ocr", "")
    user_prompt = user_prompt_template.format(title=title, ocr=ocr, transcript=transcript)
    
    try:
        if not API_KEY:
            raise RuntimeError("PCDD_DEEPSEEK_API_KEY is not set")
        thread_client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        response = None
        for attempt in range(1, API_RETRIES + 1):
            try:
                response = thread_client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    stream=False
                )
                break
            except Exception as request_error:
                if attempt >= API_RETRIES:
                    raise
                delay = API_RETRY_BASE_SECONDS * attempt
                with lock:
                    print(
                        f"[{index+1}/{total}] Request failed for ID {video_id} "
                        f"(attempt {attempt}/{API_RETRIES}): {request_error}; retrying in {delay:.1f}s"
                    )
                time.sleep(delay)
        if response is None:
            raise RuntimeError("DeepSeek request did not return a response")
        content = response.choices[0].message.content
        if isinstance(content, str):
            try:
                parsed_content = json.loads(content)
            except json.JSONDecodeError:
                with lock:
                    save_error_id(video_id)
                    print(f"[{index+1}/{total}] JSON parse failed for ID {video_id}; saved to {error_ids_file}")
                parsed_content = content
        else:
            parsed_content = content
        
        result = {"id": video_id, "response": parsed_content}
        with lock:
            results.append(result)
            save_results(results, output_path)
            print(f"[{index+1}/{total}] Finished ID {video_id}; saved")
        return result
    except Exception as e:
        with lock:
            error_result = {
                "id": video_id,
                "title": title,
                "transcript": transcript,
                "error": str(e),
            }
            results.append(error_result)
            save_results(results, output_path)
            print(f"[{index+1}/{total}] Error for ID {video_id}: {e}; saved")
        return error_result

def process_video_fakesv(item, index, total, results, output_path):
    return _process_video(
        item,
        index,
        total,
        SYSTEM_PROMPT,
        USER_PROMPT_TEMPLATE,
        results,
        output_path,
    )

def process_video_fakett(item, index, total, results, output_path):
    return _process_video(
        item,
        index,
        total,
        SYSTEM_PROMPT,
        USER_PROMPT_TEMPLATE,
        results,
        output_path,
    )

def _run_analysis_for_file(input_file, output_file, process_fn, dataset_type, max_workers):
    print(f"Loading {input_file} (dataset={dataset_type})")
    
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            video_list = json.load(f)

        original_total = len(video_list)
        video_list = [item for item in video_list if (item.get("transcript") or "").strip()]
        skipped_empty = original_total - len(video_list)
        if skipped_empty:
            print(f"Skipped {skipped_empty} samples without transcript")
        
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                    print(f"Loaded {len(results)} existing results")
                before = len(results)
                results = [
                    item for item in results
                    if item.get("id")
                    and "error" not in item
                    and is_valid_response(item.get("response"))
                ]
                dropped = before - len(results)
                if dropped:
                    print(f"Dropped {dropped} previous error or malformed results; they will be retried")
            except Exception as e:
                print(f"Failed to load existing results: {e}; starting from an empty list")
                results = []
        else:
            results = []
            print("No existing result file found")
        
        processed_ids = {item.get("id") for item in results if item.get("id")}
        videos_to_process = []
        for item in video_list:
            video_id = item.get("id")
            if video_id not in processed_ids:
                videos_to_process.append(item)
        
        processed_count = sum(1 for item in video_list if item.get("id") in processed_ids)
        total = len(video_list)
        remaining = len(videos_to_process)
        
        print(f"Total={total}, processed={processed_count}, remaining={remaining}")
        
        if remaining == 0:
            print("All samples in this file are already processed")
            return
        
        print(f"Processing remaining samples with {max_workers} workers\n")
        
        def wrapped_process(item, idx, total_count):
            return process_fn(item, idx, total_count, results, output_file)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(wrapped_process, item, processed_count + idx, total): (item, idx)
                for idx, item in enumerate(videos_to_process)
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    item, idx = futures[future]
                    with lock:
                        print(f"[{processed_count + idx + 1}/{total}] Worker failed: {e}")
        
    except FileNotFoundError:
        print(f"File not found: {input_file}")
    except json.JSONDecodeError as e:
        print(f"JSON parse failed: {e}")
    except Exception as e:
        print(f"Error: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate teacher analysis JSON files with DeepSeek.")
    parser.add_argument("--config", default="config.py")
    parser.add_argument("--dataset", choices=["fakesv", "fakett"], default=os.environ.get("PCDD_DATASET", "fakesv"))
    parser.add_argument("--mode", choices=["train", "val", "test", "all"], default=os.environ.get("PCDD_ANALYSIS_MODE", "all"))
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("PCDD_MAX_WORKERS", "20")))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = str(REPO_ROOT / config_path)
    config = load_config(config_path)
    dataset_config = config.get("datasets", {}).get(args.dataset)
    if dataset_config is None:
        raise ValueError(f"Dataset '{args.dataset}' is not defined in {config_path}")
    io_pairs = [
        (dataset_config["train_title_transcript"], dataset_config["train_analysis"]),
        (dataset_config["val_title_transcript"], dataset_config["val_analysis"]),
        (dataset_config["test_title_transcript"], dataset_config["test_analysis"]),
    ]
    io_mode = args.mode.lower()
    if io_mode in {"train", "val", "test"}:
        index = {"train": 0, "val": 1, "test": 2}[io_mode]
        io_pairs = [io_pairs[index]]
    process_fn_lookup = {
        "fakesv": process_video_fakesv,
        "fakett": process_video_fakett,
    }
    process_fn = process_fn_lookup.get(args.dataset)
    if not io_pairs or process_fn is None:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    if not os.path.exists(error_ids_file):
        with open(error_ids_file, "w", encoding="utf-8") as f:
            pass
    
    for input_file, output_file in io_pairs:
        _run_analysis_for_file(input_file, output_file, process_fn, args.dataset, args.max_workers)
    
    print("\nDone")
