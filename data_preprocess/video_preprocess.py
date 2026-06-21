import argparse
import glob
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]


MODEL_ROOT = os.path.join(os.path.dirname(__file__), "..", "models")


def first_existing(candidates: Sequence[str], fallback: str) -> str:
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return fallback


def load_config(config_path: str) -> dict:
    spec = importlib.util.spec_from_file_location("pcdd_runtime_config", config_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Cannot import config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.as_dict() if hasattr(module, "as_dict") else {
        "paths": getattr(module, "PATHS", {}),
        "models": getattr(module, "MODELS", {}),
        "datasets": getattr(module, "DATASETS", {}),
        "preprocess": getattr(module, "PREPROCESS", {}),
        "split_ratio": getattr(module, "SPLIT_RATIO", (0.70, 0.15, 0.15)),
    }
    for dataset_config in config.get("datasets", {}).values():
        for key in [
            "train_title_transcript", "val_title_transcript", "test_title_transcript",
            "train_analysis", "val_analysis", "test_analysis",
        ]:
            value = dataset_config.get(key)
            if value and not os.path.isabs(value):
                dataset_config[key] = str(REPO_ROOT / value)
    output_root = config.get("preprocess", {}).get("output_root")
    if output_root and not os.path.isabs(output_root):
        config["preprocess"]["output_root"] = str(REPO_ROOT / output_root)
    return config


def load_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list")
    return data


def load_records(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return list(data.values())
    except json.JSONDecodeError:
        pass
    records = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def save_json(data: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def save_jsonl(rows: Sequence[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def run_cmd(cmd: Sequence[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


def video_path_for_id(video_dir: str, video_id: str) -> Optional[str]:
    for suffix in (".mp4", ".mov", ".mkv", ".avi", ".webm"):
        path = os.path.join(video_dir, f"{video_id}{suffix}")
        if os.path.exists(path):
            return path
    matches = glob.glob(os.path.join(video_dir, f"{video_id}.*"))
    return matches[0] if matches else None


def get_video_duration(video_path: str, timeout: int = 30) -> Optional[float]:
    try:
        result = run_cmd(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            timeout=timeout,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def extract_frames(
    video_path: str,
    output_dir: str,
    num_frames: int,
    overwrite: bool = False,
    ffprobe_timeout: int = 30,
    ffmpeg_timeout: int = 60,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    frame_paths = [os.path.join(output_dir, f"frame_{idx:03d}.jpg") for idx in range(num_frames)]
    if not overwrite and all(os.path.exists(path) for path in frame_paths):
        return frame_paths

    duration = get_video_duration(video_path, ffprobe_timeout)
    if duration is None or duration <= 0:
        raise RuntimeError(f"Invalid video duration: {video_path}")

    timestamps = [(idx + 0.5) * duration / num_frames for idx in range(num_frames)]
    # Extract last frame first — if it fails the video is likely corrupted near the
    # end, so we bail early and let the caller retry with a trimmed video.
    last_good: Optional[str] = None
    for idx in range(num_frames - 1, -1, -1):
        try:
            run_cmd(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{timestamps[idx]:.3f}",
                    "-i",
                    video_path,
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    frame_paths[idx],
                ],
                timeout=ffmpeg_timeout,
            )
            last_good = frame_paths[idx]
        except Exception:
            pass  # keep going; missing frames will be handled by the caller retrying with a trimmed video

    if last_good is None:
        raise RuntimeError(f"Cannot extract any frame from {video_path}")
    missing = [p for p in frame_paths if not os.path.exists(p)]
    if missing:
        raise RuntimeError(
            f"Extracted only {num_frames - len(missing)}/{num_frames} frames from {video_path}"
        )
    return frame_paths


def extract_audio(video_path: str, wav_path: str, overwrite: bool = False, timeout: int = 180) -> Optional[str]:
    if os.path.exists(wav_path) and not overwrite:
        return wav_path
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    try:
        run_cmd(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                wav_path,
            ],
            timeout=timeout,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return wav_path if os.path.exists(wav_path) else None


def normalize_text(parts: Iterable[Optional[str]]) -> str:
    cleaned = []
    for part in parts:
        if not part:
            continue
        text = str(part).strip()
        if not text or text.lower() == "none":
            continue
        cleaned.append(text)
    return "\n".join(cleaned)


def _label(annotation: object, mapping: dict) -> Optional[int]:
    key = str(annotation).strip().lower()
    return mapping.get(key)


def metadata_to_record(dataset: str, row: dict) -> Optional[dict]:
    if dataset == "fakesv":
        annotation = str(row.get("annotation", "")).strip()
        if annotation == "辟谣":
            return None
        label = _label(annotation, {"假": 1, "真": 0})
        timestamp = row.get("publish_time_norm")
        title = row.get("title")
    elif dataset == "fakett":
        label = _label(row.get("annotation"), {"fake": 1, "real": 0})
        timestamp = row.get("publish_time")
        title = row.get("description")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    video_id = row.get("video_id")
    if video_id is None or label is None or timestamp is None:
        return None
    return {
        "id": str(video_id),
        "label": int(label),
        "title": "" if title is None else str(title),
        "publish_time": int(timestamp),
        "ocr": "",
        "asr": "",
        "transcript": "",
    }


def split_counts(total: int, ratio: Sequence[float]) -> tuple:
    first_ratio, second_ratio, _ = ratio
    train_count = int(total * first_ratio)
    val_count = int(total * second_ratio)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def init_splits(dataset: str, dataset_config: dict, split_ratio: Sequence[float], overwrite: bool = False) -> None:
    metadata_path = dataset_config.get("metadata_path")
    if not metadata_path:
        raise ValueError(f"No metadata_path configured for dataset={dataset}")
    rows = load_records(metadata_path)
    records = [record for row in rows if (record := metadata_to_record(dataset, row)) is not None]
    records.sort(key=lambda item: (item["publish_time"], item["id"]))

    seen = set()
    unique_records = []
    for record in records:
        if record["id"] in seen:
            continue
        seen.add(record["id"])
        unique_records.append(record)

    n_train, n_val, _ = split_counts(len(unique_records), split_ratio)
    splits = {
        "train": unique_records[:n_train],
        "val": unique_records[n_train:n_train + n_val],
        "test": unique_records[n_train + n_val:],
    }
    for split, items in splits.items():
        path = dataset_config[f"{split}_title_transcript"]
        if os.path.exists(path) and not overwrite:
            print(f"Keeping existing {split} split: {path}")
            continue
        save_json(items, path)
        print(f"Wrote {len(items)} {dataset} {split} samples to {path}")


class OcrRunner:
    def __init__(self, backend: str, lang: str, device: str):
        self.backend = backend
        self.lang = lang
        self.device = device
        if backend == "easyocr":
            import easyocr

            langs = ["ch_sim", "en"] if lang in {"ch", "zh", "ch_sim"} else ["en"]
            self.reader = easyocr.Reader(langs, gpu=device.startswith("cuda"))
        elif backend == "paddleocr":
            from paddleocr import PaddleOCR

            paddle_lang = "ch" if lang in {"ch", "zh", "ch_sim"} else "en"
            self.reader = PaddleOCR(use_angle_cls=True, lang=paddle_lang, show_log=False)
        else:
            raise ValueError(f"Unsupported OCR backend: {backend}")

    def read_frame(self, frame_path: str) -> str:
        if self.backend == "easyocr":
            texts = self.reader.readtext(frame_path, detail=0)
            return " ".join(text.strip() for text in texts if text and text.strip())

        result = self.reader.ocr(frame_path, cls=True)
        texts = []
        for page in result or []:
            for line in page or []:
                if len(line) >= 2 and line[1]:
                    texts.append(str(line[1][0]).strip())
        return " ".join(text for text in texts if text)

    def read_frames(self, frame_paths: Sequence[str]) -> str:
        texts = []
        last = None
        for frame_path in frame_paths:
            text = self.read_frame(frame_path)
            if text and text != last:
                texts.append(text)
                last = text
        return normalize_text(texts)


class AsrRunner:
    def __init__(self, model_path: str, device: str, batch_size: int):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_path)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            torch_dtype=torch_dtype,
            batch_size=batch_size,
            device=device,
        )
        self.batch_size = batch_size

    def transcribe(self, wav_paths: Sequence[str]) -> List[str]:
        if not wav_paths:
            return []
        outputs = self.pipe(list(wav_paths), batch_size=min(self.batch_size, len(wav_paths)))
        if isinstance(outputs, dict):
            outputs = [outputs]
        return [str(item.get("text", "")).strip() for item in outputs]


class TextEncoder:
    def __init__(self, model_path: str, device: str):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
        self.model.eval()
        self.device = device

    def encode(self, texts: Sequence[str], batch_size: int) -> torch.Tensor:
        outputs = []
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding text"):
            batch = texts[start : start + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                hidden = self.model(**inputs).last_hidden_state
                lengths = inputs["attention_mask"].sum(dim=1) - 1
                batch_idx = torch.arange(hidden.size(0), device=self.device)
                emb = hidden[batch_idx, lengths]
                emb = F.normalize(emb, p=2, dim=-1)
            outputs.append(emb.cpu())
        return torch.cat(outputs, dim=0) if outputs else torch.empty(0)


def selected_json_paths_from_config(dataset_config: dict, split: str) -> List[str]:
    if split == "all":
        return [
            dataset_config["train_title_transcript"],
            dataset_config["val_title_transcript"],
            dataset_config["test_title_transcript"],
        ]
    key = f"{split}_title_transcript"
    return [dataset_config[key]]


def _media_paths(output_root: str, video_id: str) -> tuple:
    return (
        os.path.join(output_root, "frames", video_id),
        os.path.join(output_root, "audios", f"{video_id}.wav"),
    )


def _trim_video_tail(video_path: str, ffprobe_timeout: int, ffmpeg_timeout: int) -> Optional[str]:
    """Trim the last second of a video and return a temp path, or None on failure."""
    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True, text=True, check=True,
            timeout=ffprobe_timeout,
        )
        duration = float(probe.stdout.strip())
    except Exception:
        return None

    new_duration = max(duration - 1.0, 0.5)
    tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()
    # Try stream copy first (fast), then fall back to re-encode (robust for corrupted streams).
    for use_copy in (True, False):
        if use_copy:
            codec_args = ["-c", "copy"]
            this_timeout = ffmpeg_timeout
        else:
            codec_args = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-c:a", "aac", "-b:a", "128k"]
            this_timeout = ffmpeg_timeout * 3
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", video_path,
                    "-t", f"{new_duration:.3f}",
                    *codec_args, tmp_path,
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=True, timeout=this_timeout,
            )
            return tmp_path
        except Exception:
            # Remove failed output before retry.
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    return None


def _fill_missing_frames(frame_dir: str, num_frames: int, result: dict) -> None:
    """Fill any missing frames by copying the last successfully extracted frame."""
    frame_paths = [os.path.join(frame_dir, f"frame_{idx:03d}.jpg") for idx in range(num_frames)]
    existing = [p for p in frame_paths if os.path.exists(p)]
    if not existing:
        return
    last_good = existing[-1]
    for p in frame_paths:
        if not os.path.exists(p):
            shutil.copyfile(last_good, p)


def _prepare_media(item: dict, video_dir: str, output_root: str, args: argparse.Namespace) -> dict:
    video_id = str(item.get("id", ""))
    if not video_id:
        return {"item": item, "id": video_id, "error": "missing id", "stage": "lookup"}

    video_path = video_path_for_id(video_dir, video_id)
    if not video_path:
        return {"item": item, "id": video_id, "error": "video file not found", "stage": "lookup"}

    frame_dir, audio_path = _media_paths(output_root, video_id)
    result = {"item": item, "id": video_id, "frame_paths": [], "wav_path": None}

    def _try_extract(working_path: str) -> None:
        if args.stage in {"all", "frames", "ocr"}:
            result["frame_paths"] = extract_frames(
                working_path, frame_dir,
                args.num_frames, args.overwrite,
                args.ffprobe_timeout, args.ffmpeg_timeout,
            )
        else:
            result["frame_paths"] = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))

        if args.stage in {"all", "audio", "asr"}:
            result["wav_path"] = extract_audio(working_path, audio_path, args.overwrite, args.audio_timeout)
        else:
            result["wav_path"] = audio_path if os.path.exists(audio_path) else None

    trimmed_path = None
    try:
        _try_extract(video_path)
    except Exception as first_exc:
        trimmed_path = _trim_video_tail(
            video_path,
            args.ffprobe_timeout,
            args.ffmpeg_timeout,
        )
        if trimmed_path is None:
            result["error"] = str(first_exc)
            result["stage"] = args.stage
        else:
            try:
                _try_extract(trimmed_path)
            except Exception:
                # Last resort: fill any still-missing frames by copying the last good one.
                _fill_missing_frames(frame_dir, args.num_frames, result)
    finally:
        if trimmed_path is not None and os.path.exists(trimmed_path):
            try:
                os.remove(trimmed_path)
            except Exception:
                pass
    return result


def _save_if_changed(data: List[dict], json_path: str, changed: bool) -> None:
    if changed:
        save_json(data, json_path)


def _needs_text_update(item: dict, args: argparse.Namespace) -> bool:
    if args.overwrite:
        return True
    if args.stage in {"frames", "audio"}:
        return True
    if args.stage == "ocr":
        return not bool(item.get("ocr"))
    if args.stage == "asr":
        if item.get("_no_audio"):
            return False
        return not bool(item.get("asr"))
    if args.stage == "transcript":
        return not bool(item.get("transcript"))
    if args.stage == "all":
        has_ocr = bool(item.get("ocr"))
        has_transcript = bool(item.get("transcript"))
        has_asr = bool(item.get("asr")) or item.get("_no_audio")
        return not (has_ocr and has_asr and has_transcript)
    return True


def _update_transcript(item: dict, args: argparse.Namespace) -> bool:
    transcript = normalize_text([item.get("ocr"), item.get("asr")])
    if transcript and (args.overwrite or item.get("transcript") != transcript):
        item["transcript"] = transcript
        return True
    return False


def update_text_fields(
    paths: Sequence[str],
    video_dir: str,
    output_root: str,
    args: argparse.Namespace,
) -> None:
    ocr_runner = None
    asr_runner = None
    if args.stage in {"all", "ocr"} and args.ocr_workers <= 1:
        ocr_runner = OcrRunner(args.ocr_backend, args.ocr_lang, args.device)
    if args.stage in {"all", "asr"}:
        asr_runner = AsrRunner(args.whisper_model, args.device, args.asr_batch_size)
    ocr_state = threading.local()

    def get_thread_ocr_runner() -> OcrRunner:
        runner = getattr(ocr_state, "runner", None)
        if runner is None:
            runner = OcrRunner(args.ocr_backend, args.ocr_lang, args.device)
            ocr_state.runner = runner
        return runner

    def run_parallel_ocr(result: dict) -> dict:
        try:
            text = get_thread_ocr_runner().read_frames(result.get("frame_paths", []))
            return {"result": result, "ocr": text}
        except Exception as exc:
            return {"result": result, "error": str(exc)}

    failures = []
    for json_path in paths:
        data = load_json(json_path)
        file_changed = False
        source_records = data[: args.limit] if args.limit else data
        records = [item for item in source_records if _needs_text_update(item, args)]
        print(f"{json_path}: {len(records)} of {len(source_records)} samples need stage={args.stage}", flush=True)
        if not records:
            continue

        media_results = []
        with ThreadPoolExecutor(max_workers=max(1, args.media_workers)) as executor:
            futures = [
                executor.submit(_prepare_media, item, video_dir, output_root, args)
                for item in records
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Preparing media {json_path}"):
                result = future.result()
                if result.get("error"):
                    failures.append({"id": result.get("id"), "stage": result.get("stage", args.stage), "error": result["error"]})
                else:
                    media_results.append(result)

        if args.stage in {"frames", "audio"}:
            continue

        if args.stage in {"all", "ocr"}:
            ocr_pending = [
                result for result in media_results
                if args.overwrite or not result["item"].get("ocr")
            ]
            if args.ocr_workers <= 1:
                for result in tqdm(ocr_pending, desc=f"OCR {json_path}"):
                    item = result["item"]
                    try:
                        item["ocr"] = ocr_runner.read_frames(result.get("frame_paths", []))
                        changed = True
                        if args.stage in {"all", "transcript"}:
                            changed = _update_transcript(item, args) or changed
                        file_changed = file_changed or changed
                        _save_if_changed(data, json_path, changed)
                    except Exception as exc:
                        failures.append({"id": result.get("id"), "stage": "ocr", "error": str(exc)})
            else:
                with ThreadPoolExecutor(max_workers=max(1, args.ocr_workers)) as executor:
                    futures = [executor.submit(run_parallel_ocr, result) for result in ocr_pending]
                    for future in tqdm(as_completed(futures), total=len(futures), desc=f"OCR {json_path}"):
                        output = future.result()
                        result = output["result"]
                        item = result["item"]
                        if output.get("error"):
                            failures.append({"id": result.get("id"), "stage": "ocr", "error": output["error"]})
                            continue
                        item["ocr"] = output.get("ocr", "")
                        changed = True
                        if args.stage in {"all", "transcript"}:
                            changed = _update_transcript(item, args) or changed
                        file_changed = file_changed or changed
                        _save_if_changed(data, json_path, changed)

        if args.stage == "transcript":
            for result in tqdm(media_results, desc=f"Updating transcript {json_path}"):
                if _update_transcript(result["item"], args):
                    file_changed = True
                    save_json(data, json_path)

        if asr_runner is not None:
            pending = [
                result for result in media_results
                if result.get("wav_path") and (args.overwrite or not result["item"].get("asr"))
            ]
            for start in tqdm(range(0, len(pending), args.asr_batch_size), desc=f"ASR {json_path}"):
                batch = pending[start:start + args.asr_batch_size]
                try:
                    texts = asr_runner.transcribe([result["wav_path"] for result in batch])
                except Exception as exc:
                    for result in batch:
                        failures.append({"id": result.get("id"), "stage": "asr", "error": str(exc)})
                    continue

                for result, text in zip(batch, texts):
                    item = result["item"]
                    item["asr"] = text
                    transcript = normalize_text([item.get("ocr"), item.get("asr")])
                    if transcript and (args.overwrite or item.get("transcript") != transcript):
                        item["transcript"] = transcript
                    file_changed = True
                _save_if_changed(data, json_path, True)

            # Videos without an audio track: mark as complete so they are not retried.
            for result in media_results:
                item = result["item"]
                if not result.get("wav_path") and not item.get("_no_audio"):
                    item["_no_audio"] = True
                    item.setdefault("asr", "")
                    if not item.get("transcript"):
                        item["transcript"] = normalize_text([item.get("ocr"), ""])
                    file_changed = True
            if file_changed:
                _save_if_changed(data, json_path, True)

        if file_changed:
            save_json(data, json_path)

    if failures:
        failure_path = os.path.join(output_root, f"{args.stage}_failures.jsonl")
        save_jsonl(failures, failure_path)
        print(f"Saved {len(failures)} failures to {failure_path}")


def encode_text_features(paths: Sequence[str], output_root: str, args: argparse.Namespace) -> None:
    records = []
    for path in paths:
        records.extend(load_json(path))
    if args.limit:
        records = records[: args.limit]
    ids = [str(item.get("id")) for item in records if item.get("id")]
    texts = [
        f"*Title*: {item.get('title', '')}. *Transcript*: {item.get('transcript') or ''}"
        for item in records
        if item.get("id")
    ]
    encoder = TextEncoder(args.text_encoder_model, args.device)
    features = encoder.encode(texts, args.text_batch_size)
    save_path = os.path.join(output_root, "text_features.pt")
    os.makedirs(output_root, exist_ok=True)
    torch.save({video_id: features[idx] for idx, video_id in enumerate(ids)}, save_path)
    print(f"Saved text features to {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess videos into OCR, ASR, transcripts, frames, and optional text feature caches.")
    parser.add_argument("--config", default="config.py")
    parser.add_argument("--dataset", choices=["fakesv", "fakett"], required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument(
        "--stage",
        choices=["init", "all", "frames", "audio", "ocr", "asr", "transcript", "text_features"],
        default="all",
    )
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ocr-backend", choices=["easyocr", "paddleocr"], default=None)
    parser.add_argument("--ocr-lang", default=None)
    parser.add_argument("--whisper-model", default=None)
    parser.add_argument("--asr-batch-size", type=int, default=None)
    parser.add_argument("--media-workers", type=int, default=None)
    parser.add_argument("--ocr-workers", type=int, default=None)
    parser.add_argument("--ffprobe-timeout", type=int, default=None)
    parser.add_argument("--ffmpeg-timeout", type=int, default=None)
    parser.add_argument("--audio-timeout", type=int, default=None)
    parser.add_argument("--text-encoder-model", default=None)
    parser.add_argument("--text-batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = str(REPO_ROOT / config_path)
    config = load_config(config_path)
    dataset_config = config.get("datasets", {}).get(args.dataset)
    if dataset_config is None:
        raise ValueError(f"Dataset '{args.dataset}' is not defined in {config_path}")
    preprocess_config = config.get("preprocess", {})
    model_config = config.get("models", {})
    video_dir = args.video_dir or dataset_config.get("video_dir")
    if not video_dir:
        raise ValueError(f"No video_dir configured for dataset={args.dataset}")
    output_base = args.output_root or preprocess_config.get("output_root") or os.path.join("checkpoint", "preprocess")
    output_root = os.path.join(output_base, args.dataset)
    args.num_frames = args.num_frames or int(preprocess_config.get("num_frames", 16))
    args.ocr_backend = args.ocr_backend or preprocess_config.get("ocr_backend", "paddleocr")
    args.asr_batch_size = args.asr_batch_size or int(preprocess_config.get("asr_batch_size", 4))
    args.media_workers = args.media_workers or int(preprocess_config.get("media_workers", 4))
    args.ocr_workers = args.ocr_workers or int(preprocess_config.get("ocr_workers", 1))
    args.ffprobe_timeout = args.ffprobe_timeout or int(preprocess_config.get("ffprobe_timeout", 30))
    args.ffmpeg_timeout = args.ffmpeg_timeout or int(preprocess_config.get("ffmpeg_timeout", 60))
    args.audio_timeout = args.audio_timeout or int(preprocess_config.get("audio_timeout", 180))
    args.text_batch_size = args.text_batch_size or int(preprocess_config.get("text_batch_size", 16))
    args.text_encoder_model = args.text_encoder_model or model_config.get("distill")
    args.ocr_lang = args.ocr_lang or ("ch" if args.dataset == "fakesv" else "en")
    args.whisper_model = args.whisper_model or first_existing(
        [
            model_config.get("whisper"),
            os.path.join(MODEL_ROOT, "whisper-large-v3"),
            os.path.join(MODEL_ROOT, "openai--whisper-large-v3"),
        ],
        "openai/whisper-large-v3",
    )

    if args.stage in {"init", "all"}:
        init_splits(args.dataset, dataset_config, config.get("split_ratio", (0.70, 0.15, 0.15)), args.overwrite)
    if args.stage == "init":
        return

    paths = selected_json_paths_from_config(dataset_config, args.split)
    Path(output_root).mkdir(parents=True, exist_ok=True)

    if args.stage in {"all", "frames", "audio", "ocr", "asr", "transcript"}:
        update_text_fields(paths, video_dir, output_root, args)
    if args.stage == "text_features":
        encode_text_features(paths, output_root, args)

if __name__ == "__main__":
    main()
