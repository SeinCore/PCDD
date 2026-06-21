from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RUN_ROOT = PROJECT_ROOT
# EDIT these to point to your downloaded datasets:
FAKESV_ROOT = Path("/home/wqy/data/dataset/FakeSVDataset")
FAKETT_ROOT = Path("/home/wqy/data/dataset/FakeTTDataset")

PATHS = {
    "checkpoint_root": str(RUN_ROOT / "checkpoint"),
    "data_root": str(RUN_ROOT / "data"),
    "model_root": str(RUN_ROOT / "models"),
}

MODELS = {
    "distill": "/home/wqy/data/model/Qwen3-Embedding-0.6B",
    "multimodal": "/home/wqy/data/model/Qwen2.5-Omni-3B",
    "whisper": "/home/wqy/data/model/whisper-large-v3",
}

SPLIT_RATIO = (0.70, 0.15, 0.15)

DATASETS = {
    "fakesv": {
        "metadata_path": str(FAKESV_ROOT / "data_complete.json"),
        "video_dir": str(FAKESV_ROOT / "filtered_video"),
        "train_title_transcript": str(RUN_ROOT / "data/fakesv/train_title&transcript.json"),
        "val_title_transcript": str(RUN_ROOT / "data/fakesv/val_title&transcript.json"),
        "test_title_transcript": str(RUN_ROOT / "data/fakesv/test_title&transcript.json"),
        "train_analysis": str(RUN_ROOT / "data/fakesv/train_analysis.json"),
        "val_analysis": str(RUN_ROOT / "data/fakesv/val_analysis.json"),
        "test_analysis": str(RUN_ROOT / "data/fakesv/test_analysis.json"),
    },
    "fakett": {
        "metadata_path": str(FAKETT_ROOT / "data.json"),
        "video_dir": str(FAKETT_ROOT / "video"),
        "train_title_transcript": str(RUN_ROOT / "data/fakett/train_title&transcript.json"),
        "val_title_transcript": str(RUN_ROOT / "data/fakett/val_title&transcript.json"),
        "test_title_transcript": str(RUN_ROOT / "data/fakett/test_title&transcript.json"),
        "train_analysis": str(RUN_ROOT / "data/fakett/train_analysis.json"),
        "val_analysis": str(RUN_ROOT / "data/fakett/val_analysis.json"),
        "test_analysis": str(RUN_ROOT / "data/fakett/test_analysis.json"),
    },
}

PREPROCESS = {
    "output_root": str(RUN_ROOT / "checkpoint/preprocess"),
    "num_frames": 16,
    "ocr_backend": "paddleocr",
    "media_workers": 16,
    "ocr_workers": 4,
    "ffprobe_timeout": 30,
    "ffmpeg_timeout": 60,
    "audio_timeout": 180,
    "asr_batch_size": 16,
    "text_batch_size": 16,
}

TRAINING = {
    # ---- user-adjustable ----
    "cuda_visible_devices": "0",
    "device": "cuda:0",
    "distill_stage_epochs": 15,
    "reconstruct_epochs": 2,
    "fusion_epochs": 15,
    "use_swanlab": False,
    # ---- usually leave as-is ----
    "batch_size": 1,
    "accumulation_steps": 16,
    "learning_rate": 5e-5,
    "reconstruct_num_workers": 0,
    "reconstruct_prefetch_factor": 1,
    "fusion_num_workers": 4,
    "fusion_prefetch_factor": 1,
    "prefer_preprocessed_media": True,
    "preprocessed_video_frames": 8,
}


def as_dict():
    return {
        "paths": dict(PATHS),
        "models": dict(MODELS),
        "datasets": {name: dict(value) for name, value in DATASETS.items()},
        "preprocess": dict(PREPROCESS),
        "training": dict(TRAINING),
        "split_ratio": SPLIT_RATIO,
    }
