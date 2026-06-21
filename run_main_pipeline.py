import argparse
from datetime import datetime
import importlib.util
import os
import sys
import builtins


def print(*args, **kwargs):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    builtins.print(f"[{ts}]", *args, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the main PCDD training pipeline.")
    parser.add_argument("--config", default="config.py")
    parser.add_argument("--stage", choices=["all", "distill", "reconstruct", "fusion"], default="all")
    parser.add_argument("--dataset", choices=["fakesv", "fakett"], default="fakett")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--distill-model-path", default=None)
    parser.add_argument("--multimodal-model-path", default=None)
    parser.add_argument("--distill-stage-epochs", type=int, default=None)
    parser.add_argument("--reconstruct-epochs", type=int, default=None)
    parser.add_argument("--fusion-epochs", type=int, default=None)
    parser.add_argument("--reconstruct-num-workers", type=int, default=None)
    parser.add_argument("--reconstruct-prefetch-factor", type=int, default=None)

    parser.add_argument("--reconstruct-experiment-name", default=None)
    parser.add_argument("--fusion-experiment-name", default=None)
    parser.add_argument("--multimodal-checkpoint-path", default=None)
    parser.add_argument("--distill-checkpoint-path", default=None)
    return parser.parse_args()


def load_config(config_path, repo_root):
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
        "training": getattr(module, "TRAINING", {}),
        "split_ratio": getattr(module, "SPLIT_RATIO", (0.70, 0.15, 0.15)),
    }
    for dataset_config in config.get("datasets", {}).values():
        for key in [
            "train_title_transcript", "val_title_transcript", "test_title_transcript",
            "train_analysis", "val_analysis", "test_analysis",
        ]:
            value = dataset_config.get(key)
            if value and not os.path.isabs(value):
                dataset_config[key] = os.path.join(repo_root, value)
    checkpoint_root = config.get("paths", {}).get("checkpoint_root")
    if checkpoint_root and not os.path.isabs(checkpoint_root):
        config["paths"]["checkpoint_root"] = os.path.join(repo_root, checkpoint_root)
    return config


def distill_checkpoint_path(root, dataset):
    return os.path.join(root, dataset, "stage3_model.pth")


def reconstruct_checkpoint_path(root, dataset, experiment_name, num_epochs):
    path = os.path.join(root, dataset, experiment_name)
    expected_path = os.path.join(path, f"{num_epochs - 1}_model.bin")
    if os.path.exists(expected_path):
        return expected_path

    if os.path.isdir(path):
        candidates = []
        for name in os.listdir(path):
            if name.endswith("_model.bin"):
                prefix = name[:-len("_model.bin")]
                if prefix.isdigit():
                    candidates.append((int(prefix), os.path.join(path, name)))
        if candidates:
            return max(candidates, key=lambda item: item[0])[1]

    return expected_path


def main():
    args = parse_args()
    repo_root = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(repo_root, config_path)
    config = load_config(config_path, repo_root)
    dataset_config = config.get("datasets", {}).get(args.dataset)
    if dataset_config is None:
        raise ValueError(f"Dataset '{args.dataset}' is not defined in {config_path}")

    training_config = config.get("training", {})
    model_config = config.get("models", {})
    path_config = config.get("paths", {})

    cuda_visible_devices = args.cuda_visible_devices or training_config.get("cuda_visible_devices", "3")
    checkpoint_root = args.checkpoint_root or path_config.get("checkpoint_root", os.path.join(repo_root, "checkpoint"))
    distill_model_path = args.distill_model_path or model_config.get("distill")
    multimodal_model_path = args.multimodal_model_path or model_config.get("multimodal")
    distill_stage_epochs = args.distill_stage_epochs or training_config.get("distill_stage_epochs", 30)
    reconstruct_epochs = args.reconstruct_epochs or training_config.get("reconstruct_epochs", 2)
    fusion_epochs = args.fusion_epochs or training_config.get("fusion_epochs", 15)
    reconstruct_num_workers = (
        args.reconstruct_num_workers
        if args.reconstruct_num_workers is not None
        else training_config.get("reconstruct_num_workers", 4)
    )
    reconstruct_prefetch_factor = (
        args.reconstruct_prefetch_factor
        if args.reconstruct_prefetch_factor is not None
        else training_config.get("reconstruct_prefetch_factor", 1)
    )
    fusion_num_workers = training_config.get("fusion_num_workers", 4)
    fusion_prefetch_factor = training_config.get("fusion_prefetch_factor", 1)
    prefer_preprocessed_media = bool(training_config.get("prefer_preprocessed_media", True))
    preprocessed_video_frames = int(training_config.get("preprocessed_video_frames", 8))
    ffprobe_timeout = config.get("preprocess", {}).get("ffprobe_timeout", 30)
    ffmpeg_timeout = config.get("preprocess", {}).get("ffmpeg_timeout", 60)
    preprocess_root = config.get("preprocess", {}).get("output_root")
    preprocessed_media_root = (
        os.path.join(preprocess_root, args.dataset)
        if preprocess_root and prefer_preprocessed_media
        else None
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    for relative_path in ["distill", "reconstruct", "fusion"]:
        module_path = os.path.join(repo_root, relative_path)
        if module_path not in sys.path:
            sys.path.append(module_path)

    from distill import full_training_pipeline
    from reconstruct import train as train_reconstruct
    from train_fusion import train_fusion

    distill_dir = os.path.join(checkpoint_root, "distill")
    reconstruct_dir = os.path.join(checkpoint_root, "reconstruct")
    fusion_dir = os.path.join(checkpoint_root, "fusion")

    reconstruct_experiment = args.reconstruct_experiment_name or f"main_{args.dataset}"
    fusion_experiment = args.fusion_experiment_name or f"main_{args.dataset}"

    distill_checkpoint = args.distill_checkpoint_path or distill_checkpoint_path(distill_dir, args.dataset)
    multimodal_checkpoint = args.multimodal_checkpoint_path or reconstruct_checkpoint_path(
        reconstruct_dir,
        args.dataset,
        reconstruct_experiment,
        reconstruct_epochs,
    )

    if args.stage in ["all", "distill"]:
        full_training_pipeline(
            dataset=args.dataset,
            model_path=distill_model_path,
            train_title_transcript_path=dataset_config["train_title_transcript"],
            train_analysis_path=dataset_config["train_analysis"],
            val_title_transcript_path=dataset_config["val_title_transcript"],
            val_analysis_path=dataset_config["val_analysis"],
            save_dir=distill_dir,
            stage1_num_epochs=distill_stage_epochs,
            stage2_num_epochs=distill_stage_epochs,
            stage3_num_epochs=distill_stage_epochs,
        )

    if args.stage in ["all", "reconstruct"]:
        train_reconstruct(
            model_path=multimodal_model_path,
            output_dir=reconstruct_dir,
            experiment_name=reconstruct_experiment,
            dataset=args.dataset,
            video_dir=dataset_config["video_dir"],
            train_json=dataset_config["train_title_transcript"],
            val_json=dataset_config["val_title_transcript"],
            num_epochs=reconstruct_epochs,
            batch_size=1,
            accumulation_steps=16,
            learning_rate=5e-5,
            lora_layers=8,
            forward_layers=8,
            num_workers=reconstruct_num_workers,
            prefetch_factor=reconstruct_prefetch_factor,
            mask_text_ratio=0.3,
            mask_audio_ratio=0.3,
            mask_video_ratio=0.3,
            ffprobe_timeout=ffprobe_timeout,
            ffmpeg_timeout=ffmpeg_timeout,
            log_samples=False,
            preprocessed_media_root=preprocessed_media_root,
            preprocessed_video_frames=preprocessed_video_frames,
            prefer_preprocessed_media=prefer_preprocessed_media,
        )

    if args.stage in ["all", "fusion"]:
        train_fusion(
            multimodal_model_path=multimodal_model_path,
            multimodal_checkpoint_path=multimodal_checkpoint,
            distill_model_path=distill_model_path,
            distill_checkpoint_path=distill_checkpoint,
            output_dir=fusion_dir,
            experiment_name=fusion_experiment,
            dataset=args.dataset,
            video_dir=dataset_config["video_dir"],
            train_json=dataset_config["train_title_transcript"],
            val_json=dataset_config["val_title_transcript"],
            test_json=dataset_config["test_title_transcript"],
            num_epochs=fusion_epochs,
            batch_size=1,
            accumulation_steps=16,
            learning_rate=5e-5,
            num_workers=fusion_num_workers,
            prefetch_factor=fusion_prefetch_factor,
            ffprobe_timeout=ffprobe_timeout,
            ffmpeg_timeout=ffmpeg_timeout,
            log_samples=False,
            preprocessed_media_root=preprocessed_media_root,
            preprocessed_video_frames=preprocessed_video_frames,
            prefer_preprocessed_media=prefer_preprocessed_media,
        )


if __name__ == "__main__":
    main()
