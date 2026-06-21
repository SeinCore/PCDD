import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["FFMPEG_LOGLEVEL"]="error"
import random
import logging
import warnings
import subprocess
import tempfile
from datetime import datetime
import builtins
import torch
from torch.utils.data import Dataset, DataLoader
from reconstruction_model import ReconstructionModel
from qwen_omni_utils import process_mm_info
import json


def print(*args, **kwargs):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    builtins.print(f"[{ts}]", *args, **kwargs)

class _NoOpSwanLab:
    def init(self, *args, **kwargs): pass
    def log(self, *args, **kwargs): pass
    def finish(self, *args, **kwargs): pass

try:
    import swanlab
except ModuleNotFoundError:
    swanlab = _NoOpSwanLab()

try:
    from config import TRAINING
    if not TRAINING.get("use_swanlab", True):
        swanlab = _NoOpSwanLab()
except Exception:
    pass

class VideoDataset(Dataset):
    """Video dataset for reconstruction training."""
    def __init__(
        self,
        video_dir,
        processor,
        use_audio_in_video=True,
        id_list=None,
        id2title=None,
        ffprobe_timeout=30,
        ffmpeg_timeout=60,
        log_samples=False,
        preprocessed_media_root=None,
        preprocessed_video_frames=8,
        prefer_preprocessed_media=True,
    ):
        self.video_dir = video_dir
        self.processor = processor
        self.use_audio_in_video = use_audio_in_video
        self.id_list = id_list
        self.id2title = id2title or {}
        self.ffprobe_timeout = ffprobe_timeout
        self.ffmpeg_timeout = ffmpeg_timeout
        self.log_samples = log_samples
        self.preprocessed_media_root = preprocessed_media_root
        self.preprocessed_video_frames = preprocessed_video_frames
        self.prefer_preprocessed_media = prefer_preprocessed_media
        
        warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
        warnings.filterwarnings("ignore", category=FutureWarning, module="librosa.core.audio")
        class _NoisyFilter(logging.Filter):
            def filter(self, record):
                msg = record.getMessage()
                return "System prompt modified, audio output may not work as expected" not in msg
        logging.getLogger().addFilter(_NoisyFilter())
        logging.getLogger("qwen_omni_utils.v2_5.vision_process").setLevel(logging.ERROR)
        logging.getLogger("qwen_omni_utils.v2_5.vision_process").propagate = False
        os.environ.setdefault("FFMPEG_LOGLEVEL", "error")
        os.environ.setdefault("LIBAV_LOG_LEVEL", "error")
        
        files = []
        for vid in self.id_list:
            # <id>.mp4
            p = os.path.join(video_dir, f"{vid}.mp4")
            if os.path.exists(p):
                files.append((p, vid))
        combined = list(zip([p for p, _ in files], [vid for _, vid in files]))
        
        print(f"Found {len(combined)} video files")
        random.seed(42)
        random.shuffle(combined)
        
        self.video_files = [p for p, _ in combined]
        self.video_ids = [vid for _, vid in combined]
        
        self.system_prompt = "Process the provided short video and its title."
    
    def __len__(self):
        return len(self.video_files)
    
    def _select_preprocessed_frames(self, frame_dir):
        frame_paths = [
            os.path.join(frame_dir, name)
            for name in sorted(os.listdir(frame_dir))
            if name.startswith("frame_") and name.endswith(".jpg")
        ]
        if len(frame_paths) < self.preprocessed_video_frames:
            return None
        if len(frame_paths) == self.preprocessed_video_frames:
            return frame_paths
        return [
            frame_paths[int(i * len(frame_paths) / self.preprocessed_video_frames)]
            for i in range(self.preprocessed_video_frames)
        ]

    def _preprocessed_media(self, vid):
        if not self.preprocessed_media_root or not self.prefer_preprocessed_media:
            return None

        frame_dir = os.path.join(self.preprocessed_media_root, "frames", vid)
        audio_path = os.path.join(self.preprocessed_media_root, "audios", f"{vid}.wav")
        if not os.path.isdir(frame_dir):
            return None

        frame_paths = self._select_preprocessed_frames(frame_dir)
        if not frame_paths:
            return None

        has_audio = os.path.exists(audio_path)
        if not has_audio:
            print(
                f"[Warning] Preprocessed audio missing for {vid}, "
                f"proceeding with frames only. "
                f"Re-run preprocessing to fix: "
                f"python data_preprocess/video_preprocess.py --config config.py "
                f"--dataset <dataset> --stage all",
                flush=True,
            )

        return {
            "frames": frame_paths,
            "audio": audio_path if has_audio else None,
            "source": frame_dir,
        }

    def _build_conversation(self, title, video_path=None, frame_paths=None, audio_path=None):
        content = []
        if frame_paths is not None:
            content.append({"type": "video", "video": frame_paths, "fps": 2.0})
            if audio_path is not None:
                content.append({"type": "audio", "audio": audio_path})
        else:
            content.append({"type": "video", "video": video_path, "nframes": 8})
        content.append({"type": "text", "text": f"Video title: {title}"})

        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": self.system_prompt}
                ],
            },
            {
                "role": "user",
                "content": content,
            },
        ]

    def _preprocess(self, video_path, title, return_raw_media=False, frame_paths=None, audio_path=None):
        using_preprocessed_media = frame_paths is not None
        effective_use_audio_in_video = self.use_audio_in_video and not using_preprocessed_media

        # Trim preprocessed audio to 30s max to avoid OOM in the Qwen2.5-Omni audio tower.
        audio_trim_tmp = None
        if audio_path is not None and os.path.exists(audio_path):
            try:
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
                    capture_output=True, text=True, timeout=self.ffprobe_timeout,
                )
                dur = float(probe.stdout.strip())
                if dur > 30.0:
                    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp_file.close()
                    subprocess.run(
                        ["ffmpeg", "-y", "-loglevel", "error",
                         "-i", audio_path, "-t", "30", tmp_file.name],
                        check=True, timeout=self.ffmpeg_timeout,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    )
                    audio_trim_tmp = tmp_file.name
                    audio_path = audio_trim_tmp
            except Exception:
                pass

        conversation = [
            *self._build_conversation(
                title,
                video_path=video_path,
                frame_paths=frame_paths,
                audio_path=audio_path,
            )
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        cap_tmp = None if using_preprocessed_media else self._cap_video_duration(video_path, max_seconds=30.0)
        eff_video_path = cap_tmp if cap_tmp is not None else video_path
        try:
            if not using_preprocessed_media:
                conversation[1]["content"][0]["video"] = eff_video_path

            # Try with audio first; if the video has no audio track, retry without.
            use_audio = effective_use_audio_in_video
            try:
                audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio)
                inputs = self.processor(
                    text=text, audio=audios, images=images, videos=videos,
                    return_tensors="pt", padding=True, use_audio_in_video=use_audio,
                )
                media_audio_in_video = use_audio
            except Exception:
                if not use_audio:
                    raise
                print(
                    f"[Warning] Audio processing failed, retrying without audio "
                    f"(video={video_path})", flush=True,
                )
                audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
                inputs = self.processor(
                    text=text, audio=audios, images=images, videos=videos,
                    return_tensors="pt", padding=True, use_audio_in_video=False,
                )
                media_audio_in_video = False
            
            if return_raw_media:
                inputs['raw_audio'] = audios[0] if audios and len(audios) > 0 else None
                inputs['raw_video'] = videos[0] if videos and len(videos) > 0 else None
            if not title or (isinstance(title, str) and title.strip() == ""):
                pass
            else:
                input_ids = inputs["input_ids"][0].tolist()
                
                full_title_text = f"Video title: {title}"
                
                title_prefix = "Video title: "
                prefix_tokens = self.processor.tokenizer.encode(title_prefix, add_special_tokens=False)
                
                full_title_tokens = self.processor.tokenizer.encode(full_title_text, add_special_tokens=False)
                
                title_token_count = len(full_title_tokens) - len(prefix_tokens)
                
                found = False
                for i in range(len(input_ids) - len(full_title_tokens) + 1):
                    if input_ids[i:i+len(full_title_tokens)] == full_title_tokens:
                        title_start = i + len(prefix_tokens)
                        title_end = i + len(full_title_tokens)
                        inputs["title_token_indices"] = torch.tensor([[title_start, title_end]], dtype=torch.long)
                        found = True
                        break
                
                if not found:
                    for i in range(len(input_ids) - len(prefix_tokens)):
                        if input_ids[i:i+len(prefix_tokens)] == prefix_tokens:
                            title_start = i + len(prefix_tokens)
                            title_end = min(title_start + title_token_count, len(input_ids))
                            
                            special_tokens = [151645, 198, 151644, 151643]
                            for j in range(title_start, title_end):
                                if j < len(input_ids) and input_ids[j] in special_tokens:
                                    title_end = j
                                    break
                            
                            if title_end > title_start:
                                inputs["title_token_indices"] = torch.tensor([[title_start, title_end]], dtype=torch.long)
                                found = True
                            break
                
                if not found:
                    raise ValueError(
                        f"Failed to locate title tokens:\n"
                        f"  video_path={video_path}\n"
                        f"  title={repr(title)}\n"
                        f"  prefix_tokens={prefix_tokens}\n"
                        f"  full_title_tokens={full_title_tokens}\n"
                        f"  input_ids_len={len(input_ids)}\n"
                        f"  input_ids_tail_100={input_ids[-100:]}"
                    )
            
            return inputs, media_audio_in_video
        finally:
            if cap_tmp is not None and os.path.exists(cap_tmp):
                try:
                    os.remove(cap_tmp)
                except Exception:
                    pass
            if audio_trim_tmp is not None and os.path.exists(audio_trim_tmp):
                try:
                    os.remove(audio_trim_tmp)
                except Exception:
                    pass

    def _trim_video(self, video_path):
        """Trim one second from the end of a video and return a temp path."""
        try:
            probe = subprocess.run(
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
                capture_output=True,
                text=True,
                check=True,
                timeout=self.ffprobe_timeout,
            )
            duration = float(probe.stdout.strip())
            new_duration = max(duration - 1.0, 0.5)
            tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_path = tmp_file.name
            tmp_file.close()
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-t",
                f"{new_duration}",
                "-c",
                "copy",
                tmp_path,
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.ffmpeg_timeout,
            )
            if result.returncode != 0:
                os.remove(tmp_path)
                raise RuntimeError(result.stderr.decode("utf-8", errors="ignore"))
            return tmp_path
        except subprocess.TimeoutExpired as exc:
            print(f"[Warning] Video tail trimming timeout ({video_path}): {exc}")
            return None
        except Exception as exc:
            print(f"[Warning] Video tail trimming failed ({video_path}): {exc}")
    
    def _cap_video_duration(self, video_path, max_seconds=30.0):
        """Trim a video to max_seconds and return a temp path when needed."""
        try:
            probe = subprocess.run(
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
                capture_output=True,
                text=True,
                check=True,
                timeout=self.ffprobe_timeout,
            )
            duration = float(probe.stdout.strip())
            if duration <= max_seconds + 1e-3:
                return None
            tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_path = tmp_file.name
            tmp_file.close()
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-t",
                f"{max_seconds}",
                "-c",
                "copy",
                tmp_path,
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.ffmpeg_timeout,
            )
            if result.returncode != 0:
                os.remove(tmp_path)
                raise RuntimeError(result.stderr.decode("utf-8", errors="ignore"))
            return tmp_path
        except subprocess.TimeoutExpired as exc:
            print(f"[Warning] Video duration capping timeout ({video_path}): {exc}")
            return None
        except Exception as exc:
            print(f"[Warning] Video duration capping failed ({video_path}): {exc}")
            return None

    def __getitem__(self, idx, return_raw_media=False):
        """Load one sample with retry and optional tail trimming."""
        max_retry = 3
        last_error = None
        trimmed_path = None
        vid = self.video_ids[idx]
        title = self.id2title.get(vid)
        preprocessed_media = self._preprocessed_media(vid)
        if self.log_samples:
            if preprocessed_media:
                print(
                    f"[sample] idx={idx} id={vid} media=preprocessed "
                    f"frames={preprocessed_media['source']} audio={preprocessed_media.get('audio')}",
                    flush=True,
                )
            else:
                print(f"[sample] idx={idx} id={vid} media=video path={self.video_files[idx]}", flush=True)
        if preprocessed_media:
            try:
                inputs, media_audio_in_video = self._preprocess(
                    self.video_files[idx],
                    title,
                    return_raw_media=return_raw_media,
                    frame_paths=preprocessed_media["frames"],
                    audio_path=preprocessed_media.get("audio"),
                )
                return inputs, vid, preprocessed_media["source"], media_audio_in_video
            except Exception as exc:
                last_error = exc
                print(
                    f"[Warning] Preprocessed media failed; falling back to video "
                    f"(id={vid}, source={preprocessed_media['source']}): {exc}",
                    flush=True,
                )
        for _ in range(max_retry):
            video_path = self.video_files[idx]
            if trimmed_path != None:
                video_path = trimmed_path
            try:
                inputs, media_audio_in_video = self._preprocess(video_path, title, return_raw_media=return_raw_media)
                return inputs, vid, video_path, media_audio_in_video
            except Exception as exc:
                last_error = exc
                trimmed_path = self._trim_video(video_path)
                if trimmed_path:
                    try:
                        inputs, media_audio_in_video = self._preprocess(trimmed_path, title, return_raw_media=return_raw_media)
                        return inputs, vid, video_path, media_audio_in_video
                    except Exception as trim_exc:
                        last_error = trim_exc
                    finally:
                        if trimmed_path and os.path.exists(trimmed_path):
                            os.remove(trimmed_path)
        raise RuntimeError(f"Failed to load video after retries: id={vid}, path={video_path}, last_error={last_error}")

def collate_fn(batch):
    """Collate function for batch_size=1."""
    if len(batch) == 1:
        return batch[0][0], [batch[0][1]], [batch[0][2]], [batch[0][3]]
    
    inputs_list, video_ids, video_paths, audio_in_video_flags = zip(*batch)
    return inputs_list[0], list(video_ids), list(video_paths), list(audio_in_video_flags)

def load_id_title_list(json_path, real_only=False):
    """Load ids and titles from a JSON list while preserving order."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    id2title = {}
    id_list = []
    for item in data:
        vid = item.get("id")
        label = int(item.get("label", 0))
        if real_only and label != 0:
            continue
        title = item.get("title")
        id2title[vid] = title
        id_list.append(vid)
    return id_list, id2title

def train(
    model_path=None,
    video_dir=None,
    output_dir="checkpoint/reconstruct",
    batch_size=1,
    accumulation_steps=16,
    num_epochs=10,
    learning_rate=5e-5,
    use_audio_in_video=True,
    lora_layers=4,
    forward_layers=8,
    mask_text_ratio=0.3,
    mask_audio_ratio=0.3,
    mask_video_ratio=0.3,
    random_mask=False,
    num_workers=4,
    prefetch_factor=1,
    ffprobe_timeout=30,
    ffmpeg_timeout=60,
    log_samples=False,
    preprocessed_media_root=None,
    preprocessed_video_frames=8,
    prefer_preprocessed_media=True,
    pin_memory=True,
    train_json=None,
    val_json=None,
    dataset="fakesv",
    experiment_name="1",
    lora_r = 8,
    lora_alpha = 16,
    pretrained_model=None,
):
    """Train the reconstruction model."""
    configured = None
    try:
        from config import DATASETS
        configured = DATASETS.get(dataset)
    except Exception:
        configured = None
    if dataset == "fakesv":
        if video_dir is None:
            video_dir = configured.get("video_dir") if configured else None
        if train_json is None and configured:
            train_json = configured.get("train_title_transcript")
        if val_json is None and configured:
            val_json = configured.get("val_title_transcript")
    elif dataset == "fakett":
        if video_dir is None:
            video_dir = configured.get("video_dir") if configured else None
        if train_json is None and configured:
            train_json = configured.get("train_title_transcript")
        if val_json is None and configured:
            val_json = configured.get("val_title_transcript")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Use 'fakesv' or 'fakett'.")
    if video_dir is None:
        raise ValueError("video_dir is required; set DATASETS.<dataset>.video_dir in config.py")
    if model_path is None:
        raise ValueError("model_path is required; set MODELS.multimodal in config.py")

    output_dir = os.path.join(output_dir, dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Dataset configuration: {dataset}")
    print(f"  video_dir: {video_dir}")
    print(f"  train_json: {train_json}")
    print(f"  val_json: {val_json}")
    print(f"  output_dir: {output_dir}")
    print(f"{'='*60}\n")
    
    print("Loading model...")
    model = ReconstructionModel(
        model_path=model_path,
        lora_layers=lora_layers,
        forward_layers=forward_layers,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        use_train=True,
        lora_r = lora_r,
        lora_alpha = lora_alpha,
    )
    model.train()
    
    if pretrained_model is not None:
        print(f"Loading pretrained checkpoint: {pretrained_model}")
        if not os.path.exists(pretrained_model):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_model}")
        
        checkpoint = torch.load(pretrained_model, map_location="cpu")
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Loaded model weights from checkpoint format")
            if 'epoch' in checkpoint:
                print(f"  source epoch: {checkpoint['epoch']}")
            if 'val_loss' in checkpoint:
                print(f"  source validation loss: {checkpoint['val_loss']:.6f}")
        else:
            model.load_state_dict(checkpoint, strict=False)
            print("Loaded model weights from raw state_dict format")
    
    print("Loading train/validation splits...")
    train_ids, train_id2title = load_id_title_list(train_json, real_only=True)
    val_ids, val_id2title = load_id_title_list(val_json, real_only=True)
    print(f"Train samples: {len(train_ids)}, validation samples: {len(val_ids)}")
    
    print("Building datasets...")
    train_dataset = VideoDataset(
        video_dir,
        model.processor,
        use_audio_in_video=use_audio_in_video,
        id_list=train_ids,
        id2title=train_id2title,
        ffprobe_timeout=ffprobe_timeout,
        ffmpeg_timeout=ffmpeg_timeout,
        log_samples=log_samples,
        preprocessed_media_root=preprocessed_media_root,
        preprocessed_video_frames=preprocessed_video_frames,
        prefer_preprocessed_media=prefer_preprocessed_media,
    )
    val_dataset = VideoDataset(
        video_dir,
        model.processor,
        use_audio_in_video=use_audio_in_video,
        id_list=val_ids,
        id2title=val_id2title,
        ffprobe_timeout=ffprobe_timeout,
        ffmpeg_timeout=ffmpeg_timeout,
        log_samples=log_samples,
        preprocessed_media_root=preprocessed_media_root,
        preprocessed_video_frames=preprocessed_video_frames,
        prefer_preprocessed_media=prefer_preprocessed_media,
    )
    train_dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": False,
    }
    if num_workers > 0:
        train_dataloader_kwargs["prefetch_factor"] = prefetch_factor
    
    val_dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": False,
    }
    if val_dataloader_kwargs["num_workers"] > 0:
        val_dataloader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_dataset, **train_dataloader_kwargs)
    val_loader = DataLoader(val_dataset, **val_dataloader_kwargs)
    
    swanlab.init(
        project="reconstruct",
        experiment_name=experiment_name,
        config={
            "dataset": dataset,
            "batch_size": batch_size,
            "accumulation_steps": accumulation_steps,
            "effective_batch_size": batch_size * accumulation_steps,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "lora_layers": lora_layers,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "forward_layers": forward_layers,
            "mask_text_ratio": mask_text_ratio,
            "mask_audio_ratio": mask_audio_ratio,
            "mask_video_ratio": mask_video_ratio,
            "random_mask": random_mask,
        },
    )
    
    print("\n" + "="*60)
    print("Model parameter summary")
    print("="*60)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs *  len(train_loader)))
    
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Steps per epoch: {len(train_loader)}")
    print("="*60)
    print("Starting training...\n")
    
    global_step = 0
    best_path = os.path.join(output_dir, experiment_name)
    best_val_loss = float('inf')
    
    optimizer.zero_grad()
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}\n")
        
        model.train()
        epoch_loss = 0.0
        epoch_cos_sum = 0.0
        total_train_steps = len(train_loader)
        
        for batch_idx, (inputs, video_ids, video_paths, audio_in_video_flags) in enumerate(train_loader):
            batch_use_audio_in_video = bool(audio_in_video_flags[0])
            print(
                f"[train] start step {batch_idx + 1}/{total_train_steps} "
                f"id={video_ids[0]} path={video_paths[0]}",
                flush=True,
            )
            inputs = inputs.to(model.base_model.device).to(model.base_model.dtype)
            if "title_token_indices" in inputs:
                inputs["title_token_indices"] = inputs["title_token_indices"].to(model.base_model.device)
            
            modalities = [
                ('text', True, False, False),
                ('audio', False, True, False),
                ('video', False, False, True),
            ]
            
            modality_features = []
            modality_cosine_losses = []
            
            for _, mask_t, mask_a, mask_v in modalities:
                features, cosine_loss = model.extract_feature(
                    **inputs,
                    use_audio_in_video=batch_use_audio_in_video,
                    mask_text=mask_t,
                    mask_text_ratio=mask_text_ratio,
                    mask_audio=mask_a,
                    mask_audio_ratio=mask_audio_ratio,
                    mask_video=mask_v,
                    mask_video_ratio=mask_video_ratio,
                    random_mask=random_mask,
                )
                modality_features.append(features)
                modality_cosine_losses.append(cosine_loss)
            
            fused_feature = torch.cat(modality_features, dim=-1)  # (batch_size, hidden_size * 3)
            if torch.isnan(fused_feature).any() or torch.isinf(fused_feature).any():
                raise ValueError(f"[train] Fused feature contains NaN/Inf at batch={batch_idx}")
            cosine_loss = torch.stack(modality_cosine_losses).mean()
            loss = cosine_loss
            
            loss = loss / accumulation_steps
            
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == total_train_steps:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            loss_value = loss.item() * accumulation_steps
            cosine_loss_value = float(cosine_loss.item())
            
            del loss, cosine_loss, fused_feature
            
            epoch_loss += loss_value
            epoch_cos_sum += cosine_loss_value
            
            is_update_step = (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == total_train_steps
            
            print(
                f"[train] step {batch_idx + 1}/{total_train_steps} "
                f"loss={loss_value:.6f} cos={cosine_loss_value:.6f} "
                f"{'[UPDATE]' if is_update_step else '[ACCUM]'}"
            , flush=True)
            
            global_step += 1
            swanlab.log({
                "train_step_loss": loss_value,
                "train_step_cosine_loss": cosine_loss_value,
                "train_step_lr": scheduler.get_last_lr()[0],
            }, step=global_step)

        denom = len(train_loader)
        avg_loss = epoch_loss / denom 
        avg_cos = epoch_cos_sum / denom 
        
        print(f"Epoch {epoch+1} complete, average train loss: {avg_loss:.6f}")
        print(f"  Cosine: {avg_cos:.6f}")
        
        swanlab.log({
            "train_epoch_loss": avg_loss,
            "train_epoch_cosine_loss": avg_cos,
        })
        
        model.eval()
        val_loss, val_cosine_total = 0.0, 0.0
        with torch.no_grad():
            total_val_steps = len(val_loader)
            for batch_idx, (inputs, video_ids, video_paths, audio_in_video_flags) in enumerate(val_loader):
                batch_use_audio_in_video = bool(audio_in_video_flags[0])
                print(
                    f"[val] start step {batch_idx + 1}/{total_val_steps} "
                    f"id={video_ids[0]} path={video_paths[0]}",
                    flush=True,
                )
                inputs = inputs.to(model.base_model.device).to(model.base_model.dtype)
                if "title_token_indices" in inputs:
                    inputs["title_token_indices"] = inputs["title_token_indices"].to(model.base_model.device)
                
                modalities = [
                    ('text', True, False, False),
                    ('audio', False, True, False),
                    ('video', False, False, True),
                ]
                
                modality_features = []
                modality_cosine_losses = []
                for _, mask_t, mask_a, mask_v in modalities:
                    features, cosine_loss = model.extract_feature(
                        **inputs,
                        use_audio_in_video=batch_use_audio_in_video,
                        mask_text=mask_t,
                        mask_text_ratio=mask_text_ratio,
                        mask_audio=mask_a,
                        mask_audio_ratio=mask_audio_ratio,
                        mask_video=mask_v,
                        mask_video_ratio=mask_video_ratio,
                        random_mask=random_mask,
                    )
                    modality_features.append(features)
                    modality_cosine_losses.append(cosine_loss)

                fused_feature = torch.cat(modality_features, dim=-1)  # (batch_size, hidden_size * 6)
                cosine_loss_value = torch.stack(modality_cosine_losses).mean()
                loss = cosine_loss_value

                loss_value = float(loss.item())
                cosine_loss_step = float(cosine_loss_value.item())
                
                val_loss += loss_value
                val_cosine_total += cosine_loss_step

                del fused_feature, cosine_loss_value
                
                print(
                    f"[val] step {batch_idx + 1}/{total_val_steps} "
                    f"loss={loss_value:.6f} cos={cosine_loss_step:.6f}"
                , flush=True)
        
        num_samples = len(val_loader)
        val_avg_loss = val_loss / num_samples
        val_avg_cos = val_cosine_total / num_samples
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
        
        swanlab.log({
            "val_loss_epoch": val_avg_loss,
            "val_cosine_epoch": val_avg_cos,
            "epoch": epoch,
        })
        
        print("Validation:")
        print(f"  average loss: {val_avg_loss:.6f}, cosine: {val_avg_cos:.6f}")
    
    os.makedirs(best_path, exist_ok=True)
    final_epoch = num_epochs - 1
    final_model_path = os.path.join(best_path, f"{final_epoch}_model.bin")
    torch.save({
        'epoch': final_epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': best_val_loss,
    }, final_model_path)
    print(f"\nTraining finished. Saved the final-epoch model to: {final_model_path}")
    print(f"  epoch: {final_epoch+1}, best validation loss: {best_val_loss:.4f}")
    
    swanlab.log({"best_val_loss": best_val_loss})
    swanlab.finish()

        
