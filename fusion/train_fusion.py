import os
import sys
from datetime import datetime
import builtins

# Allow external overrides.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["FFMPEG_LOGLEVEL"] = "error"

# Add module paths.
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'reconstruct'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'distill'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import random
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
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from fusion_model import FusionModel
# Reuse the video preprocessing path from reconstruction.
from reconstruct import VideoDataset


def print(*args, **kwargs):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    builtins.print(f"[{ts}]", *args, **kwargs)


class FusionDataset(VideoDataset):
    """Dataset that serves both video inputs and text inputs for fusion training."""
    
    def __init__(
        self,
        video_dir,
        multimodal_processor,
        id_list,
        id2label,
        id2title,
        id2transcript,
        use_audio_in_video=True,
        ffprobe_timeout=30,
        ffmpeg_timeout=60,
        log_samples=False,
        preprocessed_media_root=None,
        preprocessed_video_frames=8,
        prefer_preprocessed_media=True,
    ):
        super().__init__(
            video_dir=video_dir,
            processor=multimodal_processor,
            use_audio_in_video=use_audio_in_video,
            id_list=id_list,
            id2title=id2title,
            ffprobe_timeout=ffprobe_timeout,
            ffmpeg_timeout=ffmpeg_timeout,
            log_samples=log_samples,
            preprocessed_media_root=preprocessed_media_root,
            preprocessed_video_frames=preprocessed_video_frames,
            prefer_preprocessed_media=prefer_preprocessed_media,
        )
        self.id2label = id2label
        self.id2transcript = id2transcript
    
    def __getitem__(self, idx):
        """Return multimodal inputs, distilled-text input, label, and sample id."""
        multimodal_inputs, _, _, audio_in_video = super().__getitem__(idx, return_raw_media=False)
        vid = self.video_ids[idx]
        title = self.id2title.get(vid)
        transcript = self.id2transcript.get(vid)
        label = self.id2label.get(vid, None)
        text = f"*Title*: {title}. *Transcript*: {transcript}"
        return multimodal_inputs, text, label, vid, audio_in_video


def collate_fn(batch):
    """Batch collation for batch_size=1."""
    if len(batch) == 1:
        multimodal_inputs, text, label, vid, audio_in_video = batch[0]
        return multimodal_inputs, [text], torch.tensor([label]), [vid], [audio_in_video]

def compute_metrics(predictions, labels):
    """
    Compute detailed binary classification metrics.
    """
    accuracy = (predictions == labels).mean()
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    fake_precision = precision_score(labels, predictions, pos_label=1, zero_division=0)
    fake_recall = recall_score(labels, predictions, pos_label=1, zero_division=0)
    fake_f1 = f1_score(labels, predictions, pos_label=1, zero_division=0)
    real_precision = precision_score(labels, predictions, pos_label=0, zero_division=0)
    real_recall = recall_score(labels, predictions, pos_label=0, zero_division=0)
    real_f1 = f1_score(labels, predictions, pos_label=0, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'fake_precision': fake_precision,
        'fake_recall': fake_recall,
        'fake_f1': fake_f1,
        'real_precision': real_precision,
        'real_recall': real_recall,
        'real_f1': real_f1,
    }

def load_data(json_path):
    """Load id, label, title, and transcript from a JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    id_list = []
    id2label = {}
    id2title = {}
    id2transcript = {}
    
    for item in data:
        vid = item.get('id')
        label = int(item.get('label'))
        title = item.get('title')
        transcript = item.get('transcript', '')
        
        id_list.append(vid)
        id2label[vid] = label
        id2title[vid] = title
        id2transcript[vid] = transcript
    
    return id_list, id2label, id2title, id2transcript


def train_fusion(
    # Model paths
    multimodal_model_path: str = None,
    multimodal_checkpoint_path: str = None,
    distill_model_path: str = None,
    distill_checkpoint_path: str = None,
    fusion_checkpoint_path: str = None,
    
    dataset: str = "fakesv",
    video_dir: str = None,
    train_json: str = None,
    val_json: str = None,
    test_json: str = None,

    num_epochs: int = 15,
    batch_size: int = 1,
    accumulation_steps: int = 16,
    learning_rate: float = 5e-5,
    num_workers: int = 4,
    prefetch_factor: int = 1,
    ffprobe_timeout: int = 30,
    ffmpeg_timeout: int = 60,
    log_samples: bool = False,
    preprocessed_media_root: str = None,
    preprocessed_video_frames: int = 8,
    prefer_preprocessed_media: bool = True,
    
    distill_top_k: int = 3,
    
    use_audio_in_video: bool = True,
    mask_text_ratio: float = 0.3,
    mask_audio_ratio: float = 0.3,
    mask_video_ratio: float = 0.3,
    multimodal_lora_layers: int = 8,
    multimodal_forward_layers: int = 8,
    
    output_dir: str = "checkpoint/fusion",
    experiment_name: str = "main_fusion",

    project: str = "pcdd_fusion",
):
    """Train the final fusion model."""
    if dataset not in ["fakesv", "fakett"]:
        raise ValueError(f"Unsupported dataset: {dataset}. Use 'fakesv' or 'fakett'.")
    if not multimodal_model_path or not distill_model_path:
        raise ValueError(
            "multimodal_model_path and distill_model_path are required; "
            "set MODELS.multimodal and MODELS.distill in config.py"
        )

    if video_dir is None:
        try:
            from config import DATASETS
            video_dir = DATASETS[dataset]["video_dir"]
        except Exception as exc:
            raise ValueError("video_dir is required when config.py is unavailable") from exc

    if train_json is None or val_json is None or test_json is None:
        try:
            from config import DATASETS
            ds_cfg = DATASETS[dataset]
            if train_json is None:
                train_json = ds_cfg.get("train_title_transcript")
            if val_json is None:
                val_json = ds_cfg.get("val_title_transcript")
            if test_json is None:
                test_json = ds_cfg.get("test_title_transcript")
        except Exception:
            pass

    print("\nUsing dataset configuration:")
    print(f"  dataset    = {dataset}")
    print(f"  video_dir  = {video_dir}")
    print(f"  train_json = {train_json}")
    print(f"  val_json   = {val_json}")

    output_dir = os.path.join(output_dir, dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    swanlab.init(
        project=project,
        experiment_name=experiment_name,
        config={
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "accumulation_steps": accumulation_steps,
            "effective_batch_size": batch_size * accumulation_steps,
            "learning_rate": learning_rate,
            "distill_top_k": distill_top_k,
            "multimodal_lora_layers": multimodal_lora_layers,
            "multimodal_forward_layers": multimodal_forward_layers,
            "multimodal_checkpoint": multimodal_checkpoint_path,
            "distill_checkpoint": distill_checkpoint_path,
            "fusion_checkpoint": fusion_checkpoint_path,
        }
    )
    
    print("\n" + "="*80)
    print("Initializing fusion model")
    print("="*80)
    model = FusionModel(
        multimodal_model_path=multimodal_model_path,
        multimodal_checkpoint_path=multimodal_checkpoint_path,
        multimodal_lora_layers=multimodal_lora_layers,
        multimodal_forward_layers=multimodal_forward_layers,
        distill_model_path=distill_model_path,
        distill_checkpoint_path=distill_checkpoint_path,
        distill_top_k=distill_top_k,
        dataset=dataset
    )
    
    if fusion_checkpoint_path is not None and os.path.exists(fusion_checkpoint_path):
        print(f"\nLoading fusion checkpoint: {fusion_checkpoint_path}")
        model.load_fusion_weights(fusion_checkpoint_path)
        print("Loaded fusion checkpoint and will continue training")
    else:
        if fusion_checkpoint_path is not None:
            print(f"\nWarning: fusion checkpoint path does not exist: {fusion_checkpoint_path}")
            print("Using the current initialized fusion weights")
        else:
            print("\nUsing the current initialized fusion weights")
    
    print("\n" + "="*80)
    print("Loading data")
    print("="*80)
    train_ids, train_id2label, train_id2title, train_id2transcript = load_data(train_json)
    val_ids, val_id2label, val_id2title, val_id2transcript = load_data(val_json)
    print(f"Train samples: {len(train_ids)}, validation samples: {len(val_ids)}")
    
    train_dataset = FusionDataset(
        video_dir=video_dir,
        multimodal_processor=model.multimodal_model.processor,
        id_list=train_ids,
        id2label=train_id2label,
        id2title=train_id2title,
        id2transcript=train_id2transcript,
        use_audio_in_video=use_audio_in_video,
        ffprobe_timeout=ffprobe_timeout,
        ffmpeg_timeout=ffmpeg_timeout,
        log_samples=log_samples,
        preprocessed_media_root=preprocessed_media_root,
        preprocessed_video_frames=preprocessed_video_frames,
        prefer_preprocessed_media=prefer_preprocessed_media,
    )
    
    val_dataset = FusionDataset(
        video_dir=video_dir,
        multimodal_processor=model.multimodal_model.processor,
        id_list=val_ids,
        id2label=val_id2label,
        id2title=val_id2title,
        id2transcript=val_id2transcript,
        use_audio_in_video=use_audio_in_video,
        ffprobe_timeout=ffprobe_timeout,
        ffmpeg_timeout=ffmpeg_timeout,
        log_samples=log_samples,
        preprocessed_media_root=preprocessed_media_root,
        preprocessed_video_frames=preprocessed_video_frames,
        prefer_preprocessed_media=prefer_preprocessed_media,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=False,
    )
    
    fusion_params = list(model.projection_text.parameters()) + \
                   list(model.self_attn_A.parameters()) + \
                   list(model.self_attn_B.parameters()) + \
                   list(model.self_attn_norm_A.parameters()) + \
                   list(model.self_attn_norm_B.parameters()) + \
                   list(model.cross_attn_A.parameters()) + \
                   list(model.cross_attn_B.parameters()) + \
                   list(model.cross_attn_norm_A.parameters()) + \
                   list(model.cross_attn_norm_B.parameters()) + \
                   list(model.gate_mlp_A.parameters()) + \
                   list(model.gate_mlp_B.parameters()) + \
                   list(model.final_classifier.parameters())
    
    optimizer = torch.optim.AdamW(fusion_params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, num_epochs * len(train_loader))
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"  - Fusion-layer parameters: {sum(p.numel() for p in fusion_params):,} (lr={learning_rate})")
    print(f"Trainable ratio: {trainable_params / total_params * 100:.2f}%")
    print(f"Steps per epoch: {len(train_loader)}")
    
    global_step = 0
    best_val_acc = -1.0
    val_no_improve = 0
    lr_decay_no_improve = 0
    experiment_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    best_path = os.path.join(experiment_dir, "best.pth")
    
    optimizer.zero_grad()
    
    for epoch in range(num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*80}\n")

        print("Refreshing shuffled training loader...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=False,
        )
        
        model.train()
        
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        total_train_steps = len(train_loader)
        
        accum_correct = 0
        accum_total = 0
        accum_loss = 0.0
        train_all_preds = []
        train_all_labels = []
        
        for batch_idx, (multimodal_inputs, texts, labels, vids, audio_in_video_flags) in enumerate(train_loader):
            batch_use_audio_in_video = bool(audio_in_video_flags[0])
            multimodal_inputs = multimodal_inputs.to(model.device).to(torch.bfloat16)
            labels = labels.to(model.device).view(-1).to(torch.float32)

            logits = model(
                multimodal_inputs=multimodal_inputs,
                texts=texts,
                use_audio_in_video=batch_use_audio_in_video,
                mask_text_ratio=mask_text_ratio,
                mask_audio_ratio=mask_audio_ratio,
                mask_video_ratio=mask_video_ratio,
            )
            
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == total_train_steps:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            loss_value = loss.item() * accumulation_steps
            preds = (torch.sigmoid(logits) >= 0.5).to(labels.dtype)
            correct = int((preds == labels).sum().item())
            epoch_correct += correct
            epoch_total += int(labels.numel())
            step_acc = correct / max(1, labels.numel())
            
            epoch_loss += loss_value
            
            train_all_preds.extend(preds.cpu().numpy().tolist())
            train_all_labels.extend(labels.cpu().numpy().tolist())
            accum_correct += correct
            accum_total += int(labels.numel())
            accum_loss += loss_value
            is_update_step = (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == total_train_steps

            print(
                f"[train] step {batch_idx + 1}/{total_train_steps} "
                f"loss={loss_value:.6f} acc={step_acc:.4f} "
                f"logits={logits.detach().mean().item():.4f} "
                f"sig={torch.sigmoid(logits).detach().mean().item():.4f} "
                f"{'[UPDATE]' if is_update_step else '[ACCUM]'}"
            )

            if is_update_step:
                global_step += 1
                accum_acc = accum_correct / max(1, accum_total)
                accum_avg_loss = accum_loss / accumulation_steps if (batch_idx + 1) % accumulation_steps == 0 else accum_loss / ((batch_idx + 1) % accumulation_steps)
                
                swanlab.log({
                    "train_step_loss": accum_avg_loss,
                    "train_step_accuracy": accum_acc,
                    "train_step_lr": scheduler.get_last_lr()[0],
                    "global_step": global_step,
                }, step=global_step)
                
                accum_correct = 0
                accum_total = 0
                accum_loss = 0.0

        avg_loss = epoch_loss / len(train_loader)
        train_all_preds = np.array(train_all_preds)
        train_all_labels = np.array(train_all_labels)
        train_metrics = compute_metrics(train_all_preds, train_all_labels)
        
        print(f"Epoch {epoch+1} training complete:")
        print(f"  Loss: {avg_loss:.6f}")
        print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  Macro F1: {train_metrics['macro_f1']:.4f}")
        print(f"  Fake - Precision: {train_metrics['fake_precision']:.4f}, Recall: {train_metrics['fake_recall']:.4f}, F1: {train_metrics['fake_f1']:.4f}")
        print(f"  Real - Precision: {train_metrics['real_precision']:.4f}, Recall: {train_metrics['real_recall']:.4f}, F1: {train_metrics['real_f1']:.4f}")
        
        swanlab.log({
            "train_epoch_loss": avg_loss,
            "train_epoch_accuracy": train_metrics['accuracy'],
            "train_epoch_macro_f1": train_metrics['macro_f1'],
            "train_epoch_fake_precision": train_metrics['fake_precision'],
            "train_epoch_fake_recall": train_metrics['fake_recall'],
            "train_epoch_fake_f1": train_metrics['fake_f1'],
            "train_epoch_real_precision": train_metrics['real_precision'],
            "train_epoch_real_recall": train_metrics['real_recall'],
            "train_epoch_real_f1": train_metrics['real_f1'],
        })
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_all_preds = []
        val_all_labels = []
        
        with torch.no_grad():
            total_val_steps = len(val_loader)
            for batch_idx, (multimodal_inputs, texts, labels, vids, audio_in_video_flags) in enumerate(val_loader):
                batch_use_audio_in_video = bool(audio_in_video_flags[0])
                multimodal_inputs = multimodal_inputs.to(model.device).to(torch.bfloat16)
                labels = labels.to(model.device).view(-1).to(torch.float32)

                logits = model(
                    multimodal_inputs=multimodal_inputs,
                    texts=texts,
                    use_audio_in_video=batch_use_audio_in_video,
                    mask_text_ratio=mask_text_ratio,
                    mask_audio_ratio=mask_audio_ratio,
                    mask_video_ratio=mask_video_ratio,
                )
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                preds = (torch.sigmoid(logits) >= 0.5).to(labels.dtype)
                correct = int((preds == labels).sum().item())
                val_correct += correct
                val_total += int(labels.numel())
                step_acc = correct / max(1, labels.numel())
                
                val_loss += loss.item()
                
                val_all_preds.extend(preds.cpu().numpy().tolist())
                val_all_labels.extend(labels.cpu().numpy().tolist())
                print(
                    f"[val] step {batch_idx + 1}/{total_val_steps} "
                    f"loss={loss.item():.6f} acc={step_acc:.4f} "
                    f"logits={logits.detach().mean().item():.4f} sig={torch.sigmoid(logits).detach().mean().item():.4f}"
                )

        val_avg_loss = val_loss / len(val_loader)
        val_all_preds = np.array(val_all_preds)
        val_all_labels = np.array(val_all_labels)
        val_metrics = compute_metrics(val_all_preds, val_all_labels)
        
        swanlab.log({
            "val_loss_epoch": val_avg_loss,
            "val_acc_epoch": val_metrics['accuracy'],
            "val_macro_f1": val_metrics['macro_f1'],
            "val_fake_precision": val_metrics['fake_precision'],
            "val_fake_recall": val_metrics['fake_recall'],
            "val_fake_f1": val_metrics['fake_f1'],
            "val_real_precision": val_metrics['real_precision'],
            "val_real_recall": val_metrics['real_recall'],
            "val_real_f1": val_metrics['real_f1'],
            "epoch": epoch,
        })
        
        print("Validation:")
        print(f"  Loss: {val_avg_loss:.6f}")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"  Fake - Precision: {val_metrics['fake_precision']:.4f}, Recall: {val_metrics['fake_recall']:.4f}, F1: {val_metrics['fake_f1']:.4f}")
        print(f"  Real - Precision: {val_metrics['real_precision']:.4f}, Recall: {val_metrics['real_recall']:.4f}, F1: {val_metrics['real_f1']:.4f}")
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            val_no_improve = 0
            lr_decay_no_improve = 0
            model.save_fusion_weights(best_path)
            print(f"Saved best fusion checkpoint to: {best_path} (val_accuracy={best_val_acc:.4f})")
        else:
            val_no_improve += 1
            lr_decay_no_improve += 1
            print(f"No validation improvement for {val_no_improve} epoch(s)")

        if lr_decay_no_improve >= 2:
            for param_group in optimizer.param_groups:
                old_lr, new_lr = param_group['lr'], param_group['lr'] * 0.5
                param_group['lr'] = new_lr
                print(f"Learning rate decay: {old_lr:.2e} -> {new_lr:.2e}")
            lr_decay_no_improve = 0

        if val_no_improve >= 5:
            print(f"Early stopping at epoch {epoch+1} after 5 epochs without validation improvement")
            break

    print(f"\nTraining finished. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best fusion checkpoint: {best_path}")
    swanlab.log({"best_val_accuracy": best_val_acc})

    # ---- Test-set evaluation ----
    if test_json is not None and os.path.exists(test_json):
        print(f"\n{'='*80}")
        print("Evaluating on test set")
        print(f"{'='*80}")
        test_ids, test_id2label, test_id2title, test_id2transcript = load_data(test_json)
        print(f"Test samples: {len(test_ids)}")

        test_dataset = FusionDataset(
            video_dir=video_dir,
            multimodal_processor=model.multimodal_model.processor,
            id_list=test_ids,
            id2label=test_id2label,
            id2title=test_id2title,
            id2transcript=test_id2transcript,
            use_audio_in_video=use_audio_in_video,
            ffprobe_timeout=ffprobe_timeout,
            ffmpeg_timeout=ffmpeg_timeout,
            log_samples=False,
            preprocessed_media_root=preprocessed_media_root,
            preprocessed_video_frames=preprocessed_video_frames,
            prefer_preprocessed_media=prefer_preprocessed_media,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0, pin_memory=True,
        )

        model.load_fusion_weights(best_path)
        model.eval()
        test_all_preds, test_all_labels = [], []
        with torch.no_grad():
            for multimodal_inputs, texts, labels, vids, audio_in_video_flags in test_loader:
                batch_use_audio_in_video = bool(audio_in_video_flags[0])
                multimodal_inputs = multimodal_inputs.to(model.device).to(torch.bfloat16)
                labels = labels.to(model.device).view(-1).to(torch.float32)
                logits = model(
                    multimodal_inputs=multimodal_inputs, texts=texts,
                    use_audio_in_video=batch_use_audio_in_video,
                    mask_text_ratio=mask_text_ratio,
                    mask_audio_ratio=mask_audio_ratio,
                    mask_video_ratio=mask_video_ratio,
                )
                preds = (torch.sigmoid(logits) >= 0.5).to(labels.dtype)
                test_all_preds.extend(preds.cpu().numpy().tolist())
                test_all_labels.extend(labels.cpu().numpy().tolist())

        test_metrics = compute_metrics(
            np.array(test_all_preds), np.array(test_all_labels)
        )
        print(f"Test Accuracy:    {test_metrics['accuracy']:.4f}")
        print(f"Test Macro F1:    {test_metrics['macro_f1']:.4f}")
        print(f"Fake - P: {test_metrics['fake_precision']:.4f}  R: {test_metrics['fake_recall']:.4f}  F1: {test_metrics['fake_f1']:.4f}")
        print(f"Real - P: {test_metrics['real_precision']:.4f}  R: {test_metrics['real_recall']:.4f}  F1: {test_metrics['real_f1']:.4f}")
        swanlab.log({
            "test_accuracy": test_metrics['accuracy'],
            "test_macro_f1": test_metrics['macro_f1'],
            "test_fake_f1": test_metrics['fake_f1'],
            "test_real_f1": test_metrics['real_f1'],
        })
    elif test_json is not None:
        print(f"\nWarning: test_json not found at {test_json}, skipping test evaluation")

    swanlab.finish()
