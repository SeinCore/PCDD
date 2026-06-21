import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'reconstruct'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'distill'))

import json
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from reconstruct import ReconstructionModel
from distill import Student, _normalize_state_dict

class FusionModel(nn.Module):
    """Fusion model for multimodal discrepancy features and distilled text features."""
    
    def __init__(
        self,
        # Multimodal module
        multimodal_model_path: Optional[str] = None,
        multimodal_checkpoint_path: str = None,
        multimodal_lora_layers: int = 8,
        multimodal_forward_layers: int = 8,
        multimodal_lora_r: int = 8,
        multimodal_lora_alpha: int = 16,

        # Distillation module
        distill_model_path: Optional[str] = None,
        distill_checkpoint_path: str = None,
        distill_lora_r: int = 8,
        distill_lora_alpha: int = 16,
        distill_top_k: int = 3,

        # Data paths
        train_title_transcript_path: str = "",
        val_title_transcript_path: str = "",
        
        # Fusion module
        dropout: float = 0.0,
        device: str = "cuda",
        dataset: str = "fakesv",

    ):
        super(FusionModel, self).__init__()
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print("\n" + "="*80)
        print("Loading multimodal module (ReconstructionModel)")
        print("="*80)
        self.multimodal_model = ReconstructionModel(
            model_path=multimodal_model_path,
            lora_layers=multimodal_lora_layers,
            forward_layers=multimodal_forward_layers,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
            use_train=True,
            lora_r=multimodal_lora_r,
            lora_alpha=multimodal_lora_alpha,
        )
        
        if multimodal_checkpoint_path and os.path.exists(multimodal_checkpoint_path):
            print(f"Loading multimodal checkpoint: {multimodal_checkpoint_path}")
            state = torch.load(multimodal_checkpoint_path, map_location=self.device)
            if 'model_state_dict' in state:
                state = state['model_state_dict']
            self.multimodal_model.load_state_dict(state, strict=False)
            print("Loaded multimodal checkpoint")
        else:
            print("No multimodal checkpoint provided; using current initialization")
        
        trainable_params_count = sum(p.numel() for p in self.multimodal_model.parameters() if p.requires_grad)
        frozen_params_count = sum(p.numel() for p in self.multimodal_model.parameters() if not p.requires_grad)
        print("Configured multimodal module")
        print(f"  - Trainable parameters: {trainable_params_count:,}")
        print(f"  - Frozen parameters: {frozen_params_count:,}")
        
        self.num_multimodal_tokens = 6
        self.multimodal_token_dim = self.multimodal_model.hidden_size
        
        print("\n" + "="*80)
        print("Loading text-distillation module (Student)")
        print("="*80)
        self.distill_model = Student(
            model_path=distill_model_path,
            title_transcript_path=train_title_transcript_path,
            val_title_transcript_path=val_title_transcript_path,
            use_lora=True,
            lora_r=distill_lora_r,
            lora_alpha=distill_lora_alpha,
            dataset=dataset
        )
        
        self.distill_model.stage3_top_k = distill_top_k
        self.distill_model.stage3_score_head = torch.nn.Linear(
            self.distill_model.qwen_config.hidden_size * (1 + distill_top_k),
            self.distill_model.num_labels
        ).to(self.distill_model.device, dtype=torch.bfloat16)
        print(f"Configured distill_top_k={distill_top_k} and rebuilt the stage-3 score head")
        
        if distill_checkpoint_path and os.path.exists(distill_checkpoint_path):
            print(f"Loading text-distillation checkpoint: {distill_checkpoint_path}")
            state = torch.load(distill_checkpoint_path, map_location=self.device)
            self.distill_model.load_state_dict(_normalize_state_dict(state), strict=False)
            print("Loaded text-distillation checkpoint")
        else:
            print("No text-distillation checkpoint provided; using current initialization")
        
        if hasattr(self.distill_model, 'teacher_qwen'):
            delattr(self.distill_model, 'teacher_qwen')
            print("Dropped teacher_qwen from the fusion graph")
        
        trainable_params_count = sum(p.numel() for p in self.distill_model.parameters() if p.requires_grad)
        frozen_params_count = sum(p.numel() for p in self.distill_model.parameters() if not p.requires_grad)
        print("Configured text-distillation module")
        print(f"  - Trainable parameters: {trainable_params_count:,}")
        print(f"  - Frozen parameters: {frozen_params_count:,}")
        
        self.num_text_tokens = 1 + distill_top_k
        self.text_token_dim = self.distill_model.qwen_config.hidden_size
        
        print("\n" + "="*80)
        print("Building fusion layers")
        print("="*80)
        print(f"Multimodal: {self.num_multimodal_tokens} tokens x {self.multimodal_token_dim}")
        print(f"Text: {self.num_text_tokens} tokens x {self.text_token_dim}")
        
        self.projection_text = nn.Linear(self.text_token_dim, self.multimodal_token_dim)
        
        self.token_dim = self.multimodal_token_dim
        
        self.self_attn_A = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.self_attn_B = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.self_attn_norm_A = nn.LayerNorm(self.token_dim)
        self.self_attn_norm_B = nn.LayerNorm(self.token_dim)
        
        self.cross_attn_A = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_B = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.cross_attn_norm_A = nn.LayerNorm(self.token_dim)
        self.cross_attn_norm_B = nn.LayerNorm(self.token_dim)
        
        self.gate_mlp_A = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.token_dim // 2, 1)
        )
        self.gate_mlp_B = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.token_dim // 2, 1)
        )
        self.gate_temperature = 1.0
        
        self.dropout = nn.Dropout(dropout)
        self.final_classifier = nn.Sequential(
            nn.Linear(self.token_dim * 2, self.token_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.token_dim // 2, 1)
        )
        for layer in self.final_classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        self.projection_text = self.projection_text.to(self.device).to(torch.bfloat16)
        self.self_attn_A = self.self_attn_A.to(self.device).to(torch.bfloat16)
        self.self_attn_B = self.self_attn_B.to(self.device).to(torch.bfloat16)
        self.self_attn_norm_A = self.self_attn_norm_A.to(self.device).to(torch.bfloat16)
        self.self_attn_norm_B = self.self_attn_norm_B.to(self.device).to(torch.bfloat16)
        self.cross_attn_A = self.cross_attn_A.to(self.device).to(torch.bfloat16)
        self.cross_attn_B = self.cross_attn_B.to(self.device).to(torch.bfloat16)
        self.cross_attn_norm_A = self.cross_attn_norm_A.to(self.device).to(torch.bfloat16)
        self.cross_attn_norm_B = self.cross_attn_norm_B.to(self.device).to(torch.bfloat16)
        self.gate_mlp_A = self.gate_mlp_A.to(self.device).to(torch.bfloat16)
        self.gate_mlp_B = self.gate_mlp_B.to(self.device).to(torch.bfloat16)
        self.dropout = self.dropout.to(self.device)
        self.final_classifier = self.final_classifier.to(self.device).to(torch.bfloat16)
        
        nn.init.xavier_uniform_(self.projection_text.weight)
        nn.init.zeros_(self.projection_text.bias)

        for layer in self.gate_mlp_A:
            if isinstance(layer, nn.Linear):
                if layer.out_features == 1:
                    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(layer.bias)
                else:
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        for layer in self.gate_mlp_B:
            if isinstance(layer, nn.Linear):
                if layer.out_features == 1:
                    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(layer.bias)
                else:
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
        print("Fusion layers are ready")
        self._print_trainable_params()
    
    def _print_trainable_params(self):
        """Print parameter statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("\n" + "="*80)
        print("Parameter summary")
        print("="*80)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable ratio: {trainable_params / total_params * 100:.2f}%")
        print("="*80 + "\n")
    
    def extract_multimodal_features(
        self,
        inputs,
        use_audio_in_video: bool = True,
        mask_text_ratio: float = 0.3,
        mask_audio_ratio: float = 0.3,
        mask_video_ratio: float = 0.3,
    ):
        """Extract six multimodal discrepancy tokens.

        The multimodal backbone is frozen during fusion training — only the fusion
        layers (cross-attn, gate, classifier) receive gradients.
        """
        modalities = [
            ('text', True, False, False),
            ('audio', False, True, False),
            ('video', False, False, True),
        ]

        modality_features = []

        with torch.no_grad():
            for modality_name, mask_t, mask_a, mask_v in modalities:
                features, _ = self.multimodal_model.extract_feature(
                    **inputs,
                    use_audio_in_video=use_audio_in_video,
                    mask_text=mask_t,
                    mask_text_ratio=mask_text_ratio,
                    mask_audio=mask_a,
                    mask_audio_ratio=mask_audio_ratio,
                    mask_video=mask_v,
                    mask_video_ratio=mask_video_ratio,
                )
                batch_size = features.shape[0]
                features_tokens = features.view(batch_size, 2, self.multimodal_token_dim)
                modality_features.append(features_tokens)

        multimodal_tokens = torch.cat(modality_features, dim=1)
        return multimodal_tokens
    
    def extract_text_features(
        self,
        texts,
        max_length=512,
    ):
        """Extract the CLS token plus top-k sentence tokens from the distilled text encoder.

        The distill backbone is frozen during fusion training — only the fusion
        layers receive gradients.
        """
        encoded = self.distill_model.tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        offsets = encoded['offset_mapping']

        with torch.no_grad():
            outputs = self.distill_model.qwen(input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state
            concat_features = self.distill_model._stage3_build_concat_features_train(
                texts, last_hidden, attention_mask, offsets, self.distill_model.stage3_top_k
            )

        batch_size = concat_features.shape[0]
        num_tokens = self.num_text_tokens
        text_tokens = concat_features.view(batch_size, num_tokens, self.text_token_dim)
        return text_tokens
    
    def forward(
        self,
        multimodal_inputs,
        texts,
        use_audio_in_video: bool = True,
        mask_text_ratio: float = 0.3,
        mask_audio_ratio: float = 0.3,
        mask_video_ratio: float = 0.3,
        text_max_length: int = 512,
    ):
        """Forward pass for the final fusion classifier."""
        A_tokens = self.extract_multimodal_features(
            multimodal_inputs,
            use_audio_in_video=use_audio_in_video,
            mask_text_ratio=mask_text_ratio,
            mask_audio_ratio=mask_audio_ratio,
            mask_video_ratio=mask_video_ratio,
        )
        B_tokens = self.extract_text_features(
            texts,
            max_length=text_max_length,
        )
        B_tokens = self.projection_text(B_tokens)

        A_norm = self.self_attn_norm_A(A_tokens)
        A_self, _ = self.self_attn_A(
            query=A_norm,
            key=A_norm,
            value=A_norm
        )
        A_self = A_self + A_tokens

        B_norm = self.self_attn_norm_B(B_tokens)
        B_self, _ = self.self_attn_B(
            query=B_norm,
            key=B_norm,
            value=B_norm
        )
        B_self = B_self + B_tokens

        A_norm = self.cross_attn_norm_A(A_self)
        B_norm = self.cross_attn_norm_B(B_self)
        A_prime, _ = self.cross_attn_A(
            query=A_norm,
            key=B_norm,
            value=B_norm
        )
        A_prime = A_prime + A_self

        B_prime, _ = self.cross_attn_B(
            query=B_norm,
            key=A_norm,
            value=A_norm
        )
        B_prime = B_prime + B_self

        A_gate_logits = self.gate_mlp_A(A_prime).squeeze(-1)
        A_gate = F.softmax(A_gate_logits / self.gate_temperature, dim=-1)
        B_gate_logits = self.gate_mlp_B(B_prime).squeeze(-1)
        B_gate = F.softmax(B_gate_logits / self.gate_temperature, dim=-1)

        A_weighted = A_prime * A_gate.unsqueeze(-1)
        A_fused = A_weighted.sum(dim=1)
        B_weighted = B_prime * B_gate.unsqueeze(-1)
        B_fused = B_weighted.sum(dim=1)

        fused_feature = torch.cat([A_fused, B_fused], dim=-1)

        if self.training:
            fused_feature = self.dropout(fused_feature)

        logits = self.final_classifier(fused_feature).squeeze(-1)
        return logits
    
    def save_fusion_weights(self, save_path):
        """Save the full fusion model state."""
        torch.save(self.state_dict(), save_path)
        print(f"Saved fusion weights to: {save_path}")
    
    def load_fusion_weights(self, load_path):
        """Load the full fusion model state."""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")
        
        state_dict = torch.load(load_path, map_location=self.device)
        self.load_state_dict(state_dict, strict=False)
        
        print(f"Loaded fusion weights from: {load_path}")
