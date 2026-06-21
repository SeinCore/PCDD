import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
from transformers import AutoProcessor
from peft import LoraConfig, TaskType, get_peft_model

class ReconstructionModel(nn.Module):
    """Reconstruction model for modality-specific discrepancy encoding."""
    def __init__(
        self,
        model_path: str,
        lora_layers: int = 4,
        forward_layers: int = 12,
        hidden_size: int = None,
        num_heads: int = 8,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "cuda",
        attn_implementation: str = "flash_attention_2",
        use_train: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ):
        """Initialize the reconstruction model."""
        super().__init__()
        
        self.lora_layers = lora_layers
        self.forward_layers = forward_layers
        self.use_train = use_train
        
        self.base_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        
        self.base_model.disable_talker()
        
        if forward_layers > 0:
            total_layers = len(self.base_model.thinker.model.layers)
            if forward_layers < total_layers:
                self.base_model.thinker.model.layers = nn.ModuleList(
                    list(self.base_model.thinker.model.layers[:forward_layers])
                )
                print(
                    f"Trimmed thinker layers {forward_layers}-{total_layers - 1} "
                    f"({total_layers - forward_layers} removed)"
                )
                print(f"Remaining layers: {len(self.base_model.thinker.model.layers)} / {total_layers}")
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        hidden_size = self.base_model.config.hidden_size
        
        self.hidden_size = hidden_size
        
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        self.cross_attention_layer = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=num_heads, 
            dropout=0.0,
            batch_first=True
        )
        
        self.cross_attention_layer = self.cross_attention_layer.to(device_map).to(torch_dtype)
        
        attention_pool_query_tensor = torch.randn(1, 2, hidden_size, dtype=torch_dtype, device=device_map) * 0.02
        self.attention_pool_query = nn.Parameter(attention_pool_query_tensor)
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True
        )
        self.attention_pool = self.attention_pool.to(device_map).to(torch_dtype)
        
        self._setup_training_config()
        self._validate_mask_tokens()
    
    def _validate_mask_tokens(self):
        """Validate mask-token initialization."""
        try:
            thinker = self.base_model.thinker
            for name, param in [
                ("mask_text_token", thinker.mask_text_token),
                ("mask_audio_token", thinker.mask_audio_token),
                ("mask_video_token", thinker.mask_video_token),
            ]:
                if param.is_meta:
                    continue
                if torch.isinf(param).any() or torch.isnan(param).any():
                    raise ValueError(f"{name} contains inf or nan")
                print(f"Validated {name}: shape={param.shape}, device={param.device}, dtype={param.dtype}")
        except Exception as e:
            print(f"Mask token validation warning: {e}")
    
    def _setup_training_config(self):
        """Configure trainable parameters for reconstruction training."""
        if self.use_train:
            thinker_model = self.base_model.thinker.model
            
            target_modules = []
            for i in range(self.lora_layers):
                target_modules.extend([
                    f"layers.{i}.self_attn.q_proj",
                    f"layers.{i}.self_attn.k_proj",
                    f"layers.{i}.self_attn.v_proj",
                    f"layers.{i}.self_attn.o_proj",
                    f"layers.{i}.mlp.gate_proj",
                    f"layers.{i}.mlp.up_proj",
                    f"layers.{i}.mlp.down_proj",
                ])
            
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=target_modules,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                inference_mode=False,
            )
            
            self.base_model.thinker.model = get_peft_model(thinker_model, lora_config)
            
            for name, param in self.base_model.named_parameters():
                if 'lora' in name.lower():
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            for param in self.cross_attention_layer.parameters():
                param.requires_grad = True
            self.attention_pool_query.requires_grad = True
            for param in self.attention_pool.parameters():
                param.requires_grad = True
        else:
            for param in self.base_model.parameters():
                param.requires_grad = False
            for param in self.cross_attention_layer.parameters():
                param.requires_grad = False
            self.attention_pool_query.requires_grad = False
            for param in self.attention_pool.parameters():
                param.requires_grad = False
        if self.use_train:
            self._log_trainable_config()
    
    def _log_trainable_config(self):
        """Log the trainable parameter layout."""
        try:
            print("\n" + "="*60)
            print("Trainable parameter configuration")
            print("="*60)
            
            lora_param_names = [n for n, p in self.base_model.named_parameters() if p.requires_grad and 'lora' in n.lower()]
            lora_layer_ids = set()
            for name in lora_param_names:
                parts = name.split('.')
                if 'layers' in parts:
                    idx = parts.index('layers')
                    if idx + 1 < len(parts):
                        layer_str = parts[idx + 1]
                        if layer_str.isdigit():
                            lora_layer_ids.add(int(layer_str))
            lora_layer_ids = sorted(list(lora_layer_ids))
            
            lora_trainable_count = sum(p.numel() for n, p in self.base_model.named_parameters() if p.requires_grad and 'lora' in n.lower())
            ca_trainable_count = sum(p.numel() for _, p in self.cross_attention_layer.named_parameters() if p.requires_grad)
            attention_pool_query_count = self.attention_pool_query.numel() if self.attention_pool_query.requires_grad else 0
            attention_pool_count = sum(p.numel() for _, p in self.attention_pool.named_parameters() if p.requires_grad)
            total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            print(f"\n1. LoRA layers:")
            print(f"   - Layer ids: {lora_layer_ids if lora_layer_ids else list(range(self.lora_layers))}")
            print("   - Target modules: ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']")
            print(f"   - Parameter count: {lora_trainable_count:,}")
            
            print("\n2. Cross-attention layer:")
            print(f"   - Parameter count: {ca_trainable_count:,}")
            
            print("\n3. Attention pooling:")
            print(f"   - attention_pool_query: {attention_pool_query_count:,}")
            print(f"   - attention_pool: {attention_pool_count:,}")
            
            print(f"\nTotal trainable parameters: {total_trainable:,}")
            print("="*60 + "\n")
        except Exception as e:
            print(f"Trainable-parameter logging warning: {e}")
        
    
    
    def extract_feature(
        self,
        use_audio_in_video: bool = False,
        mask_text: bool = False,
        mask_text_ratio: float = 0.5,
        mask_audio: bool = False,
        mask_audio_ratio: float = 0.5,
        mask_video: bool = True,
        mask_video_ratio: float = 0.3,
        random_mask: bool = False,
        **kwargs
    ):
        """Extract discrepancy features for one masked modality."""
        forward_kwargs = {
            'use_audio_in_video': use_audio_in_video,
            'layers': self.forward_layers,
            'mask_text': mask_text,
            'mask_text_ratio': mask_text_ratio,
            'mask_audio': mask_audio,
            'mask_audio_ratio': mask_audio_ratio,
            'mask_video': mask_video,
            'mask_video_ratio': mask_video_ratio,
            'random_mask': random_mask,
            'simple_classify': False,
        }
        forward_kwargs.update(kwargs)

        result = self.base_model.thinker(**forward_kwargs)

        if isinstance(result, (tuple, list)):
            if len(result) == 4:
                hidden_states_unmasked, hidden_states_masked, masked_token_positions, _ = result
            else:
                hidden_states_unmasked, hidden_states_masked, masked_token_positions = result
        else:
            hidden_states_unmasked, hidden_states_masked, masked_token_positions = result

        batch_size = hidden_states_unmasked.shape[0]
        pooled_features = []
        cosine_losses = []

        for batch_idx in range(batch_size):
            batch_masked_positions = masked_token_positions[batch_idx]
            masked_indices = torch.where(batch_masked_positions)[0]
            unmasked_indices = torch.where(~batch_masked_positions)[0]

            query_tokens = hidden_states_masked[batch_idx, masked_indices]
            key_tokens = hidden_states_unmasked[batch_idx, unmasked_indices]
            value_tokens = hidden_states_unmasked[batch_idx, unmasked_indices]

            query_tokens = F.normalize(query_tokens.unsqueeze(0), p=2, dim=-1)
            key_tokens = F.normalize(key_tokens.unsqueeze(0), p=2, dim=-1)
            value_tokens = F.normalize(value_tokens.unsqueeze(0), p=2, dim=-1)

            processed_masked_tokens, _ = self.cross_attention_layer(
                query=query_tokens,
                key=key_tokens,
                value=value_tokens
            )
            masked_target_tokens = F.normalize(processed_masked_tokens.squeeze(0), p=2, dim=-1)
            unmasked_target_tokens = F.normalize(hidden_states_unmasked[batch_idx, masked_indices], p=2, dim=-1)

            cosine_sim = (masked_target_tokens * unmasked_target_tokens).sum(dim=-1)
            cosine_loss = 1 - cosine_sim
            cosine_losses.append(cosine_loss)

            diff_features = masked_target_tokens - unmasked_target_tokens
            diff_features = F.normalize(diff_features, p=2, dim=-1, eps=1e-8)

            query = self.attention_pool_query.to(diff_features.dtype)
            diff_features_expanded = diff_features.unsqueeze(0)
            pooled_output, _ = self.attention_pool(
                query=query,
                key=diff_features_expanded,
                value=diff_features_expanded
            )
            modal_diff_feature = pooled_output.squeeze(0).flatten()
            pooled_features.append(modal_diff_feature)
        
        pooled_batch = torch.stack(pooled_features, dim=0)
        cosine_loss_value = torch.stack(cosine_losses).mean()
        return pooled_batch, cosine_loss_value

    def forward(self, *args, **kwargs):
        return self.extract_feature(*args, **kwargs)
