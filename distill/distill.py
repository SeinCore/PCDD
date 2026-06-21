import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import json
import re
import random
from datetime import datetime
import builtins
from typing import List, Optional
from transformers import AutoModel, AutoTokenizer, AutoConfig
from tqdm import tqdm


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
import numpy as np
from scipy.stats import pearsonr, spearmanr
from peft import LoraConfig, TaskType, get_peft_model

class DatasetConfig:
    def __init__(self, dataset: str = "fakesv"):
        if dataset not in ["fakesv", "fakett"]:
            raise ValueError(f"Unsupported dataset: {dataset}. Use 'fakesv' or 'fakett'.")
        
        self.dataset = dataset
        self.is_chinese = (dataset == "fakesv")
        
        self.base_path = ""  # overridden by explicit paths from config.py
        self.train_title_transcript_path = f"{self.base_path}/train_title&transcript.json"
        self.train_analysis_path = f"{self.base_path}/train_analysis.json"
        self.val_title_transcript_path = f"{self.base_path}/val_title&transcript.json"
        self.val_analysis_path = f"{self.base_path}/val_analysis.json"
        self.label_names = ['CommonsenseViolation', 'LogicalFallacy', 'EmotionalManipulation']
        self.num_labels = 3
        self.field_relevant_sentences = 'EvidenceSalience'
        self.field_reasoning = 'ReasoningRationale'
        self.field_category_scores = 'StrategyProbabilities'
        self.field_sentence_content = 'sentence'
        self.field_confidence = 'score'
        
        if self.is_chinese:
            self.sentence_delimiters = r'[。！？\n]+'
            self.sentence_delimiters_strip = r'^[。！？\n]+'
            self.sentence_delimiters_optional = ['。', '！', '？']
        else:
            self.sentence_delimiters = r'[.!?\n]+'
            self.sentence_delimiters_strip = r'^[.!?\n]+'
            self.sentence_delimiters_optional = ['.', '!', '?']
    
    def __repr__(self):
        return f"DatasetConfig(dataset='{self.dataset}', num_labels={self.num_labels}, is_chinese={self.is_chinese})"

def split_sentences_in_interval(text, start_char, end_char, dataset_config: DatasetConfig):
    interval_text = text[start_char:end_char]
    cleaned_text = re.sub(dataset_config.sentence_delimiters_strip, '', interval_text)
    if not cleaned_text.strip():
        return []
    prefix_len = len(interval_text) - len(cleaned_text)
    actual_start = start_char + prefix_len
    sentences = re.split(dataset_config.sentence_delimiters, cleaned_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    result = []
    current_pos = actual_start
    for sentence in sentences:
        if len(sentence) >= 3:
            sent_start = text.find(sentence, current_pos, end_char)
            if sent_start != -1:
                sent_end = sent_start + len(sentence)
                result.append((sentence, sent_start, sent_end))
                current_pos = sent_end
    return result

def find_sentence_in_text(sentence_content, full_text, dataset_config: DatasetConfig):
    escaped = re.escape(sentence_content)
    for delim in dataset_config.sentence_delimiters_optional:
        escaped_delim = re.escape(delim)
        escaped = escaped.replace(escaped_delim, f'[{escaped_delim}]?')
    
    match = re.search(escaped, full_text)
    if match:
        return match.start(), match.end()
    
    if dataset_config.is_chinese:
        strip_chars = '。！？\n'
    else:
        strip_chars = '.!?\n'
    
    pattern = re.escape(sentence_content.rstrip(strip_chars))
    match = re.search(pattern, full_text)
    if match:
        return match.start(), match.end()
    return None, None

def compute_relation_metrics(t_flat_np, s_flat_np):
    if t_flat_np is None or s_flat_np is None:
        return None, None, None
    if len(t_flat_np) == 0 or len(s_flat_np) == 0:
        return None, None, None
    mean_abs_gap = float(np.mean(np.abs(t_flat_np - s_flat_np)))
    pear, _ = pearsonr(t_flat_np, s_flat_np)
    spear, _ = spearmanr(t_flat_np, s_flat_np)
    return mean_abs_gap, float(pear), float(spear)

def softrank(scores, temperature=0.1):
    scores = scores.to(torch.float32)
    s_i = scores.unsqueeze(0)
    s_j = scores.unsqueeze(1)
    probs = torch.sigmoid(torch.clamp((s_j - s_i) / temperature, -20, 20))
    mask = torch.eye(probs.shape[0], dtype=torch.bool, device=probs.device)
    probs = probs.masked_fill(mask, 0.0)
    soft_ranks = probs.sum(dim=0) + 1.0
    return soft_ranks

def weighted_spearman(ranks1, ranks2, alpha=0.2):
    ranks1 = ranks1.to(torch.float32)
    ranks2 = ranks2.to(torch.float32)
    n = ranks1.shape[0]
    if n <= 1:
        return torch.tensor(1.0, device=ranks1.device), torch.tensor(1.0, device=ranks1.device)
    rank_diff = ranks1 - ranks2
    correlation = 1.0 - 6.0 * (rank_diff ** 2).sum() / (n * (n ** 2 - 1))
    _, sorted_indices = torch.sort(ranks1)
    rank_weights = torch.exp(-torch.arange(n, dtype=torch.float32, device=ranks1.device) * alpha)
    weights = torch.zeros(n, dtype=torch.float32, device=ranks1.device)
    weights[sorted_indices] = rank_weights
    weights = weights / weights.sum() * n
    weighted_diff_sq = weights * (rank_diff ** 2)
    weighted_correlation = 1.0 - 6.0 * weighted_diff_sq.sum() / (n * (n ** 2 - 1))
    return weighted_correlation, correlation

def get_last_token_embedding(last_hidden_state, attention_mask):
    batch_size = last_hidden_state.size(0)
    batch_indices = torch.arange(batch_size, device=last_hidden_state.device)
    seq_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_state[batch_indices, seq_lengths]

def _records(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        records = []
        for key, value in data.items():
            if isinstance(value, dict) and "id" not in value:
                value = {"id": key, **value}
            records.append(value)
        return records
    return []

def _response_payload(item):
    if not isinstance(item, dict):
        return None
    response = item.get("response")
    if isinstance(response, dict):
        return response
    if isinstance(response, str):
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None

def _response_map(analysis_data):
    result = {}
    for item in _records(analysis_data):
        if not isinstance(item, dict) or not item.get("id"):
            continue
        payload = _response_payload(item)
        if payload is not None:
            result[item["id"]] = payload
    return result

def _normalize_state_dict(state):
    return state

class Student(nn.Module):
    def __init__(
        self, 
        model_path: Optional[str] = None,
        num_labels: Optional[int] = None,
        dropout=0.1,
        device=None,
        title_transcript_path: Optional[str] = None,
        analysis_path: Optional[str] = None,
        val_title_transcript_path: Optional[str] = None,
        val_analysis_path: Optional[str] = None,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[List[str]] = None,
        dataset: str = "fakesv",
    ):
        super(Student, self).__init__()
        
        self.dataset_config = DatasetConfig(dataset=dataset)
        
        if title_transcript_path is None:
            title_transcript_path = self.dataset_config.train_title_transcript_path
        if analysis_path is None:
            analysis_path = self.dataset_config.train_analysis_path
        if val_title_transcript_path is None:
            val_title_transcript_path = self.dataset_config.val_title_transcript_path
        if val_analysis_path is None:
            val_analysis_path = self.dataset_config.val_analysis_path
        
        if num_labels is None:
            num_labels = self.dataset_config.num_labels
        
        self.num_labels = num_labels
        
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        load_path = None
        if isinstance(model_path, str) and model_path.endswith('.pth'):
            load_path = model_path
            model_path = None

        if not model_path:
            raise ValueError(
                "model_path is required. When loading from a checkpoint (.pth), "
                "also pass the base model path via the distill_model_path config."
            )

        self.use_lora = use_lora
        self.qwen_config = AutoConfig.from_pretrained(model_path)
        base_qwen = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.pad_token_id = 0
        if self.use_lora:
            if lora_target_modules is None:
                lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            else:
                lora_target_modules = list(lora_target_modules)
            self.lora_target_modules = lora_target_modules
            self.lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.qwen = get_peft_model(base_qwen, self.lora_config)
        else:
            self.lora_target_modules = None
            self.lora_config = None
            self.qwen = base_qwen
        
        self.teacher_qwen = AutoModel.from_pretrained(model_path)
        
        self.dropout = nn.Dropout(dropout)
        self.stage3_top_k = 3
        self.stage3_score_head = nn.Linear(self.qwen_config.hidden_size * (1 + self.stage3_top_k), self.num_labels)
        
        hidden_dim = self.qwen_config.hidden_size
        self.sentence_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        self.to(self.device)
        self.qwen = self.qwen.to(dtype=torch.bfloat16)
        self.teacher_qwen = self.teacher_qwen.to(dtype=torch.bfloat16)
        self.sentence_scorer = self.sentence_scorer.to(dtype=torch.bfloat16)
        self.stage3_score_head = self.stage3_score_head.to(dtype=torch.bfloat16)
        
        if load_path is not None and os.path.exists(load_path):
            state = torch.load(load_path, map_location=self.device)
            self.load_state_dict(_normalize_state_dict(state), strict=False)
            print(f"Loaded weights from {load_path} (strict=False)")

        self.train_title_transcript_path = title_transcript_path
        self.train_analysis_path = analysis_path
        self.val_title_transcript_path = val_title_transcript_path
        self.val_analysis_path = val_analysis_path

    def _snapshot_state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
    
    def _load_stage3_data(self, title_transcript_path, analysis_path):
        with open(title_transcript_path, 'r', encoding='utf-8') as f:
            title_transcript = json.load(f)
        with open(analysis_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        title_transcript = _records(title_transcript)
        response_dict = _response_map(analysis_data)
        label_names = self.dataset_config.label_names
        
        data_list = []
        for item in title_transcript:
            if not isinstance(item, dict):
                continue
            video_id = item.get('id')
            if video_id not in response_dict:
                continue
            if self.dataset_config.is_chinese:
                text = f"*Title*: {item.get('title')}. *Transcript*: {item.get('transcript', '')}"
            else:
                text = f"*Title*: {item.get('title')}. *Transcript*: {item.get('transcript', '')}"
            scores = response_dict[video_id].get(self.dataset_config.field_category_scores)
            if not isinstance(scores, dict):
                continue
            labels = [scores.get(name) for name in label_names]
            if any(label is None for label in labels):
                continue
            data_list.append({'id': video_id, 'text': text, 'labels': labels})
        
        print(f"Loaded {len(data_list)} samples for stage 3 distillation (dataset={self.dataset_config.dataset})")
        return data_list
    
    def _load_stage2_data(self, title_transcript_path, analysis_path):
        with open(title_transcript_path, 'r', encoding='utf-8') as f:
            title_transcript = json.load(f)
        with open(analysis_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        title_transcript = _records(title_transcript)
        response_dict = _response_map(analysis_data)
        
        data_list = []
        for item in title_transcript:
            if not isinstance(item, dict):
                continue
            video_id = item.get('id')
            if video_id not in response_dict:
                continue
            if self.dataset_config.is_chinese:
                student_text = f"*Title*: {item.get('title')}. *Transcript*: {item.get('transcript', '')}"
            else:
                student_text = f"*Title*: {item.get('title')}. *Transcript*: {item.get('transcript', '')}"
            teacher_text = response_dict[video_id].get(self.dataset_config.field_reasoning)
            if not teacher_text:
                continue
            data_list.append({'id': video_id, 'student_text': student_text, 'teacher_text': teacher_text})
        
        return data_list
    
    def _load_stage1_data(self, title_transcript_path, analysis_path, strict_teacher_only: bool = False):
        with open(title_transcript_path, 'r', encoding='utf-8') as f:
            title_transcript = json.load(f)
        with open(analysis_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        title_transcript = _records(title_transcript)
        response_dict = _response_map(analysis_data)
        
        data_list = []
        for item in title_transcript:
            if not isinstance(item, dict):
                continue
            video_id = item.get('id')
            resp_item = response_dict.get(video_id)
            if resp_item is None:
                continue
            if self.dataset_config.is_chinese:
                full_text = f"*Title*: {item.get('title')}. *Transcript*: {item.get('transcript', '')}"
            else:
                full_text = f"*Title*: {item.get('title')}. *Transcript*: {item.get('transcript', '')}"
            teacher_sentences = resp_item.get(self.dataset_config.field_relevant_sentences)
            if not isinstance(teacher_sentences, list):
                continue
            teacher_sentence_info = {}
            matched_ranges = []
            
            for ts in teacher_sentences:
                sent_content = ts.get(self.dataset_config.field_sentence_content)
                conf = ts.get(self.dataset_config.field_confidence)
                if sent_content is None or conf is None:
                    continue
                start_char, end_char = find_sentence_in_text(sent_content, full_text, self.dataset_config)
                if start_char is not None and end_char is not None:
                    teacher_sentence_info[sent_content] = {
                        'confidence': conf,
                        'char_start': start_char,
                        'char_end': end_char
                    }
                    matched_ranges.append((start_char, end_char))
            matched_ranges.sort(key=lambda x: x[0])
            if not strict_teacher_only:
                unmatched_ranges = []
                prev_end = 0
                for start, end in matched_ranges:
                    if start > prev_end:
                        unmatched_ranges.append((prev_end, start))
                    prev_end = end
                if prev_end < len(full_text):
                    unmatched_ranges.append((prev_end, len(full_text)))
                for start_char, end_char in unmatched_ranges:
                    interval_length = end_char - start_char
                    if interval_length >= 3:
                        sentence_list = split_sentences_in_interval(full_text, start_char, end_char, self.dataset_config)
                        for sentence, sent_start, sent_end in sentence_list:
                            teacher_sentence_info[sentence] = {
                                'confidence': random.uniform(0.0, 0.4),
                                'char_start': sent_start,
                                'char_end': sent_end
                            }
            encoded = self.tokenizer(
                full_text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                return_offsets_mapping=True
            )
            offset_mapping = encoded['offset_mapping'][0]
            input_ids = encoded['input_ids'][0].tolist()
            sentence_token_indices = []
            valid_sentence_info = []
            
            for sent_content, info in teacher_sentence_info.items():
                char_start = info['char_start']
                char_end = info['char_end']
                conf = info['confidence']
                offsets_cpu = offset_mapping
                valid_mask = offsets_cpu[:, 0] != offsets_cpu[:, 1]
                valid_indices = torch.where(valid_mask)[0]
                
                if len(valid_indices) == 0:
                    continue
                    
                valid_offsets = offsets_cpu[valid_mask]
                start_mask = (valid_offsets[:, 0] <= char_start) & (char_start < valid_offsets[:, 1])
                end_mask = (valid_offsets[:, 0] < char_end) & (char_end <= valid_offsets[:, 1])
                
                if start_mask.any() and end_mask.any():
                    start_match = torch.where(start_mask)[0]
                    end_match = torch.where(end_mask)[0]
                    if len(start_match) > 0 and len(end_match) > 0:
                        token_start_idx = valid_indices[start_match[0]].item()
                        token_end_idx = valid_indices[end_match[-1]].item() + 1
                        
                        if token_start_idx < token_end_idx:
                            sentence_token_indices.append((token_start_idx, token_end_idx))
                            valid_sentence_info.append(conf)
            
            if len(sentence_token_indices) == 0:
                continue
            if strict_teacher_only and len(sentence_token_indices) < 2:
                continue
                
            data_list.append({
                'id': video_id,
                'full_text': full_text,
                'input_ids': input_ids,
                'sentence_token_indices': sentence_token_indices,
                'teacher_confidences': valid_sentence_info
            })
        
        print(f"Loaded {len(data_list)} samples for stage 1 distillation (pre-tokenized, dataset={self.dataset_config.dataset})")
        return data_list
    
    def _stage1_compute_batch(self, batch_data, temperature, alpha):
        batch_input_ids = []
        batch_attention_masks = []
        batch_sample_info = []
        
        for sample in batch_data:
            input_ids_list = sample['input_ids']
            sentence_token_indices = sample['sentence_token_indices']
            teacher_confidences_list = sample['teacher_confidences']
            input_ids = torch.tensor(input_ids_list, dtype=torch.long, pin_memory=True)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_mask)
            batch_sample_info.append({
                'sentence_indices': sentence_token_indices,
                'confidences': teacher_confidences_list
            })
        batch_input_ids_padded = pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch_attention_masks_padded = pad_sequence(batch_attention_masks, batch_first=True, padding_value=0)
        batch_input_ids_padded = batch_input_ids_padded.to(self.device, non_blocking=True)
        batch_attention_masks_padded = batch_attention_masks_padded.to(self.device, non_blocking=True)
        
        outputs = self.qwen(batch_input_ids_padded, attention_mask=batch_attention_masks_padded)
        batch_token_embeddings = outputs.last_hidden_state
        batch_losses = []
        batch_correlations = []
        batch_normal_correlations = []
        
        for b_idx, sample_info in enumerate(batch_sample_info):
            token_embeddings = batch_token_embeddings[b_idx]
            attention_mask = batch_attention_masks_padded[b_idx]
            valid_seq_len = attention_mask.sum().item()
            
            sentence_token_indices = sample_info['sentence_indices']
            teacher_confidences_list = sample_info['confidences']
            valid_indices = []
            valid_confidences = []
            for (token_start, token_end), conf in zip(sentence_token_indices, teacher_confidences_list):
                token_end = min(token_end, valid_seq_len)
                if token_start < token_end:
                    valid_indices.append((token_start, token_end))
                    valid_confidences.append(conf)
            if len(valid_indices) == 0:
                continue
            num_sents = len(valid_indices)
            hidden_dim = token_embeddings.size(-1)
            sentence_embeddings = torch.zeros(num_sents, hidden_dim, device=self.device, dtype=token_embeddings.dtype)
            
            for sent_idx, (token_start, token_end) in enumerate(valid_indices):
                sentence_tokens = token_embeddings[token_start:token_end]
                sentence_mask = attention_mask[token_start:token_end].unsqueeze(-1)
                masked_embeddings = sentence_tokens * sentence_mask
                sentence_emb = masked_embeddings.sum(dim=0) / sentence_mask.sum().clamp(min=1.0)
                sentence_embeddings[sent_idx] = sentence_emb
            teacher_confidences = torch.tensor(valid_confidences, dtype=torch.float32, pin_memory=True).to(self.device, non_blocking=True)
            student_scores = self.sentence_scorer(sentence_embeddings).squeeze(-1)
            teacher_ranks = softrank(teacher_confidences, temperature=temperature)
            student_ranks = softrank(student_scores, temperature=temperature)
            correlation, normal_correlation = weighted_spearman(teacher_ranks, student_ranks, alpha=alpha)
            sample_loss = 1.0 - correlation
            
            batch_losses.append(sample_loss)
            batch_correlations.append(correlation.item())
            batch_normal_correlations.append(normal_correlation.item())
        
        return batch_losses, batch_correlations, batch_normal_correlations

    def _stage2_compute_batch(self, student_texts, teacher_texts, max_length, mse_loss, temperature=1.0):
        student_encoded = self.tokenizer(
            student_texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        teacher_encoded = self.tokenizer(
            teacher_texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        student_input_ids = student_encoded['input_ids'].to(self.device, non_blocking=True)
        student_attention_mask = student_encoded['attention_mask'].to(self.device, non_blocking=True)
        teacher_input_ids = teacher_encoded['input_ids'].to(self.device, non_blocking=True)
        teacher_attention_mask = teacher_encoded['attention_mask'].to(self.device, non_blocking=True)

        student_outputs = self.qwen(student_input_ids, attention_mask=student_attention_mask)
        student_cls = get_last_token_embedding(student_outputs.last_hidden_state, student_attention_mask)
        with torch.no_grad():
            teacher_outputs = self.teacher_qwen(teacher_input_ids, attention_mask=teacher_attention_mask)
            teacher_cls = get_last_token_embedding(teacher_outputs.last_hidden_state, teacher_attention_mask)

        student_unit = torch.nn.functional.normalize(student_cls, p=2, dim=1, eps=1e-12)
        teacher_unit = torch.nn.functional.normalize(teacher_cls, p=2, dim=1, eps=1e-12)
        teacher_similarity = torch.mm(teacher_unit, teacher_unit.t())/temperature
        student_similarity = torch.mm(student_unit, student_unit.t())/temperature
        bsz = student_similarity.size(0)
        offdiag_mask = ~torch.eye(bsz, dtype=torch.bool, device=student_similarity.device)
        batch_loss = mse_loss(student_similarity[offdiag_mask], teacher_similarity[offdiag_mask])
        return batch_loss, (
            teacher_similarity[offdiag_mask].detach().cpu().float().numpy(),
            student_similarity[offdiag_mask].detach().cpu().float().numpy()
        )

    def _stage1_validate(self, val_data, batch_size, temperature, alpha):
        self.eval()
        with torch.no_grad():
            val_losses = []
            val_valid_samples = 0
            val_correlation_sum = 0.0
            val_normal_correlation_sum = 0.0
            for j in range(0, len(val_data), batch_size):
                batch_val = val_data[j:j + batch_size]
                batch_losses, batch_correlations, batch_normal_correlations = self._stage1_compute_batch(
                    batch_val, temperature, alpha
                )
                if len(batch_losses) > 0:
                    val_losses.append(torch.stack(batch_losses).mean().item())
                    val_correlation_sum += sum(batch_correlations)
                    val_normal_correlation_sum += sum(batch_normal_correlations)
                    val_valid_samples += len(batch_losses)
            val_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else float('inf')
            val_correlation = (val_correlation_sum / val_valid_samples) if val_valid_samples > 0 else 0.0
            val_normal_correlation = (val_normal_correlation_sum / val_valid_samples) if val_valid_samples > 0 else 0.0
        return val_loss, val_correlation, val_normal_correlation

    def _stage2_validate(self, val_data, batch_size, max_length, mse_loss, temperature=1.0):
        self.eval()
        with torch.no_grad():
            val_losses = []
            val_teacher_offdiag_list = []
            val_student_offdiag_list = []
            for i in range(0, len(val_data), batch_size):
                batch_val = val_data[i:i + batch_size]
                if len(batch_val) == 0:
                    continue
                if len(batch_val) == 1:
                    continue
                student_texts = [item['student_text'] for item in batch_val]
                teacher_texts = [item['teacher_text'] for item in batch_val]
                vloss, (t_off_np, s_off_np) = self._stage2_compute_batch(student_texts, teacher_texts, max_length, mse_loss, temperature)
                val_losses.append(vloss.item())
                val_teacher_offdiag_list.append(t_off_np)
                val_student_offdiag_list.append(s_off_np)
            val_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else float('inf')
            t_all_v = np.concatenate(val_teacher_offdiag_list)
            s_all_v = np.concatenate(val_student_offdiag_list)
            gap, pear, spear = compute_relation_metrics(t_all_v, s_all_v)
        return val_loss, gap, pear, spear

    def _stage3_validate(self, val_data, batch_size, max_length, bce_loss, temperature=1.0):
        self.eval()
        with torch.no_grad():
            val_losses = []
            val_correct = 0.0
            val_total = 0
            for i in range(0, len(val_data), batch_size):
                batch_val = val_data[i:i + batch_size]
                if len(batch_val) == 0:
                    continue
                texts = [item['text'] for item in batch_val]
                labels = torch.tensor([item['labels'] for item in batch_val], dtype=torch.float32, pin_memory=True).to(self.device, non_blocking=True)
                encoded = self.tokenizer(texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt', return_offsets_mapping=True)
                input_ids = encoded['input_ids'].to(self.device, non_blocking=True)
                attention_mask = encoded['attention_mask'].to(self.device, non_blocking=True)
                offsets = encoded['offset_mapping']
                outputs = self.qwen(input_ids, attention_mask=attention_mask)
                last_hidden = outputs.last_hidden_state
                concat_features = self._stage3_build_concat_features_train(texts, last_hidden, attention_mask, offsets, self.stage3_top_k)
                logits = self.stage3_score_head(concat_features)
                teacher_logits = torch.log(labels.clamp(1e-6, 1 - 1e-6) / (1 - labels.clamp(1e-6, 1 - 1e-6)))
                teacher_probs = torch.sigmoid(teacher_logits / temperature)
                vloss = bce_loss(logits / temperature, teacher_probs) * (temperature ** 2)
                val_losses.append(vloss.item())
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct = (preds == (labels > 0.5).float()).float()
                val_correct += correct.sum().item()
                val_total += labels.numel()
            val_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else float('inf')
            val_accuracy = (val_correct / val_total) if val_total > 0 else 0.0
        return val_loss, val_accuracy

    def _stage3_build_concat_features_train(self, texts, last_hidden, attention_mask, offsets, k):
        last_token_emb = get_last_token_embedding(last_hidden, attention_mask)
        batch_concat_features = []
        for b_idx, text in enumerate(texts):
            sents = split_sentences_in_interval(text, 0, len(text), self.dataset_config)
            sent_emb_list = []
            if len(sents) > 0 and k > 0:
                off = offsets[b_idx]
                off_valid_mask = (off[:, 0] != off[:, 1])
                valid_indices = torch.where(off_valid_mask)[0]
                valid_off = off[off_valid_mask]
                sample_hidden = last_hidden[b_idx]
                sample_mask = attention_mask[b_idx]
                valid_len = int(sample_mask.sum().item())
                for _, c_start, c_end in sents:
                    start_mask = (valid_off[:, 0] <= c_start) & (c_start < valid_off[:, 1])
                    end_mask = (valid_off[:, 0] < c_end) & (c_end <= valid_off[:, 1])
                    if start_mask.any() and end_mask.any():
                        t_start = int(valid_indices[torch.where(start_mask)[0][0]].item())
                        t_end = int(valid_indices[torch.where(end_mask)[0][-1]].item()) + 1
                        t_end = min(t_end, valid_len)
                        if t_start < t_end:
                            tokens = sample_hidden[t_start:t_end]
                            mask = sample_mask[t_start:t_end].unsqueeze(-1)
                            emb = (tokens * mask).sum(dim=0) / mask.sum().clamp(min=1)
                            sent_emb_list.append(emb)
            if len(sent_emb_list) == 0 or k <= 0:
                topk_vec = torch.zeros((max(0, k), self.qwen_config.hidden_size), device=self.device, dtype=last_token_emb.dtype)
            else:
                sent_embs = torch.stack(sent_emb_list, dim=0)
                with torch.no_grad():
                    scores = self.sentence_scorer(sent_embs).squeeze(-1)
                    kk = min(k, scores.size(0))
                    topk_idx = torch.topk(scores, k=kk, dim=0).indices
                topk_embs = sent_embs[topk_idx]
                if kk < k:
                    pad = torch.zeros((k - kk, self.qwen_config.hidden_size), device=self.device, dtype=last_token_emb.dtype)
                    topk_vec = torch.cat([topk_embs, pad], dim=0)
                else:
                    topk_vec = topk_embs
            concat_feat = torch.cat([last_token_emb[b_idx], topk_vec.reshape(-1)], dim=-1) if k > 0 else last_token_emb[b_idx]
            batch_concat_features.append(concat_feat)
        return torch.stack(batch_concat_features, dim=0)

    def stage1_distill(self, num_epochs=20, batch_size=16, max_length=512, lr=5e-5, save_path="stage1_model.pth", temperature=0.1, alpha=0.25):
        swanlab.init(
            project=f"pcdd_distill_{self.dataset_config.dataset}",
            experiment_name="R3_phase1",
            config={"num_epochs": num_epochs, "batch_size": batch_size, "max_length": max_length, "learning_rate": lr, "save_path": save_path, "temperature": temperature}
        )
        self.train()
        optimizer = torch.optim.Adam([
            {'params': self.qwen.parameters()},
            {'params': self.sentence_scorer.parameters()}
        ], lr=lr)
        data = self._load_stage1_data(self.train_title_transcript_path, self.train_analysis_path, strict_teacher_only=False)
        val_data = self._load_stage1_data(self.val_title_transcript_path, self.val_analysis_path, strict_teacher_only=True)
        num_batches = (len(data) + batch_size - 1) // batch_size
        best_val_loss, val_no_improve = float('inf'), 0
        lr_decay_no_improve = 0
        best_state = None
        init_val_loss, init_val_correlation, init_val_normal_correlation = self._stage1_validate(val_data, batch_size, temperature, alpha)
        print(f"Initial validation loss: {init_val_loss:.6f}, weighted correlation: {init_val_correlation:.6f}, correlation: {init_val_normal_correlation:.6f}")
        best_val_loss = init_val_loss
        best_state = self._snapshot_state_dict()
        swanlab.log({"val_loss": init_val_loss, "val_correlation": init_val_correlation, "val_normal_correlation": init_val_normal_correlation})
        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0.0
            valid_samples = 0
            epoch_correlation = 0.0
            epoch_normal_correlation = 0.0
            for i in tqdm(range(0, len(data), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}", total=num_batches):
                optimizer.zero_grad()

                batch_data = data[i:i + batch_size]
                batch_losses, batch_correlations, batch_normal_correlations = self._stage1_compute_batch(
                    batch_data, temperature, alpha
                )
                
                if len(batch_losses) == 0:
                    continue
                batch_loss = torch.stack(batch_losses).mean()
                batch_loss.backward()
                optimizer.step()
                
                epoch_loss += batch_loss.item()
                step_correlation = sum(batch_correlations)
                step_normal_correlation = sum(batch_normal_correlations)
                epoch_correlation += step_correlation
                epoch_normal_correlation += step_normal_correlation
                valid_samples += len(batch_losses)
                swanlab.log({"loss": batch_loss.item(), "correlation": step_correlation/len(batch_losses), "normal_correlation": step_normal_correlation/len(batch_losses)})
            
            if valid_samples == 0:
                continue
            epoch_loss = epoch_loss / valid_samples
            epoch_correlation = epoch_correlation / valid_samples
            epoch_normal_correlation = epoch_normal_correlation / valid_samples
            val_loss, val_correlation, val_normal_correlation = self._stage1_validate(val_data, batch_size, temperature, alpha)
            if val_loss < best_val_loss:
                best_val_loss, val_no_improve = val_loss, 0
                lr_decay_no_improve = 0
                best_state = self._snapshot_state_dict()
                if save_path:
                    torch.save(best_state, save_path)
                status = "(new best validation loss, saved)"
            else:
                val_no_improve += 1
                lr_decay_no_improve += 1
                status = f"(no validation improvement for {val_no_improve} epoch(s))"

            print(f"Epoch {epoch+1}/{num_epochs} - train loss: {epoch_loss:.6f}, val loss: {val_loss:.6f} {status}")
            current_lr = optimizer.param_groups[0]['lr']
            swanlab.log({
                "train_loss": epoch_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
                "epoch_correlation": epoch_correlation,
                "epoch_normal_correlation": epoch_normal_correlation,
                "val_correlation": val_correlation,
                "val_normal_correlation": val_normal_correlation
            })
            if lr_decay_no_improve >= 2:
                for param_group in optimizer.param_groups:
                    old_lr, new_lr = param_group['lr'], param_group['lr'] * 0.5
                    param_group['lr'] = new_lr
                    print(f"Learning rate decay: {old_lr:.2e} -> {new_lr:.2e}")
                    swanlab.log({"learning_rate_decay": new_lr})
                lr_decay_no_improve = 0
            if val_no_improve >= 5:
                print(f"Early stopping at epoch {epoch+1} after 5 epochs without validation improvement")
                swanlab.log({"early_stopped": True, "final_epoch": epoch + 1})
                break
        if save_path and best_state is not None:
            torch.save(best_state, save_path)
        return best_val_loss
    
    def stage2_distill(self, num_epochs=20, batch_size=16, max_length=512, lr=5e-5, save_path="stage2_model.pth", load_path="stage1.pth", temperature=1.0):
        swanlab.init(
            project=f"pcdd_distill_{self.dataset_config.dataset}",
            experiment_name="R3_phase2",
            config={"num_epochs": num_epochs, "batch_size": batch_size, "max_length": max_length, "learning_rate": lr, "save_path": save_path, "temperature": temperature}
        )
        if load_path and os.path.exists(load_path):
            state = torch.load(load_path, map_location=self.device)
            self.load_state_dict(_normalize_state_dict(state), strict=False)
            print(f"Loaded previous-stage weights from: {load_path}")
            self.teacher_qwen.load_state_dict(_normalize_state_dict(state), strict=False)
        self.train()
        self.teacher_qwen.eval()
        for param in self.teacher_qwen.parameters():
            param.requires_grad = False
        
        optimizer = torch.optim.Adam([
            {'params': self.qwen.parameters()}
        ], lr=lr)
        mse_loss = nn.MSELoss()
        data = self._load_stage2_data(self.train_title_transcript_path, self.train_analysis_path)
        val_data = self._load_stage2_data(self.val_title_transcript_path, self.val_analysis_path)
        num_batches = (len(data) + batch_size - 1) // batch_size
        best_val_loss, val_no_improve = float('inf'), 0
        lr_decay_no_improve = 0
        random.seed(42)
        init_val_loss, init_val_gap, init_val_pearson, init_val_spearman = self._stage2_validate(val_data, batch_size, max_length, mse_loss, temperature)
        print(f"Initial validation loss: {init_val_loss:.6f}, cosine gap: {init_val_gap:.6f}, Pearson: {init_val_pearson:.6f}, Spearman: {init_val_spearman:.6f}")
        best_val_loss = init_val_loss
        best_state = self._snapshot_state_dict()

        swanlab.log({
            "val_loss": init_val_loss,
            "val_mean_abs_gap": init_val_gap,
            "val_pearson": float(init_val_pearson) if init_val_pearson is not None else None,
            "val_spearman": float(init_val_spearman) if init_val_spearman is not None else None,
            "epoch": 0
        })
        for epoch in range(num_epochs):
            self.train()
            random.shuffle(data)
            epoch_loss = 0.0
            epoch_teacher_offdiag_list = []
            epoch_student_offdiag_list = []
            actual_batches = 0
            for i in tqdm(range(0, len(data), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}", total=num_batches):
                batch_data = data[i:i + batch_size]
                if len(batch_data) == 1:
                    continue
                
                optimizer.zero_grad()
                
                student_texts = [item['student_text'] for item in batch_data]
                teacher_texts = [item['teacher_text'] for item in batch_data]
                batch_loss, (t_off_np, s_off_np) = self._stage2_compute_batch(student_texts, teacher_texts, max_length, mse_loss, temperature)
                epoch_teacher_offdiag_list.append(t_off_np)
                epoch_student_offdiag_list.append(s_off_np)
                batch_loss.backward()
                optimizer.step()
                
                epoch_loss += batch_loss.item()
                actual_batches += 1
                swanlab.log({
                    "step_loss": batch_loss.item(),
                    "teacher_similarity_mean": float(np.mean(t_off_np)),
                    "teacher_similarity_std": float(np.std(t_off_np)),
                    "student_similarity_mean": float(np.mean(s_off_np)),
                    "student_similarity_std": float(np.std(s_off_np))
                })
            epoch_loss = epoch_loss / actual_batches if actual_batches > 0 else float('inf')
            train_mean_abs_gap, train_pearson, train_spearman = None, None, None
            if len(epoch_teacher_offdiag_list) > 0:
                t_all = np.concatenate(epoch_teacher_offdiag_list)
                s_all = np.concatenate(epoch_student_offdiag_list)
                train_mean_abs_gap, train_pearson, train_spearman = compute_relation_metrics(t_all, s_all)
            val_loss, val_mean_abs_gap, val_pearson, val_spearman = self._stage2_validate(val_data, batch_size, max_length, mse_loss, temperature)
            if val_loss < best_val_loss:
                best_val_loss, val_no_improve = val_loss, 0
                lr_decay_no_improve = 0
                best_state = self._snapshot_state_dict()
                if save_path:
                    torch.save(best_state, save_path)
                status = "(new best validation loss, saved)"
            else:
                val_no_improve += 1
                lr_decay_no_improve += 1
                status = f"(no validation improvement for {val_no_improve} epoch(s))"

            print(f"Epoch {epoch+1}/{num_epochs} - train loss: {epoch_loss:.6f}, val loss: {val_loss:.6f} {status}")
            current_lr = optimizer.param_groups[0]['lr']
            log_dict = {"train_loss": epoch_loss, "val_loss": val_loss, "learning_rate": current_lr}
            if train_mean_abs_gap is not None:
                log_dict.update({
                    "train_mean_abs_gap": train_mean_abs_gap,
                    "train_pearson": float(train_pearson),
                    "train_spearman": float(train_spearman)
                })
            if val_mean_abs_gap is not None:
                log_dict.update({
                    "val_mean_abs_gap": val_mean_abs_gap,
                    "val_pearson": float(val_pearson),
                    "val_spearman": float(val_spearman)
                })
            swanlab.log(log_dict)
            if lr_decay_no_improve >= 2:
                for param_group in optimizer.param_groups:
                    old_lr, new_lr = param_group['lr'], param_group['lr'] * 0.5
                    param_group['lr'] = new_lr
                    print(f"Learning rate decay: {old_lr:.2e} -> {new_lr:.2e}")
                    swanlab.log({"learning_rate_decay": new_lr})
                lr_decay_no_improve = 0
            if val_no_improve >= 5:
                print(f"Early stopping at epoch {epoch+1} after 5 epochs without validation improvement")
                swanlab.log({"early_stopped": True, "final_epoch": epoch + 1})
                break
        if save_path and best_state is not None:
            torch.save(best_state, save_path)
        return best_val_loss

    def stage3_distill(self, num_epochs=20, batch_size=16, max_length=512, lr=5e-5, save_path="stage3_model.pth", load_path="stage2.pth", temperature=1.0):
        swanlab.init(
            project=f"pcdd_distill_{self.dataset_config.dataset}",
            experiment_name="R3_phase3",
            config={"num_epochs": num_epochs, "batch_size": batch_size, "max_length": max_length, "learning_rate": lr, "save_path": save_path, "temperature": temperature}
        )
        if load_path and os.path.exists(load_path):
            state = torch.load(load_path, map_location=self.device)
            self.load_state_dict(_normalize_state_dict(state), strict=True)
            print(f"Loaded previous-stage weights from: {load_path}")
        self.train()
        k = self.stage3_top_k
        self.sentence_scorer.eval()
        for p in self.sentence_scorer.parameters():
            p.requires_grad = False

        optimizer = torch.optim.Adam([
            {'params': self.qwen.parameters()},
            {'params': self.stage3_score_head.parameters()},
        ], lr=lr)
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        data = self._load_stage3_data(self.train_title_transcript_path, self.train_analysis_path)
        val_data = self._load_stage3_data(self.val_title_transcript_path, self.val_analysis_path)
        random.seed(42)
        random.shuffle(data)
        num_batches = (len(data) + batch_size - 1) // batch_size
        best_val_loss, val_no_improve = float('inf'), 0
        lr_decay_no_improve = 0
        init_val_loss, init_val_accuracy = self._stage3_validate(val_data, batch_size, max_length, bce_loss, temperature)
        swanlab.log({"val_loss": init_val_loss, "val_accuracy": init_val_accuracy})
        best_val_loss = init_val_loss
        best_state = self._snapshot_state_dict()
        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0.0
            total_correct = 0.0
            for i in tqdm(range(0, len(data), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}", total=num_batches):
                optimizer.zero_grad()
                batch_data = data[i:i + batch_size]
                texts = [item['text'] for item in batch_data]
                labels = torch.tensor([item['labels'] for item in batch_data], dtype=torch.float32, pin_memory=True).to(self.device, non_blocking=True)
                encoded = self.tokenizer(texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt', return_offsets_mapping=True)
                input_ids = encoded['input_ids'].to(self.device, non_blocking=True)
                attention_mask = encoded['attention_mask'].to(self.device, non_blocking=True)
                offsets = encoded['offset_mapping']
                outputs = self.qwen(input_ids, attention_mask=attention_mask)
                last_hidden = outputs.last_hidden_state
                concat_features = self._stage3_build_concat_features_train(texts, last_hidden, attention_mask, offsets, k)
                logits = self.stage3_score_head(concat_features)
                teacher_logits = torch.log(labels.clamp(1e-6, 1 - 1e-6) / (1 - labels.clamp(1e-6, 1 - 1e-6)))
                teacher_probs = torch.sigmoid(teacher_logits / temperature)
                step_loss = bce_loss(logits / temperature, teacher_probs) * (temperature ** 2)
                step_loss.backward()
                optimizer.step()
                
                epoch_loss += step_loss.item()
                with torch.no_grad():
                    predictions = (torch.sigmoid(logits) > 0.5).float()
                    labels_binary = (labels > 0.5).float()
                    correct = (predictions == labels_binary).float()
                    step_accuracy = correct.mean(dim=1).mean().item()
                    total_correct += correct.sum().item()
                    
                    swanlab.log({"step_accuracy": step_accuracy, "step_loss": step_loss.item()})
            epoch_loss = epoch_loss / num_batches
            epoch_accuracy = total_correct / (len(data) * self.num_labels)
            val_loss, val_accuracy = self._stage3_validate(val_data, batch_size, max_length, bce_loss, temperature)
            if val_loss < best_val_loss:
                best_val_loss, val_no_improve = val_loss, 0
                lr_decay_no_improve = 0
                best_state = self._snapshot_state_dict()
                if save_path:
                    torch.save(best_state, save_path)
                status = "(new best validation loss, saved)"
            else:
                val_no_improve += 1
                lr_decay_no_improve += 1
                status = f"(no validation improvement for {val_no_improve} epoch(s))"

            print(f"Epoch {epoch+1}/{num_epochs} - train loss: {epoch_loss:.4f}, train score accuracy: {epoch_accuracy:.4f}, val loss: {val_loss:.4f}, val score accuracy: {val_accuracy:.4f} {status}")
            current_lr = optimizer.param_groups[0]['lr']
            swanlab.log({"train_loss": epoch_loss, "train_accuracy": epoch_accuracy, "val_loss": val_loss, "val_accuracy": val_accuracy, "learning_rate": current_lr})
            if lr_decay_no_improve >= 2:
                for param_group in optimizer.param_groups:
                    old_lr, new_lr = param_group['lr'], param_group['lr'] * 0.5
                    param_group['lr'] = new_lr
                    print(f"Learning rate decay: {old_lr:.2e} -> {new_lr:.2e}")
                    swanlab.log({"learning_rate_decay": new_lr})
                lr_decay_no_improve = 0
            if val_no_improve >= 5:
                print(f"Early stopping at epoch {epoch+1} after 5 epochs without validation improvement")
                swanlab.log({"early_stopped": True, "final_epoch": epoch + 1})
                break
        if save_path and best_state is not None:
            torch.save(best_state, save_path)
        return best_val_loss

def full_training_pipeline(
    dataset: str = "fakesv",
    model_path: Optional[str] = None,
    train_title_transcript_path: Optional[str] = None,
    train_analysis_path: Optional[str] = None,
    val_title_transcript_path: Optional[str] = None,
    val_analysis_path: Optional[str] = None,
    save_dir: str = "checkpoint/distill",
    stage1_num_epochs: int = 30,
    stage1_batch_size: int = 16,
    stage1_max_length: int = 512,
    stage1_lr: float = 5e-5,
    stage1_temperature: float = 0.1,
    stage1_alpha: float = 0.25,
    stage2_num_epochs: int = 30,
    stage2_batch_size: int = 16,
    stage2_max_length: int = 512,
    stage2_lr: float = 5e-5,
    stage2_temperature: float = 2.0,
    stage3_num_epochs: int = 30,
    stage3_batch_size: int = 16,
    stage3_max_length: int = 512,
    stage3_lr: float = 5e-5,
    stage3_temperature: float = 2.0,
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    lora_target_modules: Optional[List[str]] = None,
):
    if not model_path:
        raise ValueError("model_path is required; set MODELS.distill in config.py")
    dataset_config = DatasetConfig(dataset=dataset)
    save_dir = os.path.join(save_dir, dataset)

    print("=" * 80)
    print(f"Starting distillation pipeline (dataset={dataset}, labels={dataset_config.num_labels})")
    print("=" * 80)
    if train_title_transcript_path is None:
        train_title_transcript_path = dataset_config.train_title_transcript_path
    if train_analysis_path is None:
        train_analysis_path = dataset_config.train_analysis_path
    if val_title_transcript_path is None:
        val_title_transcript_path = dataset_config.val_title_transcript_path
    if val_analysis_path is None:
        val_analysis_path = dataset_config.val_analysis_path
    os.makedirs(save_dir, exist_ok=True)
    print(f"[Config] Save directory: {save_dir}")
    print(f"[Config] Train data: {train_title_transcript_path}")
    print(f"[Config] Validation data: {val_title_transcript_path}")
    stage1_save_path = os.path.join(save_dir, "stage1_model.pth")
    stage2_save_path = os.path.join(save_dir, "stage2_model.pth")
    stage3_save_path = os.path.join(save_dir, "stage3_model.pth")
    student = Student(
        model_path=model_path,
        title_transcript_path=train_title_transcript_path,
        analysis_path=train_analysis_path,
        val_title_transcript_path=val_title_transcript_path,
        val_analysis_path=val_analysis_path,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        dataset=dataset
    )
    if os.path.exists(stage1_save_path):
        print(f"\n[Stage 1] Loading existing weights: {stage1_save_path}")
        state = torch.load(stage1_save_path, map_location=student.device)
        student.load_state_dict(_normalize_state_dict(state), strict=False)
    else:
        print("\n" + "=" * 80)
        print("Stage 1: sentence ranking distillation")
        print("=" * 80)
        best_val_loss = student.stage1_distill(
            num_epochs=stage1_num_epochs,
            batch_size=stage1_batch_size,
            max_length=stage1_max_length,
            lr=stage1_lr,
            save_path=stage1_save_path,
            temperature=stage1_temperature,
            alpha=stage1_alpha,
        )
        print(f"[Stage 1] Finished, best validation loss: {best_val_loss:.6f}")
        swanlab.finish()

    if os.path.exists(stage2_save_path):
        print(f"\n[Stage 2] Loading existing weights: {stage2_save_path}")
        state = torch.load(stage2_save_path, map_location=student.device)
        student.load_state_dict(_normalize_state_dict(state), strict=False)
    else:
        print("\n" + "=" * 80)
        print("Stage 2: sentence representation alignment distillation")
        print("=" * 80)
        best_val_loss = student.stage2_distill(
            num_epochs=stage2_num_epochs,
            batch_size=stage2_batch_size,
            max_length=stage2_max_length,
            lr=stage2_lr,
            save_path=stage2_save_path,
            load_path=stage1_save_path,
            temperature=stage2_temperature,
        )
        print(f"[Stage 2] Finished, best validation loss: {best_val_loss:.6f}")
        swanlab.finish()

    if os.path.exists(stage3_save_path):
        print(f"\n[Stage 3] Loading existing weights: {stage3_save_path}")
        state = torch.load(stage3_save_path, map_location=student.device)
        student.load_state_dict(_normalize_state_dict(state), strict=False)
    else:
        print("\n" + "=" * 80)
        print("Stage 3: label score distillation")
        print("=" * 80)
        best_val_loss = student.stage3_distill(
            num_epochs=stage3_num_epochs,
            batch_size=stage3_batch_size,
            max_length=stage3_max_length,
            lr=stage3_lr,
            save_path=stage3_save_path,
            load_path=stage2_save_path,
            temperature=stage3_temperature,
        )
        print(f"[Stage 3] Finished, best validation loss: {best_val_loss:.6f}")
        swanlab.finish()
    
    print("\n" + "=" * 80)
    print("Distillation pipeline finished")
    print("=" * 80)
    print(f"\nModel files saved under: {save_dir}")
    print(f"  - Stage 1 model: {stage1_save_path}")
    print(f"  - Stage 2 model: {stage2_save_path}")
    print(f"  - Stage 3 model: {stage3_save_path}")

