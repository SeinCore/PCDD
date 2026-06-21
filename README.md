# PCDD: Perception-Cognition Dual-driven Detector for Short Video Fake News Detection

Implementation of **PCDD**, a three-stage framework for detecting fake news in short videos through multimodal perception-cognition dual-driven learning.

## Overview

PCDD tackles short-video fake news detection through three stages:

1. **Perceptual Clue Discovery** — masked multimodal reconstruction captures cross-modal discrepancy by selectively masking one modality and reconstructing it from the other two.
2. **Cognitive Logic Reasoning** — three-stage distillation transfers LLM reasoning about commonsense violation, logical fallacy, and emotional manipulation into a lightweight text encoder.
3. **Adaptive Fusion** — gated cross-attention combines perceptual discrepancy features with cognitive reasoning features for binary classification.

## Source Code Structure

```
config.py                         # Paths, hyperparameters, defaults
data_preprocess/
  video_preprocess.py             # Metadata → splits, frames, audio, OCR, ASR
  deepseek_analyse.py             # Transcript → teacher annotations (DeepSeek)
distill/distill.py                # Cognitive distillation (3 stages)
reconstruct/
  reconstruct.py                  # Masked multimodal reconstruction
  reconstruction_model.py
  modeling_qwen2_5_omni.py
fusion/
  train_fusion.py                 # Gated fusion classifier
  fusion_model.py
run_main_pipeline.py              # Entrypoint: distill / reconstruct / fusion
requirements.txt                  # Pinned Python dependencies
```

## Datasets

Due to copyright restrictions, raw videos are not included. Download them from the original sources:

| Dataset | Language | Source |
|---------|----------|--------|
| FakeSV  | Chinese  | [ICTMCG/FakeSV](https://github.com/ICTMCG/FakeSV) (AAAI 2023) |
| FakeTT  | English  | [ICTMCG/FakingRecipe](https://github.com/ICTMCG/FakingRecipe) (ACM MM 2024) |

Splits are chronological (70/15/15 train/val/test) by publish timestamp.

Place the downloaded files as follows (matching the default paths in `config.py`):

- FakeSV: `data_complete.json` and the video directory under `FAKESV_ROOT`
- FakeTT: `data.json` under `FAKETT_ROOT`, video directory under `video/` inside it

## Models

Download the required models into `models/` under the project root:

```bash
# (Optional) For users in mainland China, set a mirror first:
# export HF_ENDPOINT=https://hf-mirror.com

hf download Qwen/Qwen2.5-Omni-3B --local-dir models/Qwen2.5-Omni-3B
hf download Qwen/Qwen3-Embedding-0.6B --local-dir models/Qwen3-Embedding-0.6B
hf download openai/whisper-large-v3 --local-dir models/whisper-large-v3
```

## Setup

### Requirements

- Python 3.10, CUDA 12.1+, ffmpeg

```bash
conda create -n pcdd python=3.10.16 -y
conda activate pcdd
pip install -r requirements.txt
```

### cuDNN

PaddlePaddle looks for `libcudnn.so` but nvidia-cudnn-cu12 ships `libcudnn.so.9`. Run this once:

```bash
CUDNN_LIB=$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0]+'/lib')")
ln -sf libcudnn.so.9 "$CUDNN_LIB/libcudnn.so"
export LD_LIBRARY_PATH="$CUDNN_LIB:$LD_LIBRARY_PATH"
```

## Preprocessing

### 1. Generate splits, frames, audio, OCR, ASR

```bash
python data_preprocess/video_preprocess.py --config config.py --dataset fakesv --stage all
python data_preprocess/video_preprocess.py --config config.py --dataset fakett --stage all
```

### 2. Generate teacher annotations

```bash
export PCDD_DEEPSEEK_API_KEY=<your_deepseek_api_key>
python data_preprocess/deepseek_analyse.py --config config.py --dataset fakesv --mode all
python data_preprocess/deepseek_analyse.py --config config.py --dataset fakett --mode all
```

## Training and Evaluation

Each stage can be run independently, or the full pipeline at once:

```bash
# Individual stages
python run_main_pipeline.py --config config.py --dataset fakesv --stage distill
python run_main_pipeline.py --config config.py --dataset fakesv --stage reconstruct
python run_main_pipeline.py --config config.py --dataset fakesv --stage fusion

# Or the full pipeline
python run_main_pipeline.py --config config.py --dataset fakesv --stage all
```

Replace `fakesv` with `fakett` for the FakeTT dataset.

After fusion training finishes, the best checkpoint is automatically evaluated on the held-out test set — per-class precision, recall, and F1 are printed.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
