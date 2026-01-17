# Conditional Handwriting Diffusion Model

This project implements a conditional diffusion model that generates handwriting images from text prompts while conditioning on a reference handwriting style image. It uses Hugging Face `diffusers` for the UNet backbone and `transformers` for the text encoder.

## Setup

1. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Optional (not required for training or generation):

```bash
pip install -r requirements-dev.txt
```

## Dataset (IAM)

This project expects the IAM Handwriting Database (words):

```
project/
├── iam_data/
│   ├── words.txt
│   └── words/
│       ├── a01/
│       │   ├── a01-000u/
│       │   │   └── ...png
│       └── ...
```

The data loader skips lines marked `err` and replaces `|` with spaces in transcriptions.

## Configuration

Defaults live in `config.yaml`. You can edit the YAML directly or override from the CLI.

Key sections:
- `data`: dataset root, image size, max token length, writer filtering
- `model`: text encoder name, style embedding size, scheduler settings
- `train`: batch size, epochs, checkpoint frequency, device selection
- `generate`: output directory and sampler choice

## Usage

Inspect a few samples:

```bash
python inspect_data.py --config config.yaml
```

Train (auto-resumes from the latest checkpoint if available):

```bash
python train.py --config config.yaml
```

Generate a sample:

```bash
python generate.py --config config.yaml --text "hello world" --style a01
```

Generate with classifier-free guidance and style mixing:

```bash
python generate.py --config config.yaml --text "hello world" --style a01 --style-b a02 --style-mix 0.4 --guidance-scale 3.0
```

Pick a specific checkpoint:

```bash
python generate.py --config config.yaml --epoch 10 --text "test" --style /path/to/style.png
```

Run interactively:

```bash
python generate.py --config config.yaml --interactive
```

View training metrics (TensorBoard):

```bash
tensorboard --logdir ./logs
```

## Notes

- The training checkpoint format now stores optimizer and scaler state for reliable resume.
- Training supports cosine schedules, EMA weights, CFG dropout, gradient clipping, and optional min-SNR loss reweighting via `config.yaml`.
- Style conditioning can be strengthened via writer-ID classification and supervised contrastive loss (weights configurable in `config.yaml`).
- Validation logging includes loss curves, sample grids, and CLIP text-image similarity when enabled.
- Generated samples are saved to `generated_outputs/` by default.
- Set `train.device` in `config.yaml` to `cuda`, `mps`, or `cpu` if you want to override auto-detection.
- To use EMA weights for inference, keep `generate.use_ema: true` in `config.yaml`.
- Latent diffusion is enabled in `config.yaml`; keep `train.autoencoder_recon_weight > 0` so the decoder learns to reconstruct images.
- If you run offline, set `train.clip_metric: false` to skip CLIP downloads.
