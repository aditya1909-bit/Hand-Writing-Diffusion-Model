# Conditional Handwriting Diffusion Model

This project implements a conditional diffusion model that generates handwriting images from text prompts while conditioning on a reference handwriting style image. It uses a frozen BERT text encoder, a CNN style encoder, and a DDPM-based UNet backbone via Hugging Face `diffusers`.

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
- `data`: dataset root, image size, max token length, writer filtering, cache paths
- `model`: text encoder name, style embedding size, scheduler settings (including custom noise schedules)
- `train`: batch size, epochs, checkpoint frequency, device selection, mixed precision
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

## Benchmarking

Precompute caches (token + image tensors):

```bash
python scripts/preprocess_iam.py --cache-images --cache-tokens
```

Dataset stats:

```bash
python scripts/dataset_stats.py
```

Training throughput:

```bash
python scripts/benchmark_train.py
```

Inference throughput:

```bash
python scripts/benchmark_infer.py --checkpoint /path/to/checkpoint.pth
```

FID evaluation:

```bash
python scripts/eval_fid.py --checkpoint /path/to/checkpoint.pth
```

Update the benchmark summary in this README:

```bash
python scripts/report_benchmarks.py --write-readme
```

## Benchmarks

- Dataset: 
- FID (IAM): 
- Train throughput: 
- Inference throughput: 

## Notes

- The training checkpoint format now stores optimizer and scaler state for reliable resume.
- Training supports EMA weights, CFG dropout, gradient clipping, optional min-SNR loss reweighting, and custom noise schedules via `config.yaml`.
- Set `model.scheduler.custom_beta_schedule` to `linear`, `quadratic`, `sigmoid`, or `cosine` to enable custom schedules.
- Data preprocessing can cache tokenized text and resized image tensors for faster IO-bound runs.
- Style conditioning can be strengthened via writer-ID classification and supervised contrastive loss (weights configurable in `config.yaml`).
- Validation logging includes loss curves, sample grids, and CLIP text-image similarity when enabled.
- Generated samples are saved to `generated_outputs/` by default.
- Set `train.device` in `config.yaml` to `cuda`, `mps`, or `cpu` if you want to override auto-detection.
- To use EMA weights for inference, keep `generate.use_ema: true` in `config.yaml`.
- Latent diffusion is enabled in `config.yaml`; keep `train.autoencoder_recon_weight > 0` so the decoder learns to reconstruct images.
- If you run offline, set `train.clip_metric: false` to skip CLIP downloads.
