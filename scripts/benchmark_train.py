import argparse
import json
import os
import time

import torch
from torch import amp

from config_utils import load_config, resolve_device
from data_loader import get_dataloader
from ema_utils import EMA
from model import HandwritingDiffusionSystem


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark training throughput.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--steps", type=int, default=100, help="Timed training steps.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--device", type=str, help="Device override (cuda, mps, cpu).")
    parser.add_argument("--output", default="benchmarks/train.json", help="Output JSON path.")
    return parser.parse_args()


def move_batch_to_device(batch, device, channels_last=False):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved_value = value.to(device, non_blocking=True)
            if channels_last and moved_value.ndim == 4:
                moved_value = moved_value.to(memory_format=torch.channels_last)
            moved[key] = moved_value
        else:
            moved[key] = value
    return moved


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.device is not None:
        config["train"]["device"] = args.device

    device = resolve_device(config["train"].get("device", "auto"))
    config["train"]["device"] = device

    batch_size = config["train"]["batch_size"]
    data_loader = get_dataloader(config, batch_size=batch_size, shuffle=True)
    if len(data_loader) == 0:
        raise RuntimeError("Benchmark dataloader is empty.")

    if device == "cuda":
        if config["train"].get("tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = config["train"].get("cudnn_benchmark", True)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    model_system = HandwritingDiffusionSystem(
        image_size=tuple(config["data"]["image_size"]),
        device=device,
        text_encoder_name=config["model"]["text_encoder"],
        style_dim=config["model"]["style_dim"],
        scheduler_config=config["model"]["scheduler"],
        min_snr_gamma=config["train"].get("min_snr_gamma"),
        num_writers=None,
        text_drop_prob=config["train"].get("text_drop_prob", 0.0),
        style_drop_prob=config["train"].get("style_drop_prob", 0.0),
        cond_drop_prob=config["train"].get("cond_drop_prob", 0.0),
        style_cls_weight=config["train"].get("style_cls_weight", 0.0),
        style_contrastive_weight=config["train"].get("style_contrastive_weight", 0.0),
        style_contrastive_temperature=config["train"].get("style_contrastive_temperature", 0.07),
        latent_enabled=config["model"].get("latent", {}).get("enabled", False),
        latent_channels=config["model"].get("latent", {}).get("latent_channels", 4),
        latent_downsample_factor=config["model"].get("latent", {}).get("downsample_factor", 4),
        autoencoder_recon_weight=config["train"].get("autoencoder_recon_weight", 0.0),
    ).to(device)
    model_system.train()

    if config["train"].get("channels_last", False) and device == "cuda":
        model_system = model_system.to(memory_format=torch.channels_last)

    if config["train"].get("gradient_checkpointing", False):
        model_system.unet.enable_gradient_checkpointing()

    base_model = model_system
    optim_params = list(base_model.unet.parameters()) + list(base_model.style_encoder.parameters())
    if base_model.style_classifier is not None:
        optim_params += list(base_model.style_classifier.parameters())
    if base_model.latent_enabled and base_model.autoencoder is not None:
        optim_params += list(base_model.autoencoder.parameters())

    optimizer = torch.optim.AdamW(
        optim_params,
        lr=config["train"]["lr"],
        weight_decay=config["train"].get("weight_decay", 0.0),
        betas=tuple(config["train"].get("betas", (0.9, 0.999))),
    )

    amp_enabled = config["train"]["amp"] and device == "cuda" and torch.cuda.is_available()
    if amp_enabled:
        try:
            scaler = amp.GradScaler(device="cuda")
        except TypeError:
            scaler = amp.GradScaler()
    else:
        scaler = None

    ema = None
    if config["train"].get("ema", False):
        ema = EMA(base_model, decay=config["train"].get("ema_decay", 0.9999))

    loader_iter = iter(data_loader)

    def next_batch():
        nonlocal loader_iter
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(data_loader)
            batch = next(loader_iter)
        return batch

    def train_step(batch):
        batch = move_batch_to_device(
            batch,
            device,
            channels_last=config["train"].get("channels_last", False),
        )
        if amp_enabled:
            with amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model_system(batch)
                loss = outputs["loss"].mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model_system(batch)
            loss = outputs["loss"].mean()
            loss.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema:
            ema.update(base_model)

    for _ in range(max(0, args.warmup)):
        train_step(next_batch())

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    for _ in range(max(1, args.steps)):
        train_step(next_batch())

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    steps = max(1, args.steps)
    steps_per_sec = steps / elapsed if elapsed > 0 else 0.0
    samples_per_sec = steps_per_sec * batch_size

    payload = {
        "device": device,
        "batch_size": batch_size,
        "steps": steps,
        "warmup_steps": max(0, args.warmup),
        "elapsed_sec": elapsed,
        "steps_per_sec": steps_per_sec,
        "samples_per_sec": samples_per_sec,
        "amp": bool(amp_enabled),
        "channels_last": bool(config["train"].get("channels_last", False)),
        "num_workers": int(config["train"].get("num_workers", 0)),
        "cache_tokens": bool(config["data"].get("cache_tokens", False)),
        "cache_images": bool(config["data"].get("cache_images", False)),
        "scheduler": config["model"].get("scheduler", {}),
    }
    if device == "cuda":
        payload["gpu_name"] = torch.cuda.get_device_name(0)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"Wrote training benchmark to {args.output}")


if __name__ == "__main__":
    main()
