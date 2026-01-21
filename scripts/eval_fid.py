import argparse
import json
import os

import torch
from diffusers import DDPMScheduler, DDIMScheduler
from torchmetrics.image.fid import FrechetInceptionDistance

from config_utils import load_config, resolve_device
from data_loader import get_dataloader
from generate import load_model, resolve_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Compute FID on IAM samples.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", help="Checkpoint path or filename in save_dir.")
    parser.add_argument("--epoch", type=int, help="Select checkpoint by epoch.")
    parser.add_argument("--num-samples", type=int, default=512, help="Number of samples to score.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--steps", type=int, help="Diffusion steps override.")
    parser.add_argument("--scheduler", choices=["ddpm", "ddim"], help="Sampler override.")
    parser.add_argument("--guidance-scale", type=float, help="CFG guidance scale.")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA weights.")
    parser.add_argument("--device", type=str, help="Device override (cuda, mps, cpu).")
    parser.add_argument("--output", default="benchmarks/fid.json", help="Output JSON path.")
    return parser.parse_args()


def build_scheduler(model, scheduler_name, num_steps, device):
    scheduler_name = scheduler_name.lower()
    if scheduler_name == "ddim":
        scheduler = DDIMScheduler.from_config(model.scheduler.config)
    else:
        scheduler = DDPMScheduler.from_config(model.scheduler.config)
    scheduler.set_timesteps(num_steps)
    if hasattr(scheduler, "timesteps"):
        scheduler.timesteps = scheduler.timesteps.to(device)
    return scheduler


def trim_batch(batch, max_items):
    if max_items is None:
        return batch
    trimmed = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            trimmed[key] = value[:max_items]
        elif isinstance(value, list):
            trimmed[key] = value[:max_items]
        else:
            trimmed[key] = value
    return trimmed


@torch.no_grad()
def sample_batch(model, batch, scheduler, guidance_scale, device):
    input_ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    style_images = batch["style_pixel_values"].to(device)

    text_emb = model.text_encoder(input_ids, attention_mask=mask)[0]
    style_emb = model.style_encoder(style_images)

    if guidance_scale and guidance_scale > 1.0:
        uncond_text = torch.zeros_like(text_emb)
        uncond_style = torch.zeros_like(style_emb)

    bsz = input_ids.shape[0]
    latents = torch.randn(
        (bsz, model.sample_channels, model.sample_size[0], model.sample_size[1]),
        device=device,
    )

    for t in scheduler.timesteps:
        if guidance_scale and guidance_scale > 1.0:
            latents_in = torch.cat([latents, latents], dim=0)
            text_in = torch.cat([uncond_text, text_emb], dim=0)
            style_in = torch.cat([uncond_style, style_emb], dim=0)
            noise_pred = model.unet(
                latents_in,
                t,
                encoder_hidden_states=text_in,
                class_labels=style_in,
            ).sample
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = model.unet(
                latents,
                t,
                encoder_hidden_states=text_emb,
                class_labels=style_emb,
            ).sample

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    if model.latent_enabled:
        images = model.autoencoder.decode(latents)
    else:
        images = latents
    return images


def to_uint8(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    return (images * 255).to(torch.uint8)


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.steps is not None:
        config["generate"]["num_steps"] = args.steps
    if args.scheduler is not None:
        config["generate"]["scheduler"] = args.scheduler
    if args.guidance_scale is not None:
        config["generate"]["guidance_scale"] = args.guidance_scale
    if args.no_ema:
        config["generate"]["use_ema"] = False
    if args.device is not None:
        config["train"]["device"] = args.device

    device = resolve_device(config["train"].get("device", "auto"))
    config["train"]["device"] = device

    checkpoint_path = resolve_checkpoint(
        config["train"]["save_dir"], args.checkpoint, args.epoch
    )
    model = load_model(device, checkpoint_path, config)

    scheduler = build_scheduler(
        model,
        config["generate"]["scheduler"],
        config["generate"]["num_steps"],
        device,
    )

    dataloader = get_dataloader(config, batch_size=args.batch_size, shuffle=True)
    if len(dataloader) == 0:
        raise RuntimeError("FID dataloader is empty.")

    metric = FrechetInceptionDistance(feature=2048).to(device)

    loader_iter = iter(dataloader)
    samples_processed = 0
    while samples_processed < args.num_samples:
        remaining = args.num_samples - samples_processed
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dataloader)
            batch = next(loader_iter)
        batch = trim_batch(batch, remaining)

        real_images = batch["pixel_values"].to(device)
        fake_images = sample_batch(
            model,
            batch,
            scheduler,
            config["generate"].get("guidance_scale", 1.0),
            device,
        )

        metric.update(to_uint8(real_images), real=True)
        metric.update(to_uint8(fake_images), real=False)
        samples_processed += batch["input_ids"].shape[0]

    fid_score = metric.compute().item()

    payload = {
        "device": device,
        "num_samples": int(samples_processed),
        "batch_size": int(args.batch_size),
        "num_steps": config["generate"]["num_steps"],
        "scheduler": config["generate"]["scheduler"],
        "guidance_scale": config["generate"].get("guidance_scale"),
        "fid": fid_score,
    }
    if device == "cuda":
        payload["gpu_name"] = torch.cuda.get_device_name(0)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"Wrote FID results to {args.output}")


if __name__ == "__main__":
    main()
