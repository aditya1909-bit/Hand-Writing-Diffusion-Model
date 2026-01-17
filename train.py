import argparse
import glob
import math
import os
import random
import re

import torch
from torch import amp
from tqdm import tqdm
from transformers import get_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from diffusers import DDPMScheduler

from config_utils import load_config, resolve_device
from data_loader import get_dataloaders
from ema_utils import EMA
from model import HandwritingDiffusionSystem
from clip_utils import ClipScorer


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sanitize_state_dict(state_dict):
    if any(key.startswith("module.") for key in state_dict.keys()):
        return {key[7:]: value for key, value in state_dict.items()}
    return state_dict


def find_latest_checkpoint(save_dir):
    if not os.path.exists(save_dir):
        return None, -1

    patterns = [
        os.path.join(save_dir, "checkpoint_epoch_*.pth"),
        os.path.join(save_dir, "model_epoch_*.pth"),
    ]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))

    if not paths:
        return None, -1

    def rank(path):
        match = re.search(r"epoch_(\d+)", os.path.basename(path))
        epoch = int(match.group(1)) if match else -1
        is_full = 1 if os.path.basename(path).startswith("checkpoint_") else 0
        return (epoch, is_full)

    best_path = max(paths, key=rank)
    best_epoch = rank(best_path)[0]
    return best_path, best_epoch


def load_checkpoint(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state["model_state"] = _sanitize_state_dict(state["model_state"])
        return state
    if isinstance(state, dict) and "state_dict" in state:
        state["state_dict"] = _sanitize_state_dict(state["state_dict"])
        return {"model_state": state["state_dict"]}
    if isinstance(state, dict) and "model" in state:
        return {"model_state": _sanitize_state_dict(state["model"])}
    return {"model_state": _sanitize_state_dict(state)}


def save_checkpoint(path, model, optimizer, scaler, epoch, config, ema_state=None):
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler else None,
        "ema_state": ema_state,
        "config": config,
    }
    torch.save(payload, path)


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


def reduce_loss(value):
    if value is None:
        return torch.tensor(0.0)
    if torch.is_tensor(value):
        return value.mean()
    return torch.tensor(value)


def build_eval_scheduler(model, num_steps, device):
    scheduler = DDPMScheduler.from_config(model.scheduler.config)
    scheduler.set_timesteps(num_steps)
    if hasattr(scheduler, "timesteps"):
        scheduler.timesteps = scheduler.timesteps.to(device)
    return scheduler


@torch.no_grad()
def sample_from_batch(model, batch, device, num_steps, guidance_scale):
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

    scheduler = build_eval_scheduler(model, num_steps, device)
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


def apply_ema_weights(model, ema):
    if not ema:
        return None
    backup = {key: value.detach().clone() for key, value in model.state_dict().items()}
    ema.copy_to(model)
    return backup


def restore_weights(model, backup):
    if backup:
        model.load_state_dict(backup, strict=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a handwriting diffusion model.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--epochs", type=int, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--lr", type=float, help="Override learning rate.")
    parser.add_argument("--save-dir", type=str, help="Override checkpoint directory.")
    parser.add_argument("--save-every", type=int, help="Override save frequency.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume.")
    parser.add_argument("--device", type=str, help="Device override (cuda, mps, cpu).")
    parser.add_argument("--num-writers", type=int, help="Limit dataset to the first N writers.")
    parser.add_argument("--mock", action="store_true", help="Enable mock mode for quick smoke tests.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["train"]["lr"] = args.lr
    if args.save_dir is not None:
        config["train"]["save_dir"] = args.save_dir
    if args.save_every is not None:
        config["train"]["save_every"] = args.save_every
    if args.device is not None:
        config["train"]["device"] = args.device
    if args.num_writers is not None:
        config["data"]["num_writers"] = args.num_writers
    if args.mock:
        config["data"]["mock_mode"] = True
    if args.no_resume:
        config["train"]["resume"] = False
    if args.resume:
        config["train"]["resume"] = True

    device = resolve_device(config["train"].get("device", "auto"))
    config["train"]["device"] = device

    set_seed(config["data"].get("seed", 1337))

    os.makedirs(config["train"]["save_dir"], exist_ok=True)
    print("--- Training Configuration ---")
    print(f"Device: {device.upper()}")

    if device == "cuda":
        if config["train"].get("tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = config["train"].get("cudnn_benchmark", True)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if config["train"].get("val_split_by_writer") and (
        config["train"].get("style_cls_weight", 0.0) > 0
        or config["train"].get("style_contrastive_weight", 0.0) > 0
    ):
        print("Warning: val_split_by_writer=True with style losses may under-train some writers.")

    train_loader, val_loader, writer_count = get_dataloaders(config)
    if len(train_loader) == 0:
        raise RuntimeError("Training dataloader is empty. Check your dataset path and filters.")

    model_system = HandwritingDiffusionSystem(
        image_size=tuple(config["data"]["image_size"]),
        device=device,
        text_encoder_name=config["model"]["text_encoder"],
        style_dim=config["model"]["style_dim"],
        scheduler_config=config["model"]["scheduler"],
        min_snr_gamma=config["train"].get("min_snr_gamma"),
        num_writers=writer_count,
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
    if config["train"].get("channels_last", False) and device == "cuda":
        model_system = model_system.to(memory_format=torch.channels_last)

    if config["train"].get("gradient_checkpointing", False):
        model_system.unet.enable_gradient_checkpointing()

    latest_path, last_epoch = (None, -1)
    if config["train"]["resume"]:
        latest_path, last_epoch = find_latest_checkpoint(config["train"]["save_dir"])

    start_epoch = 0
    optimizer_state = None
    scaler_state = None
    checkpoint = None
    if latest_path:
        print(f"FOUND CHECKPOINT: {latest_path}")
        checkpoint = load_checkpoint(latest_path, device)
        model_system.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer_state = checkpoint.get("optimizer_state")
        scaler_state = checkpoint.get("scaler_state")
        start_epoch = checkpoint.get("epoch", last_epoch) + 1
        print(f"Resuming training from Epoch {start_epoch}")
    else:
        print("No checkpoints found. Starting from scratch.")

    use_data_parallel = device == "cuda" and torch.cuda.device_count() > 1
    if use_data_parallel:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model_system = torch.nn.DataParallel(model_system)

    base_model = model_system.module if isinstance(model_system, torch.nn.DataParallel) else model_system
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
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    amp_enabled = config["train"]["amp"] and device == "cuda" and torch.cuda.is_available()
    if amp_enabled:
        try:
            scaler = amp.GradScaler(device="cuda")
        except TypeError:
            scaler = amp.GradScaler()
    else:
        scaler = None
    if scaler and scaler_state:
        scaler.load_state_dict(scaler_state)

    grad_accum_steps = max(1, int(config["train"].get("grad_accum_steps", 1)))
    log_every = max(1, int(config["train"].get("log_every", 25)))
    max_grad_norm = config["train"].get("max_grad_norm", 0.0)

    if config["train"].get("compile", False) and not use_data_parallel and device == "cuda":
        compile_mode = config["train"].get("compile_mode", "max-autotune")
        try:
            base_model.unet = torch.compile(base_model.unet, mode=compile_mode)
        except Exception as exc:
            print(f"torch.compile skipped: {exc}")

    ema = None
    if config["train"].get("ema", False):
        ema = EMA(base_model, decay=config["train"].get("ema_decay", 0.9999))
        if latest_path:
            ema_state = checkpoint.get("ema_state")
            if ema_state:
                ema.load_state_dict(ema_state)

    lr_scheduler_name = config["train"].get("lr_scheduler", "cosine")
    num_update_steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    max_train_steps = config["train"]["epochs"] * num_update_steps_per_epoch
    warmup_steps = int(config["train"].get("warmup_steps", 0))
    if warmup_steps <= 0:
        warmup_ratio = float(config["train"].get("warmup_ratio", 0.0))
        warmup_steps = int(max_train_steps * warmup_ratio)
    scheduler = None
    if lr_scheduler_name and lr_scheduler_name.lower() != "none":
        scheduler = get_scheduler(
            lr_scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
        )

    writer = SummaryWriter(config["train"]["log_dir"])
    clip_scorer = None
    if config["train"].get("clip_metric", False):
        try:
            clip_scorer = ClipScorer(config["train"]["clip_model"], device)
        except Exception as exc:
            print(f"CLIP metric disabled: {exc}")
            clip_scorer = None

    print("Starting training loop...")
    global_step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, config["train"]["epochs"]):
        model_system.train()
        epoch_loss = 0.0
        epoch_diffusion = 0.0
        epoch_style_cls = 0.0
        epoch_style_contrastive = 0.0
        epoch_recon = 0.0

        num_batches = len(train_loader)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['train']['epochs']}", leave=True)
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress_bar, start=1):
            if not use_data_parallel:
                batch = move_batch_to_device(
                    batch,
                    device,
                    channels_last=config["train"].get("channels_last", False),
                )
            is_update_step = (step % grad_accum_steps == 0) or (step == num_batches)

            if amp_enabled:
                with amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model_system(batch)
                    loss = outputs["loss"].mean()
                scaled_loss = loss / grad_accum_steps
                scaler.scale(scaled_loss).backward()
                if is_update_step:
                    if max_grad_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    if ema:
                        ema.update(base_model)
            else:
                outputs = model_system(batch)
                loss = outputs["loss"].mean()
                (loss / grad_accum_steps).backward()
                if is_update_step:
                    if max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_grad_norm)
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    if ema:
                        ema.update(base_model)

            diffusion_loss = reduce_loss(outputs.get("diffusion_loss", None))
            style_cls_loss = reduce_loss(outputs.get("style_cls_loss", None))
            style_contrastive_loss = reduce_loss(outputs.get("style_contrastive_loss", None))
            recon_loss = reduce_loss(outputs.get("recon_loss", None))

            epoch_loss += loss.item()
            epoch_diffusion += diffusion_loss.item() if diffusion_loss is not None else 0.0
            epoch_style_cls += style_cls_loss.item() if style_cls_loss is not None else 0.0
            epoch_style_contrastive += (
                style_contrastive_loss.item() if style_contrastive_loss is not None else 0.0
            )
            epoch_recon += recon_loss.item() if recon_loss is not None else 0.0

            if is_update_step:
                global_step += 1
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar(
                    "train/diffusion_loss",
                    diffusion_loss.item() if diffusion_loss is not None else 0.0,
                    global_step,
                )
                if style_cls_loss is not None:
                    writer.add_scalar("train/style_cls_loss", style_cls_loss.item(), global_step)
                if style_contrastive_loss is not None:
                    writer.add_scalar(
                        "train/style_contrastive_loss",
                        style_contrastive_loss.item(),
                        global_step,
                    )
                if recon_loss is not None:
                    writer.add_scalar("train/recon_loss", recon_loss.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            if step % log_every == 0:
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(train_loader)
        avg_diffusion = epoch_diffusion / len(train_loader)
        avg_style_cls = epoch_style_cls / len(train_loader)
        avg_style_contrastive = epoch_style_contrastive / len(train_loader)
        avg_recon = epoch_recon / len(train_loader)
        print(f"--- Epoch {epoch} Finished. Avg Loss: {avg_loss:.4f} ---")
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        writer.add_scalar("train/epoch_diffusion_loss", avg_diffusion, epoch)
        writer.add_scalar("train/epoch_style_cls_loss", avg_style_cls, epoch)
        writer.add_scalar("train/epoch_style_contrastive_loss", avg_style_contrastive, epoch)
        writer.add_scalar("train/epoch_recon_loss", avg_recon, epoch)

        if val_loader and ((epoch + 1) % config["train"].get("val_every", 1) == 0):
            eval_model = base_model
            eval_model.eval()
            ema_backup = None
            if config["train"].get("eval_use_ema", False):
                ema_backup = apply_ema_weights(eval_model, ema)

            val_loss = 0.0
            val_diffusion = 0.0
            val_style_cls = 0.0
            val_style_contrastive = 0.0
            val_recon = 0.0
            val_batches = 0
            max_batches = config["train"].get("val_max_batches", 0)

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if max_batches and batch_idx >= max_batches:
                        break
                    batch = move_batch_to_device(
                        batch,
                        device,
                        channels_last=config["train"].get("channels_last", False),
                    )
                    outputs = eval_model(batch)
                    val_loss += reduce_loss(outputs["loss"]).item()
                    val_diffusion += reduce_loss(outputs["diffusion_loss"]).item()
                    val_style_cls += reduce_loss(outputs.get("style_cls_loss")).item()
                    val_style_contrastive += reduce_loss(outputs.get("style_contrastive_loss")).item()
                    val_recon += reduce_loss(outputs.get("recon_loss")).item()
                    val_batches += 1

            if val_batches > 0:
                val_loss /= val_batches
                val_diffusion /= val_batches
                val_style_cls /= val_batches
                val_style_contrastive /= val_batches
                val_recon /= val_batches
                writer.add_scalar("val/loss", val_loss, epoch)
                writer.add_scalar("val/diffusion_loss", val_diffusion, epoch)
                writer.add_scalar("val/style_cls_loss", val_style_cls, epoch)
                writer.add_scalar("val/style_contrastive_loss", val_style_contrastive, epoch)
                writer.add_scalar("val/recon_loss", val_recon, epoch)

            if val_loader and ((epoch + 1) % config["train"].get("log_images_every", 1) == 0):
                log_count = int(config["train"].get("log_images", 8))
                if log_count > 0:
                    eval_batch = next(iter(val_loader))
                    eval_batch = move_batch_to_device(
                        eval_batch,
                        device,
                        channels_last=config["train"].get("channels_last", False),
                    )
                    eval_batch = {
                        key: value[:log_count] if torch.is_tensor(value) else value[:log_count]
                        for key, value in eval_batch.items()
                    }
                    samples = sample_from_batch(
                        eval_model,
                        eval_batch,
                        device,
                        num_steps=int(config["train"].get("eval_steps", 30)),
                        guidance_scale=float(config["train"].get("eval_guidance_scale", 1.0)),
                    )
                    samples = (samples / 2 + 0.5).clamp(0, 1)
                    grid = make_grid(samples.detach().cpu(), nrow=min(4, log_count))
                    writer.add_image("val/samples", grid, epoch)

                    if clip_scorer and ((epoch + 1) % config["train"].get("clip_every", 1) == 0):
                        texts = eval_batch["text"]
                        clip_scores = clip_scorer.score(samples * 2 - 1, texts)
                        writer.add_scalar("val/clip_similarity", clip_scores.mean().item(), epoch)

            restore_weights(eval_model, ema_backup)

        if (epoch + 1) % config["train"]["save_every"] == 0:
            save_path = os.path.join(config["train"]["save_dir"], f"checkpoint_epoch_{epoch}.pth")
            save_checkpoint(
                save_path,
                base_model,
                optimizer,
                scaler,
                epoch,
                config,
                ema_state=ema.state_dict() if ema else None,
            )
            print(f"Checkpoint saved to {save_path}")

    writer.close()


if __name__ == "__main__":
    main()
