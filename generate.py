import argparse
import os
import re
import sys

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
from diffusers import DDPMScheduler, DDIMScheduler
from transformers import BertTokenizerFast
from torchvision import transforms
from PIL import Image

from config_utils import load_config, resolve_device
from image_utils import resize_with_pad
from model import HandwritingDiffusionSystem


def find_image_in_path(user_input, data_root):
    if os.path.isfile(user_input):
        return user_input

    search_path = user_input
    if not os.path.exists(search_path):
        candidate = os.path.join(data_root, user_input)
        if os.path.exists(candidate):
            search_path = candidate

    if os.path.isdir(search_path):
        print(f"Searching inside folder: {search_path}...")
        for root, _, files in os.walk(search_path):
            for filename in files:
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    found_path = os.path.join(root, filename)
                    print(f"-> Auto-selected style image: {filename}")
                    return found_path

    return None


def load_model(device, model_path, config):
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    num_writers = None
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state_dict = state["model_state"]
    elif isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state_dict = state["model"]
    else:
        state_dict = state
    if "style_classifier.weight" in state_dict:
        num_writers = state_dict["style_classifier.weight"].shape[0]

    system = HandwritingDiffusionSystem(
        image_size=tuple(config["data"]["image_size"]),
        device=device,
        text_encoder_name=config["model"]["text_encoder"],
        style_dim=config["model"]["style_dim"],
        scheduler_config=config["model"]["scheduler"],
        num_writers=num_writers,
        latent_enabled=config["model"].get("latent", {}).get("enabled", False),
        latent_channels=config["model"].get("latent", {}).get("latent_channels", 4),
        latent_downsample_factor=config["model"].get("latent", {}).get("downsample_factor", 4),
    )

    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key[7:]: value for key, value in state_dict.items()}

    try:
        if config["generate"].get("use_ema", False) and isinstance(state, dict):
            ema_state = state.get("ema_state")
            if ema_state and "shadow" in ema_state:
                state_dict = ema_state["shadow"]
        system.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        print(f"Error loading weights: {exc}")
        sys.exit(1)

    system.to(device)
    system.eval()
    print("Model loaded successfully.")
    return system


def build_scheduler(system, scheduler_name, num_steps, device):
    scheduler_name = scheduler_name.lower()
    if scheduler_name == "ddim":
        scheduler = DDIMScheduler.from_config(system.scheduler.config)
    else:
        scheduler = DDPMScheduler.from_config(system.scheduler.config)

    scheduler.set_timesteps(num_steps)
    if hasattr(scheduler, "timesteps"):
        scheduler.timesteps = scheduler.timesteps.to(device)
    return scheduler


def resolve_checkpoint(save_dir, checkpoint_arg, epoch_arg):
    if checkpoint_arg and os.path.isfile(checkpoint_arg):
        return checkpoint_arg

    if not os.path.isdir(save_dir):
        print(f"No checkpoint directory found at {save_dir}")
        sys.exit(1)

    paths = [p for p in os.listdir(save_dir) if p.endswith(".pth")]
    if not paths:
        print(f"No checkpoint files found in {save_dir}")
        sys.exit(1)

    def extract_epoch(name):
        match = re.search(r"epoch_(\d+)", name)
        return int(match.group(1)) if match else -1

    if epoch_arg is not None:
        for name in paths:
            if extract_epoch(name) == epoch_arg:
                return os.path.join(save_dir, name)
        print(f"No checkpoint found for epoch {epoch_arg}")
        sys.exit(1)

    if checkpoint_arg:
        for name in paths:
            if name == checkpoint_arg or os.path.basename(name) == checkpoint_arg:
                return os.path.join(save_dir, name)
        print(f"Checkpoint '{checkpoint_arg}' not found in {save_dir}")
        sys.exit(1)

    latest = max(paths, key=extract_epoch)
    return os.path.join(save_dir, latest)


def generate_handwriting(
    system,
    text,
    style_path,
    style_path_b,
    style_mix,
    device,
    config,
    scheduler,
    tokenizer,
    transform,
    guidance_scale,
):
    def load_style_tensor(path):
        try:
            style_img = Image.open(path).convert("RGB")
            style_img = resize_with_pad(style_img, config["data"]["image_size"], fill=config["data"]["pad_value"])
            return transform(style_img).unsqueeze(0).to(device)
        except Exception as exc:
            print(f"Error opening image '{path}': {exc}")
            return None

    style_tensor = load_style_tensor(style_path)
    if style_tensor is None:
        return None

    style_tensor_b = None
    if style_path_b:
        style_tensor_b = load_style_tensor(style_path_b)
        if style_tensor_b is None:
            return None

    text_inputs = tokenizer(
        text,
        padding="max_length",
        max_length=config["data"]["max_length"],
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    mask = text_inputs.attention_mask.to(device)

    with torch.no_grad():
        text_emb = system.text_encoder(input_ids, attention_mask=mask)[0]
        style_emb = system.style_encoder(style_tensor)
        if style_tensor_b is not None:
            style_emb_b = system.style_encoder(style_tensor_b)
            style_mix = max(0.0, min(1.0, float(style_mix)))
            style_emb = (1 - style_mix) * style_emb + style_mix * style_emb_b

    if guidance_scale and guidance_scale > 1.0:
        uncond_text = torch.zeros_like(text_emb)
        uncond_style = torch.zeros_like(style_emb)

    height, width = system.sample_size
    latents = torch.randn((1, system.sample_channels, height, width), device=device)

    print(f"Generating '{text}'...")
    for t in scheduler.timesteps:
        with torch.no_grad():
            if guidance_scale and guidance_scale > 1.0:
                latents_in = torch.cat([latents, latents], dim=0)
                text_in = torch.cat([uncond_text, text_emb], dim=0)
                style_in = torch.cat([uncond_style, style_emb], dim=0)
                noise_pred = system.unet(
                    latents_in,
                    t,
                    encoder_hidden_states=text_in,
                    class_labels=style_in,
                ).sample
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = system.unet(
                    latents,
                    t,
                    encoder_hidden_states=text_emb,
                    class_labels=style_emb,
                ).sample

            latents = scheduler.step(noise_pred, t, latents).prev_sample

    if system.latent_enabled:
        return system.autoencoder.decode(latents)
    return latents


def save_image(tensor, filename):
    image = tensor.squeeze(0).cpu().detach()
    image = (image / 2 + 0.5).clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(filename)
    print(f"Saved output to: {filename}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate handwriting from a trained checkpoint.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", help="Checkpoint path or filename in save_dir.")
    parser.add_argument("--epoch", type=int, help="Select checkpoint by epoch.")
    parser.add_argument("--text", help="Text prompt to render.")
    parser.add_argument("--style", help="Path/folder/id for style image.")
    parser.add_argument("--style-b", help="Optional second style image for mixing.")
    parser.add_argument("--style-mix", type=float, help="Mix ratio for style_b (0-1).")
    parser.add_argument("--output", help="Output filename or directory.")
    parser.add_argument("--steps", type=int, help="Number of diffusion steps.")
    parser.add_argument("--scheduler", choices=["ddpm", "ddim"], help="Scheduler type.")
    parser.add_argument("--guidance-scale", type=float, help="CFG guidance scale.")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA weights for sampling.")
    parser.add_argument("--device", type=str, help="Device override (cuda, mps, cpu).")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.steps is not None:
        config["generate"]["num_steps"] = args.steps
    if args.scheduler is not None:
        config["generate"]["scheduler"] = args.scheduler
    if args.style_mix is not None:
        config["generate"]["style_mix"] = args.style_mix
    if args.guidance_scale is not None:
        config["generate"]["guidance_scale"] = args.guidance_scale
    if args.no_ema:
        config["generate"]["use_ema"] = False
    if args.device is not None:
        config["train"]["device"] = args.device

    device = resolve_device(config["train"].get("device", "auto"))
    print(f"Running on device: {device.upper()}")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    checkpoint_path = resolve_checkpoint(
        config["train"]["save_dir"], args.checkpoint, args.epoch
    )

    system = load_model(device, checkpoint_path, config)
    scheduler = build_scheduler(system, config["generate"]["scheduler"], config["generate"]["num_steps"], device)
    tokenizer = BertTokenizerFast.from_pretrained(config["model"]["text_encoder"])
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    data_root = os.path.join(config["data"]["root_dir"], config["data"]["words_dir"])

    if args.interactive or not args.text or not args.style:
        print("\n--- Interactive Handwriting Generator ---")
        print("Press Ctrl+C to quit.\n")
        while True:
            try:
                text_input = input("Enter text to write: ").strip()
                if not text_input:
                    continue
                style_input = input("Enter path OR folder OR ID (e.g. a01): ").strip()
                style_input = style_input.replace("'", "").replace('"', "").strip()
                if not style_input:
                    continue
                style_path = find_image_in_path(style_input, data_root)
                if not style_path:
                    print(f"Could not find any images inside '{style_input}'")
                    continue
                style_path_b = None
                style_mix = float(config["generate"].get("style_mix", 0.5))
                style_input_b = input("Enter second style (optional): ").strip()
                style_input_b = style_input_b.replace("'", "").replace('"', "").strip()
                if style_input_b:
                    style_path_b = find_image_in_path(style_input_b, data_root)
                    if not style_path_b:
                        print(f"Could not find any images inside '{style_input_b}'")
                        continue
                    mix_input = input(f"Enter style mix ratio 0-1 [default {style_mix}]: ").strip()
                    if mix_input:
                        try:
                            style_mix = float(mix_input)
                        except ValueError:
                            print("Invalid mix ratio, using default.")
                run_single_generation(
                    system,
                    scheduler,
                    config,
                    device,
                    text_input,
                    style_path,
                    style_path_b,
                    style_mix,
                    float(config["generate"].get("guidance_scale", 1.0)),
                    args.output,
                    tokenizer,
                    transform,
                )
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as exc:
                print(f"An error occurred: {exc}")
    else:
        style_path = find_image_in_path(args.style, data_root)
        if not style_path:
            print(f"Could not find any images inside '{args.style}'")
            sys.exit(1)
        style_path_b = None
        if args.style_b:
            style_path_b = find_image_in_path(args.style_b, data_root)
            if not style_path_b:
                print(f"Could not find any images inside '{args.style_b}'")
                sys.exit(1)
        run_single_generation(
            system,
            scheduler,
            config,
            device,
            args.text,
            style_path,
            style_path_b,
            float(config["generate"].get("style_mix", 0.5)),
            float(config["generate"].get("guidance_scale", 1.0)),
            args.output,
            tokenizer,
            transform,
        )


def run_single_generation(
    system,
    scheduler,
    config,
    device,
    text,
    style_path,
    style_path_b,
    style_mix,
    guidance_scale,
    output_path,
    tokenizer,
    transform,
):
    result_tensor = generate_handwriting(
        system,
        text,
        style_path,
        style_path_b,
        style_mix,
        device,
        config,
        scheduler,
        tokenizer,
        transform,
        guidance_scale,
    )
    if result_tensor is None:
        return

    safe_text = "".join([c for c in text if c.isalnum() or c in (" ", "_")]).rstrip()
    safe_text = safe_text.replace(" ", "_") or "sample"

    if output_path:
        _, ext = os.path.splitext(output_path)
        if ext.lower() in (".png", ".jpg", ".jpeg"):
            filename = output_path
        else:
            os.makedirs(output_path, exist_ok=True)
            filename = os.path.join(output_path, f"{safe_text}.png")
    else:
        os.makedirs(config["generate"]["output_dir"], exist_ok=True)
        filename = os.path.join(config["generate"]["output_dir"], f"{safe_text}.png")

    save_image(result_tensor, filename)


if __name__ == "__main__":
    main()
