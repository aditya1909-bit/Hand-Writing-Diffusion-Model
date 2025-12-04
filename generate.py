import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import torch
import sys
import re
from transformers import BertTokenizer
from model import HandwritingDiffusionSystem
from torchvision import transforms
from PIL import Image

# CONFIGURATION
SAVE_DIR = "./saved_models"
OUTPUT_FOLDER = "./generated_outputs"
DATA_ROOT = "./iam_data/words"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def find_image_in_path(user_input: str):
    "Resolve a user-specified style path / folder / ID to a concrete image file."
    if os.path.isfile(user_input):
        return user_input

    search_path = user_input
    if not os.path.exists(search_path):
        potential_path = os.path.join(DATA_ROOT, user_input)
        if os.path.exists(potential_path):
            search_path = potential_path

    if os.path.isdir(search_path):
        print(f"Searching inside folder: {search_path}...")
        for root, dirs, files in os.walk(search_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    found_path = os.path.join(root, file)
                    print(f"-> Auto-selected style image: {file}")
                    return found_path

    return None


def load_model(device: str, model_path: str):
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    system = HandwritingDiffusionSystem(device=device)

    try:
        state_dict = torch.load(model_path, map_location=device)

        # Try to load as-is first
        try:
            system.load_state_dict(state_dict)
        except RuntimeError as e:
            # Common case: checkpoint from DataParallel (keys prefixed with "module.")
            if any(k.startswith("module.") for k in state_dict.keys()):
                print("Detected DataParallel checkpoint, stripping 'module.' prefixes...")
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k[len("module."):] if k.startswith("module.") else k
                    new_state_dict[new_key] = v
                system.load_state_dict(new_state_dict)
            else:
                raise e

    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)

    system.to(device)

    # Ensure scheduler tensors are on the right device
    if hasattr(system, "scheduler"):
        for key, value in system.scheduler.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(system.scheduler, key, value.to(device))

    system.eval()
    print("Model loaded successfully.")
    return system


def get_all_model_paths():
    """Return a sorted list of all checkpoint paths in SAVE_DIR."""
    if not os.path.isdir(SAVE_DIR):
        print(f"No saved_models directory found at {SAVE_DIR}")
        return []

    paths = []
    for name in os.listdir(SAVE_DIR):
        if name.endswith(".pth"):
            paths.append(os.path.join(SAVE_DIR, name))

    # Sort by epoch number if present in filename, otherwise lexicographically
    def extract_epoch(p):
        m = re.search(r"epoch_(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else float("inf")

    paths.sort(key=extract_epoch)
    return paths


def generate_handwriting(system, text: str, style_path: str, device: str, num_steps: int = 50):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    try:
        style_img = Image.open(style_path).convert("RGB")
        style_tensor = transform(style_img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error opening image '{style_path}': {e}")
        return None

    # Text encoding
    text_inputs = tokenizer(
        text,
        padding="max_length",
        max_length=32,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = text_inputs.input_ids.to(device)
    mask = text_inputs.attention_mask.to(device)

    with torch.no_grad():
        text_emb = system.text_encoder(input_ids, attention_mask=mask)[0]
        style_emb = system.style_encoder(style_tensor)

    # Latent initialization (3 channels, matching UNet in_channels)
    latents = torch.randn((1, 3, 64, 256), device=device)

    # 1. Set timesteps
    system.scheduler.set_timesteps(num_steps)

    # 2. Ensure timesteps live on the same device as latents
    if hasattr(system.scheduler, "timesteps"):
        system.scheduler.timesteps = system.scheduler.timesteps.to(device)

    print(f"Generating '{text}'...")
    for t in system.scheduler.timesteps:
        with torch.no_grad():
            noise_pred = system.unet(
                latents,
                t,
                encoder_hidden_states=text_emb,
                class_labels=style_emb
            ).sample

            latents = system.scheduler.step(
                noise_pred,
                t,
                latents
            ).prev_sample

    return latents


def save_image(tensor: torch.Tensor, filename: str):
    image = tensor.squeeze(0).cpu().detach()
    # Map from [-1, 1] back to [0, 1]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(filename)
    print(f"Saved output to: {filename}")


def main():
    # Force the allocator to restart on MPS
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    device = get_device()
    print(f"Running on device: {device.upper()}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Collect all checkpoints once
    model_paths = get_all_model_paths()
    if not model_paths:
        print(f"No model checkpoints found in {SAVE_DIR}")
        sys.exit(1)
    print(f"Found {len(model_paths)} model checkpoints.")

    print("\n--- Interactive Handwriting Generator ---")
    print("Press Ctrl+C to quit.\n")

    while True:
        try:
            text_input = input("Enter text to write: ").strip()
            if not text_input:
                continue

            style_input_raw = input("Enter path OR folder OR ID (e.g. a01): ").strip()
            style_input = style_input_raw.replace("'", "").replace('"', "").strip()
            if not style_input:
                continue

            real_style_path = find_image_in_path(style_input)
            if not real_style_path:
                print(f"Could not find any images inside '{style_input}'")
                continue

            # Prepare folder and safe text name
            safe_text = "".join([c for c in text_input if c.isalnum() or c in (' ', '_')]).rstrip()
            safe_text = safe_text.replace(" ", "_")
            text_folder = os.path.join(OUTPUT_FOLDER, safe_text)
            os.makedirs(text_folder, exist_ok=True)

            last_filename = None

            # For each checkpoint, load the model, generate, and save
            for model_path in model_paths:
                # Extract epoch number from filename, default to 'unknown' if not found
                m_epoch = re.search(r"epoch_(\d+)", os.path.basename(model_path))
                epoch_str = m_epoch.group(1) if m_epoch else "unknown"

                print(f"\n=== Using checkpoint epoch {epoch_str} ===")
                model_system = load_model(device, model_path)
                result_tensor = generate_handwriting(model_system, text_input, real_style_path, device)

                if result_tensor is not None:
                    filename = os.path.join(text_folder, f"{safe_text}_{epoch_str}.png")
                    save_image(result_tensor, filename)
                    last_filename = filename

            # Auto-open the last generated image on macOS
            if sys.platform == "darwin" and last_filename is not None:
                os.system(f"open '{last_filename}'")

            print("-" * 30)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()