import os

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import torch
import sys
import random
from transformers import BertTokenizer
from model import HandwritingDiffusionSystem
from torchvision import transforms
from PIL import Image

# CONFIGURATION
MODEL_PATH = "./saved_models/model_epoch_9.pth"
OUTPUT_FOLDER = "./generated_outputs"
DATA_ROOT = "./iam_data/words"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def find_image_in_path(user_input):
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

def load_model(device):
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        sys.exit(1)

    system = HandwritingDiffusionSystem(device=device)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        system.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)

    system.to(device)
    
    # SCHEDULER DEVICE FIX
    for key, value in system.scheduler.__dict__.items():
        if isinstance(value, torch.Tensor):
            setattr(system.scheduler, key, value.to(device))
            
    system.eval()
    print("Model loaded successfully.")
    return system

def generate_handwriting(system, text, style_path, device, num_steps=50):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    try:
        style_img = Image.open(style_path).convert("RGB")
        style_tensor = transform(style_img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    text_inputs = tokenizer(text, padding="max_length", max_length=32, truncation=True, return_tensors="pt")
    input_ids = text_inputs.input_ids.to(device)
    mask = text_inputs.attention_mask.to(device)
    
    with torch.no_grad():
        text_emb = system.text_encoder(input_ids, attention_mask=mask)[0]
        style_emb = system.style_encoder(style_tensor)
        
    latents = torch.randn((1, 3, 64, 256), device=device)
    
    # 1. Set timesteps
    system.scheduler.set_timesteps(num_steps)
    
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
            latents = system.scheduler.step(noise_pred, t, latents).prev_sample
            
    return latents

def save_image(tensor, filename):
    image = tensor.squeeze(0).cpu().detach()
    image = (image / 2 + 0.5).clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(filename)
    print(f"Saved output to: {filename}")

def main():
    # Force the allocator to restart
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    device = get_device()
    print(f"Running on device: {device.upper()}")
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    model_system = load_model(device)

    print("\n--- Interactive Handwriting Generator ---")
    print("Press Ctrl+C to quit.\n")

    while True:
        try:
            text_input = input("Enter text to write: ").strip()
            if not text_input: continue
            
            style_input_raw = input("Enter path OR folder OR ID (e.g. a01): ").strip()
            style_input = style_input_raw.replace("'", "").replace('"', "").strip()
            
            if not style_input: continue

            real_style_path = find_image_in_path(style_input)
            
            if not real_style_path:
                print(f"Could not find any images inside '{style_input}'")
                continue

            result_tensor = generate_handwriting(model_system, text_input, real_style_path, device)
            
            if result_tensor is not None:
                safe_text = "".join([c for c in text_input if c.isalnum() or c in (' ', '_')]).rstrip()
                filename = f"{OUTPUT_FOLDER}/{safe_text.replace(' ', '_')}.png"
                save_image(result_tensor, filename)
                
                if sys.platform == "darwin":
                    os.system(f"open {filename}")
                
            print("-" * 30)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()