import torch
import os
import glob
import re
from tqdm import tqdm
from data_loader import get_dataloader
from model import HandwritingDiffusionSystem
from torch.cuda.amp import autocast, GradScaler

SAVE_DIR = "./saved_models"

def get_device():
    if torch.cuda.is_available(): return "cuda"
    elif torch.backends.mps.is_available(): return "mps"
    else: return "cpu"

CONFIG = {
    "epochs": 250,
    "batch_size": 32,
    "lr": 1e-4,
    "save_dir": SAVE_DIR,
    "device": get_device()
}

def find_latest_checkpoint(save_dir):
    "Scans the save_dir for the checkpoint with the highest epoch number."
    if not os.path.exists(save_dir):
        return None, -1
        
    checkpoints = glob.glob(os.path.join(save_dir, "model_epoch_*.pth"))
    if not checkpoints:
        return None, -1

    latest_epoch = -1
    latest_path = None
    
    for path in checkpoints:
        match = re.search(r"model_epoch_(\d+).pth", path)
        if match:
            epoch_num = int(match.group(1))
            if epoch_num > latest_epoch:
                latest_epoch = epoch_num
                latest_path = path
                
    return latest_path, latest_epoch

def train():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    print(f"--- Training Configuration ---")
    print(f"Device: {CONFIG['device'].upper()}")
    
    # 1. Initialize Model
    model_system = HandwritingDiffusionSystem(device=CONFIG["device"]).to(CONFIG["device"])

    latest_path, last_epoch = find_latest_checkpoint(CONFIG["save_dir"])
    start_epoch = 0

    if latest_path:
        print(f"FOUND CHECKPOINT: {latest_path}")
        print("Loading weights...")
        state_dict = torch.load(latest_path, map_location=CONFIG["device"])
        
        # Sanitize keys (remove 'module.' prefix if it exists)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        model_system.load_state_dict(new_state_dict)
        start_epoch = last_epoch + 1
        print(f"Resuming training from Epoch {start_epoch}")
    else:
        print("No checkpoints found. Starting from scratch.")

    # Multi-GPU Support (DataParallel)
    if CONFIG["device"] == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model_system = torch.nn.DataParallel(model_system)
    
    # Load Data
    dataloader = get_dataloader(batch_size=CONFIG["batch_size"], mock_mode=False)
    
    # 2. Optimizer & Scaler
    base_model = model_system.module if isinstance(model_system, torch.nn.DataParallel) else model_system

    optimizer = torch.optim.AdamW(
        list(base_model.unet.parameters()) + list(base_model.style_encoder.parameters()),
        lr=CONFIG["lr"]
    )
    
    amp_enabled = (CONFIG["device"] == "cuda" and torch.cuda.is_available())
    scaler = GradScaler() if amp_enabled else None
    
    # 3. Training Loop
    print("Starting training loop...")
    for epoch in range(start_epoch, CONFIG["epochs"]):
        model_system.train()
        epoch_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{CONFIG['epochs']}", leave=True)
        for step, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            
            # Forward Pass
            if amp_enabled:
                with autocast():
                    raw_loss = model_system(batch)
                    loss = raw_loss.mean() 
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                raw_loss = model_system(batch)
                loss = raw_loss.mean() 
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"--- Epoch {epoch} Finished. Avg Loss: {avg_loss:.4f} ---")
        
        # Save Checkpoint (Only every 5th epoch)
        if (epoch + 1) % 5 == 0:
            save_path = f"{CONFIG['save_dir']}/model_epoch_{epoch}.pth"
            state_to_save = base_model.state_dict()
            torch.save(state_to_save, save_path)
            print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()