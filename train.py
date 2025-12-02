import torch
import os
from tqdm import tqdm
from data_loader import get_dataloader
from model import HandwritingDiffusionSystem
from torch.amp import autocast, GradScaler

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

def train():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    print(f"--- Training Configuration ---")
    print(f"Device: {CONFIG['device'].upper()}")
    print(f"Saving to: {CONFIG['save_dir']}")
    amp_enabled = (CONFIG["device"] == "cuda" and torch.cuda.is_available())
    print(f"Mixed Precision (FP16): {'ENABLED' if amp_enabled else 'DISABLED'}")
    
    # 1. Initialize
    model_system = HandwritingDiffusionSystem(device=CONFIG["device"]).to(CONFIG["device"])

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
    
    # The Scaler handles the 16-bit math stability when using CUDA
    scaler = GradScaler(enabled=amp_enabled)
    
    # 3. Loop
    for epoch in range(CONFIG["epochs"]):
        model_system.train()
        epoch_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", leave=True)
        for step, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            
            if amp_enabled:
                # Run forward in mixed precision on CUDA
                with autocast(device_type="cuda"):
                    loss = model_system(batch)
                if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                    loss = loss.mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Fallback: standard 32-bit training
                loss = model_system(batch)
                if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                    loss = loss.mean()
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"--- Epoch {epoch} Finished. Avg Loss: {avg_loss:.4f} ---")
        
        # Save only every 5 epochs to reduce file count
        if (epoch + 1) % 5 == 0:
            save_path = f"{CONFIG['save_dir']}/model_epoch_{epoch+1}.pth"
            torch.save(model_system.state_dict(), save_path)
            print(f"Saved checkpoint: {save_path}")

if __name__ == "__main__":
    train()