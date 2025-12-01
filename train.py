import torch
import os
from tqdm import tqdm
from data_loader import get_dataloader
from model import HandwritingDiffusionSystem

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

# Configuration
CONFIG = {
    "epochs": 5, 
    "batch_size": 12,
    "lr": 1e-4,
    "save_dir": "./saved_models",
    "device": get_device()
}

def train():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    print(f"--- Training Configuration ---")
    print(f"Device: {CONFIG['device'].upper()}")
    if CONFIG['device'] == 'mps':
        print("Apple Metal acceleration enabled.")
    
    # 1. Initialize
    model_system = HandwritingDiffusionSystem(device=CONFIG["device"]).to(CONFIG["device"])
    
    dataloader = get_dataloader(batch_size=CONFIG["batch_size"], mock_mode=False)
    
    if len(dataloader) == 0:
        print("No samples found in dataset. Check your IAM root_dir and words.txt path in Data_Loader.py.")
        return
    
    # 2. Optimizer
    optimizer = torch.optim.AdamW(
        list(model_system.unet.parameters()) + list(model_system.style_encoder.parameters()),
        lr=CONFIG["lr"]
    )
    
    # 3. Loop
    for epoch in range(CONFIG["epochs"]):
        model_system.train()
        epoch_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", leave=True)
        for step, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            
            loss = model_system(batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            # update tqdm bar with current loss
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"--- Epoch {epoch} Finished. Avg Loss: {avg_loss:.4f} ---")
        
        # Save Checkpoint
        save_path = f"{CONFIG['save_dir']}/model_epoch_{epoch}.pth"
        torch.save(model_system.state_dict(), save_path)
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()