import torch
from data_loader import HandwritingDataset
from transformers import BertTokenizer
import matplotlib.pyplot as plt
from torchvision import transforms

def inspect():
    print("Loading dataset (this may take a moment)...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    dataset = HandwritingDataset(root_dir="./iam_data", tokenizer=tokenizer, mock_mode=False)
    
    print(f"Found {len(dataset)} samples.")
    
    # 1. Grab 5 random samples
    indices = torch.randperm(len(dataset))[:5]
    
    fig, axes = plt.subplots(5, 1, figsize=(10, 8))
    
    for i, idx in enumerate(indices):
        idx = idx.item()
        item = dataset[idx]
        
        # Un-normalize from [-1, 1] back to [0, 1] for viewing
        img_tensor = item["pixel_values"]
        img_display = (img_tensor / 2 + 0.5).clamp(0, 1)
        img_display = img_display.permute(1, 2, 0).numpy()
        
        # Decode the text (BERT tokens -> String)
        text_ids = item["input_ids"]
        decoded_text = tokenizer.decode(text_ids, skip_special_tokens=True)
        
        ax = axes[i]
        ax.imshow(img_display)
        ax.set_title(f"Label: '{decoded_text}'")
        ax.axis("off")
        
    plt.tight_layout()
    plt.savefig("data_inspection.png")
    print("Saved inspection to 'data_inspection.png'. Open it to check quality!")

if __name__ == "__main__":
    inspect()