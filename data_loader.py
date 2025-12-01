import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from PIL import Image
import os
import random

class HandwritingDataset(Dataset):
    def __init__(self, root_dir, tokenizer, image_size=(64, 256), mock_mode=False):
        """
        Args:
            root_dir: Path to the unzipped IAM dataset (e.g., "./iam_data")
            tokenizer: BERT tokenizer.
            image_size: Target (H, W).
            mock_mode: Set to True to test without data.
        """
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.mock_mode = mock_mode
        
        # Transforms: Resize and Normalize to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ])

        self.samples = self._build_index()
        print(f"Dataset loaded: {len(self.samples)} samples found.")

    def _build_index(self):
        if self.mock_mode:
            return [
                {"text": "mock text", "writer_id": "001", "path": "mock.png"}
            ] * 100

        samples = []
        
        words_file = os.path.join(self.root_dir, "words.txt")
        
        if not os.path.exists(words_file):
            print(f"Warning: {words_file} not found. Check your extracted folder structure.")
            return []

        with open(words_file, "r") as f:
            lines = f.readlines()

        # Parse IAM format
        for line in lines:
            if line.startswith("#") or line.strip() == "":
                continue
            
            parts = line.strip().split()
            if len(parts) < 9:
                continue

            # IAM parsing logic
            file_id = parts[0]
            writer_id = parts[1]
            transcription = parts[-1] 
            
            # Construct image path based on ID structure
            folder_1 = file_id.split("-")[0]
            folder_2 = f"{folder_1}-{file_id.split('-')[1]}"
            
            img_path = os.path.join(self.root_dir, "words", folder_1, folder_2, f"{file_id}.png")
            
            # Only add if file actually exists
            if os.path.exists(img_path):
                samples.append({
                    "text": transcription,
                    "writer_id": writer_id,
                    "path": img_path
                })
        
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 1. Text Tokenization
        text_tokens = self.tokenizer(
            item["text"], 
            padding="max_length", 
            max_length=32, 
            truncation=True, 
            return_tensors="pt"
        )

        if self.mock_mode:
            target_image = torch.randn(3, self.image_size[0], self.image_size[1])
            style_image = torch.randn(3, self.image_size[0], self.image_size[1])
        else:
            try:
                # Load Target Image
                target_image = Image.open(item["path"]).convert("RGB")
                target_image = self.transform(target_image)
                
                # Load Style Image (Random image from SAME writer)
                same_writer_samples = [s for s in self.samples if s["writer_id"] == item["writer_id"]]
                if len(same_writer_samples) > 1:
                    style_item = random.choice(same_writer_samples)
                else:
                    style_item = item # Fallback if writer has only 1 image
                
                style_image = Image.open(style_item["path"]).convert("RGB")
                style_image = self.transform(style_image)
                
            except Exception as e:
                print(f"Error loading {item['path']}: {e}")
                # Fallback to noise in case of corrupt image
                target_image = torch.randn(3, self.image_size[0], self.image_size[1])
                style_image = torch.randn(3, self.image_size[0], self.image_size[1])

        return {
            "pixel_values": target_image,
            "style_pixel_values": style_image,
            "input_ids": text_tokens.input_ids.squeeze(0),
            "attention_mask": text_tokens.attention_mask.squeeze(0)
        }

def get_dataloader(batch_size=8, mock_mode=False):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = HandwritingDataset(root_dir="./iam_data", tokenizer=tokenizer, mock_mode=mock_mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)