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
            root_dir: Path to the IAM dataset (or where it will be).
            tokenizer: BERT tokenizer for text.
            image_size: Tuple (height, width).
            mock_mode: If True, generates random noise data for testing code flow.
        """
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.mock_mode = mock_mode
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # Normalize to [-1, 1] for Diffusion
        ])

        # In a real scenario, parse the IAM 'lines.txt' file here 
        # to build a dictionary of {writer_id: [list_of_image_paths]}
        self.samples = self._build_index()

    def _build_index(self):
        if self.mock_mode:
            # Create fake data for testing
            return [
                {"text": "Hello world", "writer_id": "001", "path": "mock1.png"},
                {"text": "Diffusion test", "writer_id": "002", "path": "mock2.png"},
                {"text": "Style transfer", "writer_id": "001", "path": "mock3.png"},
            ] * 100 # Repeat to simulate a larger dataset
        
        # TODO: Implement actual IAM parsing logic here
        return []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 1. Get Text Embeddings
        text_tokens = self.tokenizer(
            item["text"], 
            padding="max_length", 
            max_length=32, 
            truncation=True, 
            return_tensors="pt"
        )

        if self.mock_mode:
            # Generate random noise tensors to simulate images
            target_image = torch.randn(3, self.image_size[0], self.image_size[1])
            style_image = torch.randn(3, self.image_size[0], self.image_size[1])
        else:
            # Load actual images
            # Logic: Load the target image, and pick a RANDOM image from same writer_id for style
            img_path = os.path.join(self.root_dir, item["path"])
            target_image = Image.open(img_path).convert("RGB")
            target_image = self.transform(target_image)
            
            # Placeholder
            style_image = target_image.clone() 

        return {
            "pixel_values": target_image,      # The image we want to generate
            "style_pixel_values": style_image, # The reference style
            "input_ids": text_tokens.input_ids.squeeze(0),
            "attention_mask": text_tokens.attention_mask.squeeze(0)
        }

def get_dataloader(batch_size=8, mock_mode=True):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = HandwritingDataset(root_dir="./iam", tokenizer=tokenizer, mock_mode=mock_mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)