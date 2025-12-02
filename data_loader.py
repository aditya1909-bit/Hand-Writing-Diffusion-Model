import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from PIL import Image
import os
import random

class HandwritingDataset(Dataset):
    def __init__(self, root_dir, tokenizer, image_size=(64, 256), mock_mode=False, num_writers=None):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.mock_mode = mock_mode
        self.num_writers = num_writers
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])
        ])

        self.samples = self._build_index()
        print(f"Dataset loaded: {len(self.samples)} samples found.")
        if num_writers:
            print(f"--- FILTER ACTIVE: Training on the first {num_writers} writers ---")

        # Precompute mapping from writer_id to list of indices for fast style sampling
        self.writer_to_indices = {}
        for idx, s in enumerate(self.samples):
            wid = s["writer_id"]
            if wid not in self.writer_to_indices:
                self.writer_to_indices[wid] = []
            self.writer_to_indices[wid].append(idx)

    def _build_index(self):
        if self.mock_mode:
            return [{"text": "mock", "writer_id": "001", "path": "mock.png"}] * 100

        samples = []
        words_file = os.path.join(self.root_dir, "words.txt")
        
        if not os.path.exists(words_file):
            print(f"Warning: {words_file} not found.")
            return []

        with open(words_file, "r") as f:
            lines = f.readlines()

        # 1. First pass: Identify the top N writers
        allowed_writers = set()
        if self.num_writers:
            seen_writers = []
            for line in lines:
                if line.startswith("#") or line.strip() == "": continue
                parts = line.strip().split()
                if len(parts) < 9: continue
                
                # FIX: Extract writer ID from the file ID (first column)
                file_id = parts[0]
                w_id = file_id.split("-")[0] 
                
                if w_id not in seen_writers:
                    seen_writers.append(w_id)
                    if len(seen_writers) >= self.num_writers:
                        break
            allowed_writers = set(seen_writers)
            print(f"Selected writers: {sorted(list(allowed_writers))}")

        # 2. Second pass: Collect samples
        for line in lines:
            if line.startswith("#") or line.strip() == "": continue
            
            parts = line.strip().split()
            if len(parts) < 9: continue

            file_id = parts[0]

            writer_id = file_id.split("-")[0]
            
            transcription = parts[-1] 
            
            if self.num_writers and writer_id not in allowed_writers:
                continue
            
            folder_1 = file_id.split("-")[0]
            folder_2 = f"{folder_1}-{file_id.split('-')[1]}"
            img_path = os.path.join(self.root_dir, "words", folder_1, folder_2, f"{file_id}.png")
            
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
                target_image = Image.open(item["path"]).convert("RGB")
                target_image = self.transform(target_image)

                # Use precomputed indices for this writer to avoid O(N) scans each time
                writer_id = item["writer_id"]
                candidate_indices = self.writer_to_indices.get(writer_id, [idx])
                if len(candidate_indices) > 1:
                    style_idx = random.choice(candidate_indices)
                    style_item = self.samples[style_idx]
                else:
                    style_item = item
                
                style_image = Image.open(style_item["path"]).convert("RGB")
                style_image = self.transform(style_image)
                
            except Exception as e:
                print(f"Error loading {item['path']}: {e}")
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

    NUM_WRITERS = 10

    dataset = HandwritingDataset(
        root_dir="./iam_data",
        tokenizer=tokenizer,
        mock_mode=mock_mode,
        num_writers=NUM_WRITERS,
    )

    # Use more workers when not in mock mode; fall back to 0 on platforms that don't like multiprocessing
    num_workers = 0 if mock_mode else max(2, (os.cpu_count() or 4) // 2)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )