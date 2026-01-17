import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizerFast
from PIL import Image

from image_utils import resize_with_pad


def _parse_words_line(line, root_dir, words_dir, skip_err):
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = line.split()
    if len(parts) < 9:
        return None

    file_id = parts[0]
    status = parts[1]
    if skip_err and status != "ok":
        return None

    file_parts = file_id.split("-")
    if len(file_parts) < 2:
        return None

    transcription = " ".join(parts[8:]).replace("|", " ").strip()
    if not transcription:
        return None

    writer_id = file_parts[0]
    folder_1 = writer_id
    folder_2 = f"{folder_1}-{file_parts[1]}"
    img_path = os.path.join(root_dir, words_dir, folder_1, folder_2, f"{file_id}.png")

    return {
        "text": transcription,
        "writer_id": writer_id,
        "path": img_path,
    }


class HandwritingDataset(Dataset):
    def __init__(
        self,
        root_dir,
        image_size=(64, 256),
        max_length=32,
        mock_mode=False,
        num_writers=None,
        words_file="words.txt",
        words_dir="words",
        pad_value=255,
        skip_err=True,
        samples=None,
        writer_id_to_index=None,
    ):
        self.root_dir = root_dir
        self.image_size = tuple(image_size)
        self.max_length = max_length
        self.mock_mode = mock_mode
        self.num_writers = num_writers
        self.words_file = words_file
        self.words_dir = words_dir
        self.pad_value = pad_value
        self.skip_err = skip_err

        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        if samples is None:
            self.samples = self._build_index()
        else:
            self.samples = samples
        print(f"Dataset loaded: {len(self.samples)} samples found.")
        if num_writers:
            print(f"--- FILTER ACTIVE: Training on the first {num_writers} writers ---")

        self.writer_to_indices = {}
        for idx, sample in enumerate(self.samples):
            writer_id = sample["writer_id"]
            self.writer_to_indices.setdefault(writer_id, []).append(idx)

        if writer_id_to_index is None:
            writer_ids = sorted({sample["writer_id"] for sample in self.samples})
            writer_id_to_index = {writer_id: idx for idx, writer_id in enumerate(writer_ids)}
        self.writer_id_to_index = writer_id_to_index
        self.writer_count = len(self.writer_id_to_index)

    def _build_index(self):
        if self.mock_mode:
            return [{"text": "mock text", "writer_id": "000", "path": None}] * 100

        words_path = os.path.join(self.root_dir, self.words_file)
        if not os.path.exists(words_path):
            print(f"Warning: {words_path} not found.")
            return []

        with open(words_path, "r", encoding="utf-8", errors="ignore") as handle:
            lines = handle.readlines()

        allowed_writers = None
        if self.num_writers:
            allowed_writers = []
            for line in lines:
                parsed = _parse_words_line(line, self.root_dir, self.words_dir, self.skip_err)
                if not parsed:
                    continue
                writer_id = parsed["writer_id"]
                if writer_id not in allowed_writers:
                    allowed_writers.append(writer_id)
                if len(allowed_writers) >= self.num_writers:
                    break
            allowed_writers = set(allowed_writers)
            print(f"Selected writers: {sorted(list(allowed_writers))}")

        samples = []
        for line in lines:
            parsed = _parse_words_line(line, self.root_dir, self.words_dir, self.skip_err)
            if not parsed:
                continue
            if allowed_writers and parsed["writer_id"] not in allowed_writers:
                continue
            if not os.path.exists(parsed["path"]):
                continue
            samples.append(parsed)

        return samples

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        image = Image.open(path).convert("RGB")
        image = resize_with_pad(image, self.image_size, fill=self.pad_value)
        return self.to_tensor(image)

    def __getitem__(self, idx):
        item = self.samples[idx]
        writer_label = self.writer_id_to_index.get(item["writer_id"], -1)

        if self.mock_mode or item["path"] is None:
            height, width = self.image_size
            target_image = torch.randn(3, height, width)
            style_image = torch.randn(3, height, width)
            return {
                "pixel_values": target_image,
                "style_pixel_values": style_image,
                "text": item["text"],
                "writer_id": item["writer_id"],
                "writer_label": writer_label,
            }

        try:
            target_image = self._load_image(item["path"])

            writer_id = item["writer_id"]
            candidate_indices = self.writer_to_indices.get(writer_id, [idx])
            if len(candidate_indices) > 1:
                style_candidates = [i for i in candidate_indices if i != idx]
                style_idx = random.choice(style_candidates)
                style_item = self.samples[style_idx]
            else:
                style_item = item

            style_image = self._load_image(style_item["path"])
        except Exception as exc:
            print(f"Error loading {item['path']}: {exc}")
            height, width = self.image_size
            target_image = torch.randn(3, height, width)
            style_image = torch.randn(3, height, width)

        return {
            "pixel_values": target_image,
            "style_pixel_values": style_image,
            "text": item["text"],
            "writer_id": item["writer_id"],
            "writer_label": writer_label,
        }


class HandwritingCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts = [item["text"] for item in batch]
        tokenized = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        style_pixel_values = torch.stack([item["style_pixel_values"] for item in batch])
        writer_labels = torch.tensor([item["writer_label"] for item in batch], dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "style_pixel_values": style_pixel_values,
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
            "text": texts,
            "writer_labels": writer_labels,
        }


def _split_samples(samples, val_split, seed, split_by_writer):
    if val_split <= 0:
        return samples, []

    rng = random.Random(seed)

    if split_by_writer:
        writer_to_samples = {}
        for sample in samples:
            writer_to_samples.setdefault(sample["writer_id"], []).append(sample)
        writer_ids = list(writer_to_samples.keys())
        rng.shuffle(writer_ids)
        val_writer_count = max(1, int(len(writer_ids) * val_split))
        val_writers = set(writer_ids[:val_writer_count])

        train_samples = []
        val_samples = []
        for writer_id, writer_samples in writer_to_samples.items():
            if writer_id in val_writers:
                val_samples.extend(writer_samples)
            else:
                train_samples.extend(writer_samples)
    else:
        indices = list(range(len(samples)))
        rng.shuffle(indices)
        split_idx = max(1, int(len(indices) * (1.0 - val_split)))
        split_idx = min(split_idx, len(indices) - 1)
        train_samples = [samples[i] for i in indices[:split_idx]]
        val_samples = [samples[i] for i in indices[split_idx:]]

    return train_samples, val_samples


def get_dataloaders(config):
    data_cfg = config["data"]
    train_cfg = config.get("train", {})
    model_cfg = config["model"]

    tokenizer = BertTokenizerFast.from_pretrained(model_cfg["text_encoder"])

    base_dataset = HandwritingDataset(
        root_dir=data_cfg["root_dir"],
        image_size=data_cfg["image_size"],
        max_length=data_cfg["max_length"],
        mock_mode=data_cfg["mock_mode"],
        num_writers=data_cfg["num_writers"],
        words_file=data_cfg["words_file"],
        words_dir=data_cfg["words_dir"],
        pad_value=data_cfg["pad_value"],
        skip_err=data_cfg["skip_err"],
    )

    writer_id_to_index = base_dataset.writer_id_to_index
    all_samples = base_dataset.samples

    train_samples, val_samples = _split_samples(
        all_samples,
        val_split=float(train_cfg.get("val_split", 0.0)),
        seed=data_cfg.get("seed", 1337),
        split_by_writer=bool(train_cfg.get("val_split_by_writer", False)),
    )

    train_dataset = HandwritingDataset(
        root_dir=data_cfg["root_dir"],
        image_size=data_cfg["image_size"],
        max_length=data_cfg["max_length"],
        mock_mode=data_cfg["mock_mode"],
        num_writers=None,
        words_file=data_cfg["words_file"],
        words_dir=data_cfg["words_dir"],
        pad_value=data_cfg["pad_value"],
        skip_err=data_cfg["skip_err"],
        samples=train_samples,
        writer_id_to_index=writer_id_to_index,
    )

    val_dataset = None
    if val_samples:
        val_dataset = HandwritingDataset(
            root_dir=data_cfg["root_dir"],
            image_size=data_cfg["image_size"],
            max_length=data_cfg["max_length"],
            mock_mode=data_cfg["mock_mode"],
            num_writers=None,
            words_file=data_cfg["words_file"],
            words_dir=data_cfg["words_dir"],
            pad_value=data_cfg["pad_value"],
            skip_err=data_cfg["skip_err"],
            samples=val_samples,
            writer_id_to_index=writer_id_to_index,
        )

    collator = HandwritingCollator(tokenizer, data_cfg["max_length"])

    num_workers = train_cfg.get("num_workers", 0)
    if data_cfg["mock_mode"]:
        num_workers = 0
    prefetch_factor = train_cfg.get("prefetch_factor", 2)

    loader_kwargs = {
        "batch_size": train_cfg.get("batch_size", 8),
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "collate_fn": collator,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = None
    if val_dataset:
        val_batch_size = train_cfg.get("val_batch_size", loader_kwargs["batch_size"])
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            collate_fn=collator,
        )

    return train_loader, val_loader, train_dataset.writer_count


def get_dataloader(config, batch_size=None, shuffle=True):
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config.get("train", {})

    tokenizer = BertTokenizerFast.from_pretrained(model_cfg["text_encoder"])

    dataset = HandwritingDataset(
        root_dir=data_cfg["root_dir"],
        image_size=data_cfg["image_size"],
        max_length=data_cfg["max_length"],
        mock_mode=data_cfg["mock_mode"],
        num_writers=data_cfg["num_writers"],
        words_file=data_cfg["words_file"],
        words_dir=data_cfg["words_dir"],
        pad_value=data_cfg["pad_value"],
        skip_err=data_cfg["skip_err"],
    )

    collator = HandwritingCollator(tokenizer, data_cfg["max_length"])

    num_workers = train_cfg.get("num_workers", 0)
    if data_cfg["mock_mode"]:
        num_workers = 0
    prefetch_factor = train_cfg.get("prefetch_factor", 2)

    loader_kwargs = {
        "batch_size": batch_size or train_cfg.get("batch_size", 8),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "collate_fn": collator,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(
        dataset,
        **loader_kwargs,
    )
