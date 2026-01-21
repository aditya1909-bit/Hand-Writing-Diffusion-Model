import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from config_utils import load_config
from data_loader import HandwritingDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute IAM caches for faster training.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--cache-dir", help="Cache directory override.")
    parser.add_argument("--cache-images", action="store_true", help="Cache resized image tensors.")
    parser.add_argument("--cache-tokens", action="store_true", help="Cache tokenized text.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for preprocessing.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes.")
    parser.add_argument("--max-items", type=int, help="Optional cap on number of items.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    cache_dir = args.cache_dir or config["data"].get("cache_dir") or "./cache"
    os.makedirs(cache_dir, exist_ok=True)

    cache_images = args.cache_images or config["data"].get("cache_images", False)
    cache_tokens = args.cache_tokens or config["data"].get("cache_tokens", False)

    images_dir = os.path.join(cache_dir, "images")
    if cache_images:
        os.makedirs(images_dir, exist_ok=True)

    tokenizer = BertTokenizerFast.from_pretrained(config["model"]["text_encoder"])

    dataset = HandwritingDataset(
        root_dir=config["data"]["root_dir"],
        image_size=config["data"]["image_size"],
        max_length=config["data"]["max_length"],
        mock_mode=config["data"]["mock_mode"],
        num_writers=config["data"]["num_writers"],
        words_file=config["data"]["words_file"],
        words_dir=config["data"]["words_dir"],
        pad_value=config["data"]["pad_value"],
        skip_err=config["data"]["skip_err"],
        cache_dir=None,
        cache_tokens=False,
        cache_images=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=False,
        collate_fn=lambda batch: batch,
    )

    tokens = {} if cache_tokens else None
    total = 0
    start_time = time.time()

    for batch in loader:
        if args.max_items is not None and total >= args.max_items:
            break

        file_ids = [item.get("file_id") for item in batch]
        texts = [item.get("text") for item in batch]

        if cache_tokens:
            tokenized = tokenizer(
                texts,
                padding="max_length",
                max_length=config["data"]["max_length"],
                truncation=True,
                return_tensors="pt",
            )

        for idx, item in enumerate(batch):
            file_id = file_ids[idx]
            if not file_id:
                continue

            if cache_images:
                image_path = os.path.join(images_dir, f"{file_id}.pt")
                if args.overwrite or not os.path.exists(image_path):
                    torch.save(item["pixel_values"].cpu(), image_path)

            if cache_tokens:
                if args.overwrite or file_id not in tokens:
                    tokens[file_id] = {
                        "input_ids": tokenized.input_ids[idx].cpu(),
                        "attention_mask": tokenized.attention_mask[idx].cpu(),
                    }

            total += 1
            if args.max_items is not None and total >= args.max_items:
                break

    if cache_tokens:
        tokens_path = os.path.join(cache_dir, "tokens.pt")
        if args.overwrite or not os.path.exists(tokens_path):
            torch.save(tokens, tokens_path)

    elapsed = time.time() - start_time
    print(f"Preprocessing complete. Items: {total}, elapsed: {elapsed:.2f}s")
    if cache_images:
        print(f"Image cache: {images_dir}")
    if cache_tokens:
        print(f"Token cache: {os.path.join(cache_dir, 'tokens.pt')}")


if __name__ == "__main__":
    main()
