import argparse

import torch
import matplotlib.pyplot as plt

from config_utils import load_config
from data_loader import HandwritingDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect a few dataset samples.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to visualize.")
    parser.add_argument("--output", default="data_inspection.png", help="Output image path.")
    parser.add_argument("--mock", action="store_true", help="Enable mock mode for quick tests.")
    return parser.parse_args()


def inspect():
    args = parse_args()
    config = load_config(args.config)
    if args.mock:
        config["data"]["mock_mode"] = True

    print("Loading dataset (this may take a moment)...")
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
    )

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Check your data path and filters.")

    count = min(args.num_samples, len(dataset))
    indices = torch.randperm(len(dataset))[:count]

    fig, axes = plt.subplots(count, 1, figsize=(10, 2 * count))
    if count == 1:
        axes = [axes]

    for axis, idx in zip(axes, indices):
        item = dataset[idx.item()]
        img_tensor = item["pixel_values"]
        img_display = (img_tensor / 2 + 0.5).clamp(0, 1)
        img_display = img_display.permute(1, 2, 0).numpy()
        axis.imshow(img_display)
        axis.set_title(f"Label: '{item['text']}'")
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved inspection to '{args.output}'.")


if __name__ == "__main__":
    inspect()
