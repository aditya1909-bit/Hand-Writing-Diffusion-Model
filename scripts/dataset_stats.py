import argparse
import json
import os

from config_utils import load_config
from data_loader import _parse_words_line


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize IAM dataset stats.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--output", default="benchmarks/dataset.json", help="Output JSON path.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]

    words_path = os.path.join(data_cfg["root_dir"], data_cfg["words_file"])
    if not os.path.exists(words_path):
        raise FileNotFoundError(f"words file not found at {words_path}")

    with open(words_path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    total_lines = 0
    parsed_lines = 0
    missing_images = 0
    writers = set()
    text_lengths = []

    for line in lines:
        total_lines += 1
        parsed = _parse_words_line(
            line,
            data_cfg["root_dir"],
            data_cfg["words_dir"],
            data_cfg.get("skip_err", True),
        )
        if not parsed:
            continue
        parsed_lines += 1
        if not os.path.exists(parsed["path"]):
            missing_images += 1
            continue
        writers.add(parsed["writer_id"])
        text_lengths.append(len(parsed["text"]))

    count_with_images = parsed_lines - missing_images
    avg_text_len = sum(text_lengths) / len(text_lengths) if text_lengths else 0.0

    stats = {
        "words_file": words_path,
        "total_lines": total_lines,
        "parsed_lines": parsed_lines,
        "missing_images": missing_images,
        "samples_with_images": count_with_images,
        "unique_writers": len(writers),
        "avg_text_length": avg_text_len,
        "image_size": data_cfg["image_size"],
        "max_length": data_cfg["max_length"],
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
    print(f"Wrote dataset stats to {args.output}")


if __name__ == "__main__":
    main()
