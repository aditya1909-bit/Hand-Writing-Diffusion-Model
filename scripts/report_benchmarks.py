import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate benchmark JSON files.")
    parser.add_argument("--bench-dir", default="benchmarks", help="Benchmark directory.")
    parser.add_argument("--output", default="benchmarks/summary.json", help="Summary JSON path.")
    parser.add_argument("--readme", default="README.md", help="README file to update.")
    parser.add_argument("--write-readme", action="store_true", help="Update README section.")
    return parser.parse_args()


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def format_benchmark_lines(summary):
    lines = []

    dataset = summary.get("dataset")
    if dataset:
        lines.append(
            f"- Dataset: {dataset.get('samples_with_images', 'n/a')} samples, "
            f"{dataset.get('unique_writers', 'n/a')} writers"
        )
    else:
        lines.append("- Dataset: not recorded")

    fid = summary.get("fid")
    if fid:
        fid_value = fid.get("fid")
        fid_str = f"{fid_value:.2f}" if isinstance(fid_value, (int, float)) else "n/a"
        lines.append(
            f"- FID (IAM, {fid.get('num_samples', 'n/a')} samples, "
            f"{fid.get('num_steps', 'n/a')} steps, {fid.get('scheduler', 'n/a')}): {fid_str}"
        )
    else:
        lines.append("- FID (IAM): not recorded")

    train = summary.get("train")
    if train:
        samples_per_sec = train.get("samples_per_sec")
        speed = f"{samples_per_sec:.2f} samples/s" if isinstance(samples_per_sec, (int, float)) else "n/a"
        lines.append(
            f"- Train throughput: {speed} (batch {train.get('batch_size', 'n/a')})"
        )
    else:
        lines.append("- Train throughput: not recorded")

    infer = summary.get("infer")
    if infer:
        samples_per_sec = infer.get("samples_per_sec")
        speed = f"{samples_per_sec:.2f} samples/s" if isinstance(samples_per_sec, (int, float)) else "n/a"
        lines.append(
            f"- Inference throughput: {speed} "
            f"(batch {infer.get('batch_size', 'n/a')}, {infer.get('num_steps', 'n/a')} steps)"
        )
    else:
        lines.append("- Inference throughput: not recorded")

    return "\n".join(lines)


def update_readme(readme_path, content):
    if not os.path.exists(readme_path):
        raise FileNotFoundError(f"README not found at {readme_path}")

    with open(readme_path, "r", encoding="utf-8") as handle:
        text = handle.read()

    start = "<!-- BENCHMARKS:START -->"
    end = "<!-- BENCHMARKS:END -->"
    if start not in text or end not in text:
        raise ValueError("README is missing benchmark markers.")

    before = text.split(start)[0]
    after = text.split(end)[1]
    new_block = f"{start}\n{content}\n{end}"
    updated = before + new_block + after

    with open(readme_path, "w", encoding="utf-8") as handle:
        handle.write(updated)


def main():
    args = parse_args()
    bench_dir = args.bench_dir

    summary = {
        "dataset": load_json(os.path.join(bench_dir, "dataset.json")),
        "fid": load_json(os.path.join(bench_dir, "fid.json")),
        "train": load_json(os.path.join(bench_dir, "train.json")),
        "infer": load_json(os.path.join(bench_dir, "infer.json")),
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"Wrote summary to {args.output}")

    if args.write_readme:
        lines = format_benchmark_lines(summary)
        update_readme(args.readme, lines)
        print(f"Updated {args.readme}")


if __name__ == "__main__":
    main()
