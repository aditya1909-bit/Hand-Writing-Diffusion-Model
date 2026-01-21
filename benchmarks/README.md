# Benchmarks

This folder stores JSON outputs from the benchmarking scripts in `scripts/`.

Expected files:
- `dataset.json` (from `scripts/dataset_stats.py`)
- `fid.json` (from `scripts/eval_fid.py`)
- `train.json` (from `scripts/benchmark_train.py`)
- `infer.json` (from `scripts/benchmark_infer.py`)
- `summary.json` (from `scripts/report_benchmarks.py`)

Run `python scripts/report_benchmarks.py --write-readme` to update the summary
block in `README.md`.
