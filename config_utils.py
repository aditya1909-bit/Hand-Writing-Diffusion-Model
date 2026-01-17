import copy
import os

import yaml

DEFAULT_CONFIG = {
    "data": {
        "root_dir": "./iam_data",
        "words_file": "words.txt",
        "words_dir": "words",
        "image_size": [64, 256],
        "max_length": 32,
        "num_writers": None,
        "mock_mode": False,
        "pad_value": 255,
        "skip_err": True,
        "seed": 1337,
    },
    "model": {
        "text_encoder": "bert-base-uncased",
        "style_dim": 512,
        "latent": {
            "enabled": True,
            "latent_channels": 4,
            "downsample_factor": 4,
        },
        "scheduler": {
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
        },
    },
    "train": {
        "epochs": 250,
        "batch_size": 32,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.05,
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
        "text_drop_prob": 0.1,
        "style_drop_prob": 0.1,
        "cond_drop_prob": 0.05,
        "style_cls_weight": 0.1,
        "style_contrastive_weight": 0.1,
        "style_contrastive_temperature": 0.07,
        "autoencoder_recon_weight": 1.0,
        "save_dir": "./saved_models",
        "save_every": 5,
        "resume": True,
        "num_workers": 4,
        "prefetch_factor": 4,
        "grad_accum_steps": 1,
        "amp": True,
        "tf32": True,
        "cudnn_benchmark": True,
        "channels_last": True,
        "gradient_checkpointing": False,
        "compile": False,
        "compile_mode": "max-autotune",
        "ema": True,
        "ema_decay": 0.9999,
        "min_snr_gamma": 5.0,
        "val_split": 0.05,
        "val_split_by_writer": False,
        "val_batch_size": 16,
        "val_every": 1,
        "val_max_batches": 20,
        "eval_steps": 30,
        "eval_guidance_scale": 3.0,
        "log_dir": "./logs",
        "log_images_every": 5,
        "log_images": 8,
        "clip_metric": True,
        "clip_model": "openai/clip-vit-base-patch32",
        "clip_every": 5,
        "eval_use_ema": True,
        "device": "auto",
        "log_every": 25,
    },
    "generate": {
        "output_dir": "./generated_outputs",
        "num_steps": 50,
        "scheduler": "ddpm",
        "use_ema": True,
        "guidance_scale": 3.0,
        "style_mix": 0.5,
    },
}


def _deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def load_config(path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Config file must contain a YAML mapping at the top level.")
        _deep_update(config, loaded)
    return config


def resolve_device(preferred):
    if preferred and preferred != "auto":
        return preferred

    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
