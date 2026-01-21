import math

import torch
from diffusers import DDPMScheduler


_ALLOWED_SCHEDULER_KEYS = {
    "num_train_timesteps",
    "beta_start",
    "beta_end",
    "beta_schedule",
    "prediction_type",
    "clip_sample",
    "clip_sample_range",
    "thresholding",
    "dynamic_thresholding_ratio",
    "sample_max_value",
    "timestep_spacing",
    "steps_offset",
    "rescale_betas_zero_snr",
    "variance_type",
}


def _linear_beta_schedule(num_steps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64)


def _quadratic_beta_schedule(num_steps, beta_start, beta_end):
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps, dtype=torch.float64) ** 2


def _sigmoid_beta_schedule(num_steps, beta_start, beta_end, start=-6.0, end=6.0):
    betas = torch.sigmoid(torch.linspace(start, end, num_steps, dtype=torch.float64))
    return betas * (beta_end - beta_start) + beta_start


def _cosine_beta_schedule(num_steps, s=0.008):
    steps = num_steps + 1
    x = torch.linspace(0, num_steps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas


def _filter_scheduler_kwargs(cfg):
    return {key: value for key, value in cfg.items() if key in _ALLOWED_SCHEDULER_KEYS}


def build_scheduler(scheduler_cfg):
    cfg = dict(scheduler_cfg or {})
    num_steps = int(cfg.get("num_train_timesteps", 1000))
    beta_start = float(cfg.get("beta_start", 0.0001))
    beta_end = float(cfg.get("beta_end", 0.02))

    custom_type = cfg.get("custom_beta_schedule")
    kwargs = _filter_scheduler_kwargs(cfg)
    kwargs.setdefault("num_train_timesteps", num_steps)
    kwargs.setdefault("beta_start", beta_start)
    kwargs.setdefault("beta_end", beta_end)
    kwargs.setdefault("beta_schedule", cfg.get("beta_schedule", "linear"))

    if custom_type:
        custom_type = str(custom_type).lower()
        if custom_type == "linear":
            betas = _linear_beta_schedule(num_steps, beta_start, beta_end)
        elif custom_type == "quadratic":
            betas = _quadratic_beta_schedule(num_steps, beta_start, beta_end)
        elif custom_type == "sigmoid":
            start = float(cfg.get("custom_sigmoid_start", -6.0))
            end = float(cfg.get("custom_sigmoid_end", 6.0))
            betas = _sigmoid_beta_schedule(num_steps, beta_start, beta_end, start=start, end=end)
        elif custom_type == "cosine":
            s = float(cfg.get("custom_cosine_s", 0.008))
            betas = _cosine_beta_schedule(num_steps, s=s)
        else:
            raise ValueError(f"Unknown custom_beta_schedule '{custom_type}'.")

        beta_clip = float(cfg.get("custom_beta_clip", 0.999))
        betas = betas.clamp(1e-8, beta_clip)
        kwargs["trained_betas"] = betas

    return DDPMScheduler(**kwargs)
