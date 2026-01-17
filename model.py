import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler
from diffusers.models.attention_processor import AttnProcessor
from transformers import BertModel

from autoencoder import SimpleAutoencoder
from losses import supervised_contrastive_loss

class StyleEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class HandwritingDiffusionSystem(nn.Module):
    def __init__(
        self,
        image_size=(64, 256),
        device="cpu",
        text_encoder_name="bert-base-uncased",
        style_dim=512,
        scheduler_config=None,
        min_snr_gamma=None,
        num_writers=None,
        text_drop_prob=0.0,
        style_drop_prob=0.0,
        cond_drop_prob=0.0,
        style_cls_weight=0.0,
        style_contrastive_weight=0.0,
        style_contrastive_temperature=0.07,
        latent_enabled=False,
        latent_channels=4,
        latent_downsample_factor=4,
        autoencoder_recon_weight=1.0,
    ):
        super().__init__()
        self.device = device
        self.image_size = tuple(image_size)
        self.min_snr_gamma = min_snr_gamma
        self.text_drop_prob = text_drop_prob
        self.style_drop_prob = style_drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.style_cls_weight = style_cls_weight
        self.style_contrastive_weight = style_contrastive_weight
        self.style_contrastive_temperature = style_contrastive_temperature
        self.autoencoder_recon_weight = autoencoder_recon_weight
        self.latent_enabled = latent_enabled
        self.latent_channels = latent_channels
        self.latent_downsample_factor = latent_downsample_factor
        
        if scheduler_config is None:
            scheduler_config = {
                "num_train_timesteps": 1000,
                "beta_start": 0.0001,
                "beta_end": 0.02,
                "beta_schedule": "linear",
            }
        self.scheduler = DDPMScheduler(**scheduler_config)
        
        self.text_encoder = BertModel.from_pretrained(text_encoder_name)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        
        self.style_dim = style_dim
        self.style_encoder = StyleEncoder(output_dim=self.style_dim)

        if self.latent_enabled:
            if (
                self.image_size[0] % self.latent_downsample_factor != 0
                or self.image_size[1] % self.latent_downsample_factor != 0
            ):
                raise ValueError("image_size must be divisible by latent_downsample_factor.")
            self.sample_size = (
                self.image_size[0] // self.latent_downsample_factor,
                self.image_size[1] // self.latent_downsample_factor,
            )
            self.sample_channels = self.latent_channels
            self.autoencoder = SimpleAutoencoder(
                in_channels=3,
                latent_channels=self.latent_channels,
                downsample_factor=self.latent_downsample_factor,
            )
        else:
            self.sample_size = self.image_size
            self.sample_channels = 3
            self.autoencoder = None

        self.unet = UNet2DConditionModel(
            sample_size=self.sample_size,
            in_channels=self.sample_channels,
            out_channels=self.sample_channels,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D"
            ),
            up_block_types=(
                "CrossAttnUpBlock2D", 
                "CrossAttnUpBlock2D", 
                "UpBlock2D",
                "UpBlock2D"
            ),
            cross_attention_dim=768,
            class_embed_type="projection", 
            projection_class_embeddings_input_dim=self.style_dim
        )

        self.unet.set_attn_processor(AttnProcessor())

        if num_writers and num_writers > 0 and self.style_cls_weight > 0:
            self.style_classifier = nn.Linear(self.style_dim, num_writers)
        else:
            self.style_classifier = None

    def train(self, mode=True):
        super().train(mode)
        self.text_encoder.eval()
        return self
        
    def forward(self, batch):
        clean_images = batch["pixel_values"]
        device = clean_images.device
        style_images = batch["style_pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        writer_labels = batch.get("writer_labels")
        if writer_labels is not None:
            writer_labels = writer_labels.to(device)

        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(input_ids, attention_mask=mask)[0]

        style_emb_raw = self.style_encoder(style_images)
        style_emb = style_emb_raw

        bsz = clean_images.shape[0]
        if self.training:
            if self.cond_drop_prob:
                cond_drop = (torch.rand(bsz, device=device) < self.cond_drop_prob).float()
                encoder_hidden_states = encoder_hidden_states * (1 - cond_drop).view(bsz, 1, 1)
                style_emb = style_emb * (1 - cond_drop).view(bsz, 1)
            if self.text_drop_prob:
                text_drop = (torch.rand(bsz, device=device) < self.text_drop_prob).float()
                encoder_hidden_states = encoder_hidden_states * (1 - text_drop).view(bsz, 1, 1)
            if self.style_drop_prob:
                style_drop = (torch.rand(bsz, device=device) < self.style_drop_prob).float()
                style_emb = style_emb * (1 - style_drop).view(bsz, 1)

        recon_loss = torch.tensor(0.0, device=device)
        if self.latent_enabled:
            latents = self.autoencoder.encode(clean_images)
            if self.autoencoder_recon_weight > 0:
                recon = self.autoencoder.decode(latents)
                recon_loss = F.l1_loss(recon, clean_images)
        else:
            latents = clean_images

        noise = torch.randn_like(latents)

        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (bsz,),
            device=device,
        ).long()

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=style_emb,
        ).sample

        diffusion_loss = F.mse_loss(noise_pred, noise, reduction="none")
        diffusion_loss = diffusion_loss.mean(dim=(1, 2, 3))

        if self.min_snr_gamma:
            if not hasattr(self, "_alphas_cumprod") or self._alphas_cumprod.device != device:
                self._alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
            snr = self._alphas_cumprod[timesteps] / (1 - self._alphas_cumprod[timesteps])
            gamma = torch.full_like(snr, float(self.min_snr_gamma))
            snr_weight = torch.minimum(snr, gamma) / snr
            diffusion_loss = diffusion_loss * snr_weight

        diffusion_loss = diffusion_loss.mean()

        style_cls_loss = torch.tensor(0.0, device=device)
        if self.style_classifier is not None and writer_labels is not None:
            valid = writer_labels >= 0
            if valid.any():
                logits = self.style_classifier(style_emb_raw)
                style_cls_loss = F.cross_entropy(logits[valid], writer_labels[valid])

        style_contrastive_loss = torch.tensor(0.0, device=device)
        if self.style_contrastive_weight and writer_labels is not None:
            style_contrastive_loss = supervised_contrastive_loss(
                style_emb_raw, writer_labels, temperature=self.style_contrastive_temperature
            )

        total_loss = diffusion_loss
        total_loss = total_loss + self.style_cls_weight * style_cls_loss
        total_loss = total_loss + self.style_contrastive_weight * style_contrastive_loss
        total_loss = total_loss + self.autoencoder_recon_weight * recon_loss

        return {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "style_cls_loss": style_cls_loss,
            "style_contrastive_loss": style_contrastive_loss,
            "recon_loss": recon_loss,
        }
