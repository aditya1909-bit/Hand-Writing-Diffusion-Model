import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, DDPMScheduler
from diffusers.models.attention_processor import AttnProcessor
from transformers import BertModel

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
    def __init__(self, image_size=(64, 256), device="cpu"):
        super().__init__()
        self.device = device
        self.image_size = image_size
        
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        
        self.style_dim = 512
        self.style_encoder = StyleEncoder(output_dim=self.style_dim)
        
        self.unet = UNet2DConditionModel(
            sample_size=image_size,
            in_channels=3,
            out_channels=3,
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
        
    def forward(self, batch):
        clean_images = batch["pixel_values"].to(self.device)
        style_images = batch["style_pixel_values"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        mask = batch["attention_mask"].to(self.device)
        
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(input_ids, attention_mask=mask)[0]
            
        style_emb = self.style_encoder(style_images)
        
        latents = clean_images
        
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()
        
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=style_emb 
        ).sample
        
        return torch.nn.functional.mse_loss(noise_pred, noise)