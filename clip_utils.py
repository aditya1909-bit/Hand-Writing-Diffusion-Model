import torch
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor


class ClipScorer:
    def __init__(self, model_name, device):
        self.device = device
        try:
            self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
        except TypeError:
            self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.to_pil = transforms.ToPILImage()

    @torch.no_grad()
    def score(self, images, texts):
        if images.ndim != 4:
            raise ValueError("images must be [batch, 3, H, W]")

        images = (images / 2 + 0.5).clamp(0, 1).cpu()
        pil_images = [self.to_pil(img) for img in images]

        inputs = self.processor(
            text=texts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        similarity = (image_embeds * text_embeds).sum(dim=-1)
        return similarity
