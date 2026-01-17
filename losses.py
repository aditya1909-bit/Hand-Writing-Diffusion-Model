import torch
import torch.nn.functional as F


def supervised_contrastive_loss(embeddings, labels, temperature=0.07):
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D [batch, dim].")
    device = embeddings.device
    labels = labels.to(device)

    valid_mask = labels >= 0
    if valid_mask.sum() < 2:
        return embeddings.new_tensor(0.0)

    embeddings = embeddings[valid_mask]
    labels = labels[valid_mask]

    embeddings = F.normalize(embeddings, dim=1)
    similarity = torch.matmul(embeddings, embeddings.T) / temperature

    logits_max = similarity.max(dim=1, keepdim=True).values
    logits = similarity - logits_max.detach()

    batch_size = embeddings.size(0)
    mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask.fill_diagonal_(False)

    logits_mask = torch.ones((batch_size, batch_size), device=device)
    logits_mask.fill_diagonal_(0)
    exp_logits = torch.exp(logits) * logits_mask

    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    mask_sum = mask.sum(dim=1)
    valid = mask_sum > 0
    if not valid.any():
        return embeddings.new_tensor(0.0)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum.clamp(min=1)
    loss = -mean_log_prob_pos[valid].mean()
    return loss
