"""Map RGB validity to Wan's causal 1+4n temporal compression."""

import torch


def latent_video_validity(frame_valid):
    if frame_valid.ndim != 2 or (frame_valid.shape[1] - 1) % 4:
        raise ValueError("Wan VAE requires one current frame plus 4n future frames")
    frame_valid = frame_valid.bool()
    if not frame_valid[:, 0].all():
        raise ValueError("The conditioning frame must be valid")
    # Causal convolutions carry history across groups: conservatively exclude
    # the group containing the first padded RGB and every later latent group.
    prefix_valid = frame_valid.to(torch.int64).cumprod(dim=1).bool()
    return torch.cat((prefix_valid[:, :1], prefix_valid[:, 4::4]), dim=1)


def masked_video_loss(prediction, target, latent_valid):
    valid = latent_valid.clone()
    valid[:, 0] = False  # teacher-forced conditioning latent
    per_frame = (prediction.float() - target.float()).square().mean(dim=(1, 3, 4))
    return (per_frame * valid).sum() / valid.sum().clamp_min(1)
