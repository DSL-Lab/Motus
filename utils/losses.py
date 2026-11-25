"""Shared masked video reconstruction loss."""

def masked_video_loss(prediction, target, latent_valid):
    if latent_valid.shape != (prediction.shape[0], prediction.shape[2]):
        raise ValueError("VAE latent validity shape mismatch")
    keep = latent_valid.clone().bool()
    keep[:, 0] = False
    mask = keep[:, None, :, None, None].expand_as(prediction)
    error = (prediction.float() - target.float()).square()
    # A tail window may have only the conditioning latent. Its video loss
    # is zero, while all action parameter rows remain supervised.
    valid_count = mask.flatten(1).sum(1).clamp_min(1)
    valid_mean = ((error * mask).flatten(1).sum(1) / valid_count).mean()
    # Retain the original dense loss scale, which averages over the current
    # (zero-loss) latent as well as future latents. No 1.5x video-weight shift.
    return valid_mean * (prediction.shape[2] - 1) / prediction.shape[2]
