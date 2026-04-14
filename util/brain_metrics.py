import math
import numpy as np
from skimage.metrics import structural_similarity


def tensor_to_01(x):
    """
    Convert a [1,1,H,W] tensor in [-1,1] to a 2D numpy array in [0,1].
    """
    arr = x.detach().cpu().float().squeeze().numpy()
    arr = (arr + 1.0) / 2.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr.astype(np.float32, copy=False)


def compute_sample_metrics(real_tensor, fake_tensor, mask_threshold=0.01):
    real = tensor_to_01(real_tensor)
    fake = tensor_to_01(fake_tensor)

    mask = real > mask_threshold

    # MAE / PSNR on masked foreground
    if np.any(mask):
        real_vals = real[mask]
        fake_vals = fake[mask]
    else:
        real_vals = real.ravel()
        fake_vals = fake.ravel()

    mae = float(np.mean(np.abs(fake_vals - real_vals)))
    mse = float(np.mean((fake_vals - real_vals) ** 2))
    psnr = float("inf") if mse == 0 else 20.0 * math.log10(1.0 / math.sqrt(mse))

    # SSIM on foreground bounding-box crop
    if np.any(mask):
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        real_crop = real[y0:y1, x0:x1]
        fake_crop = fake[y0:y1, x0:x1]
    else:
        real_crop = real
        fake_crop = fake

    if min(real_crop.shape) < 7:
        real_crop = real
        fake_crop = fake

    min_dim = min(real_crop.shape)
    win_size = 7 if min_dim >= 7 else min_dim
    if win_size % 2 == 0:
        win_size -= 1

    if win_size < 3:
        ssim = float("nan")
    else:
        ssim = float(
            structural_similarity(
                real_crop,
                fake_crop,
                data_range=1.0,
                win_size=win_size,
            )
        )

    return {
        "psnr": psnr,
        "ssim": ssim,
        "mae": mae,
    }