import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

log_path = Path(r"checkpoints\brats_27k_maskedL1_SSIM_bs4_inst_lsgan\loss_log.txt")
text = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()

pattern = re.compile(
    r"epoch:\s*(\d+),\s*iters:\s*(\d+).*?"
    r"G_GAN:\s*([0-9.]+),\s*G_L1:\s*([0-9.]+),\s*G_SSIM:\s*([0-9.]+),\s*D_real:\s*([0-9.]+),\s*D_fake:\s*([0-9.]+)"
)

records = []
for line in text:
    m = pattern.search(line)
    if m:
        records.append({
            "epoch": int(m.group(1)),
            "iters": int(m.group(2)),
            "G_GAN": float(m.group(3)),
            "G_L1": float(m.group(4)),
            "G_SSIM": float(m.group(5)),
            "D_real": float(m.group(6)),
            "D_fake": float(m.group(7)),
        })

if not records:
    raise RuntimeError(f"No loss lines parsed from {log_path}")

def moving_average(arr, window=15):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")

for key in ["G_GAN", "G_L1", "G_SSIM", "D_real", "D_fake"]:
    y = [r[key] for r in records]
    y_smooth = moving_average(y, window=15)
    x_smooth = np.arange(len(y_smooth))
    plt.figure(figsize=(8, 4.5))
    plt.plot(x_smooth, y_smooth)
    plt.xlabel("Logged training point")
    plt.ylabel(key)
    plt.title(f"{key} (smoothed)")
    plt.tight_layout()
    plt.show()