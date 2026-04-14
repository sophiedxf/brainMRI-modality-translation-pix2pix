import argparse
import csv
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from skimage.metrics import structural_similarity

from options.test_options import TestOptions
from data import create_dataset
from models import create_model


def parse_args():
    # Custom args first; everything else is passed through to TestOptions
    custom_parser = argparse.ArgumentParser(add_help=False)
    custom_parser.add_argument(
        "--epochs",
        type=str,
        required=True,
        help="Comma-separated checkpoints, e.g. 5,10,15,20,latest",
    )
    custom_parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.01,
        help="Foreground threshold after mapping tensors from [-1,1] to [0,1].",
    )
    custom_parser.add_argument(
        "--summary_csv",
        type=str,
        default="",
        help="Optional CSV path for per-checkpoint summary.",
    )
    custom_parser.add_argument(
        "--per_sample_csv",
        type=str,
        default="",
        help="Optional CSV path for per-sample metrics.",
    )

    custom_args, remaining = custom_parser.parse_known_args()

    # Let the official repo parser handle the normal test.py arguments
    sys.argv = [sys.argv[0]] + remaining
    opt = TestOptions().parse()

    # Match test.py behaviour
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True

    return custom_args, opt


def tensor_to_01(x: torch.Tensor) -> np.ndarray:
    """
    Convert a [1,1,H,W] tensor in [-1,1] to a 2D numpy array in [0,1].
    """
    arr = x.detach().cpu().float().squeeze().numpy()
    arr = (arr + 1.0) / 2.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr.astype(np.float32, copy=False)


def get_foreground_mask(real_01: np.ndarray, threshold: float) -> np.ndarray:
    return real_01 > threshold


def crop_to_mask(real_01: np.ndarray, fake_01: np.ndarray, mask: np.ndarray):
    if not np.any(mask):
        return real_01, fake_01

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return real_01[y0:y1, x0:x1], fake_01[y0:y1, x0:x1]


def compute_metrics(real_01: np.ndarray, fake_01: np.ndarray, mask_threshold: float):
    """
    MAE and PSNR are computed on the masked foreground voxels.
    SSIM is computed on the target-derived foreground bounding box crop.
    """
    mask = get_foreground_mask(real_01, mask_threshold)

    # Masked MAE / PSNR
    if np.any(mask):
        real_vals = real_01[mask]
        fake_vals = fake_01[mask]
    else:
        real_vals = real_01.ravel()
        fake_vals = fake_01.ravel()

    mae = float(np.mean(np.abs(fake_vals - real_vals)))
    mse = float(np.mean((fake_vals - real_vals) ** 2))
    psnr = float("inf") if mse == 0 else 20.0 * math.log10(1.0 / math.sqrt(mse))

    # SSIM on bounding-box crop of foreground
    real_crop, fake_crop = crop_to_mask(real_01, fake_01, mask)

    # If crop too small for SSIM, fall back to full image
    if min(real_crop.shape) < 7:
        real_crop, fake_crop = real_01, fake_01

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

    return psnr, ssim, mae


def summarise_rows(rows):
    if not rows:
        return None

    psnr = np.asarray([r["psnr"] for r in rows], dtype=np.float64)
    ssim = np.asarray([r["ssim"] for r in rows], dtype=np.float64)
    mae = np.asarray([r["mae"] for r in rows], dtype=np.float64)

    return {
        "n": len(rows),
        "psnr_mean": float(np.nanmean(psnr)),
        "psnr_std": float(np.nanstd(psnr)),
        "ssim_mean": float(np.nanmean(ssim)),
        "ssim_std": float(np.nanstd(ssim)),
        "mae_mean": float(np.nanmean(mae)),
        "mae_std": float(np.nanstd(mae)),
    }


def save_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoint(opt, dataset, epoch_label: str, mask_threshold: float):
    opt.epoch = str(epoch_label)

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    rows = []

    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()

        real_01 = tensor_to_01(model.real_B)
        fake_01 = tensor_to_01(model.fake_B)

        psnr, ssim, mae = compute_metrics(real_01, fake_01, mask_threshold)

        image_path = model.get_image_paths()
        if isinstance(image_path, (list, tuple)):
            image_id = Path(image_path[0]).stem
        else:
            image_id = Path(image_path).stem

        rows.append(
            {
                "epoch": str(epoch_label),
                "sample_id": image_id,
                "psnr": psnr,
                "ssim": ssim,
                "mae": mae,
            }
        )

    return rows


def main():
    custom_args, opt = parse_args()

    epochs = [e.strip() for e in custom_args.epochs.split(",") if e.strip()]
    if not epochs:
        raise RuntimeError("No epochs provided")

    dataset = create_dataset(opt)

    all_rows = []
    summary_rows = []

    for epoch_label in epochs:
        print(f"\nEvaluating checkpoint: {epoch_label}")
        rows = evaluate_checkpoint(
            opt=opt,
            dataset=dataset,
            epoch_label=epoch_label,
            mask_threshold=custom_args.mask_threshold,
        )
        all_rows.extend(rows)

        summary = summarise_rows(rows)
        summary_row = {
            "epoch": str(epoch_label),
            **summary,
        }
        summary_rows.append(summary_row)

        print(
            f"  n={summary_row['n']} | "
            f"SSIM={summary_row['ssim_mean']:.4f} ± {summary_row['ssim_std']:.4f} | "
            f"PSNR={summary_row['psnr_mean']:.4f} ± {summary_row['psnr_std']:.4f} | "
            f"MAE={summary_row['mae_mean']:.4f} ± {summary_row['mae_std']:.4f}"
        )

    ranked = sorted(
        summary_rows,
        key=lambda r: (-r["ssim_mean"], -r["psnr_mean"], r["mae_mean"]),
    )

    print("\n=== Ranked checkpoints ===")
    for r in ranked:
        print(
            f"{r['epoch']:>8} | "
            f"n={r['n']:4d} | "
            f"SSIM={r['ssim_mean']:.4f} ± {r['ssim_std']:.4f} | "
            f"PSNR={r['psnr_mean']:.4f} ± {r['psnr_std']:.4f} | "
            f"MAE={r['mae_mean']:.4f} ± {r['mae_std']:.4f}"
        )

    if custom_args.summary_csv:
        save_csv(
            Path(custom_args.summary_csv),
            summary_rows,
            ["epoch", "n", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std", "mae_mean", "mae_std"],
        )
        print(f"\nSaved summary CSV to: {custom_args.summary_csv}")

    if custom_args.per_sample_csv:
        save_csv(
            Path(custom_args.per_sample_csv),
            all_rows,
            ["epoch", "sample_id", "psnr", "ssim", "mae"],
        )
        print(f"Saved per-sample CSV to: {custom_args.per_sample_csv}")


if __name__ == "__main__":
    main()