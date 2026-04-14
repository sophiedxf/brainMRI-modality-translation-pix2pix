import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm


def load_single_modality(patient_dir: Path, suffix: str) -> Path:
    matches = list(patient_dir.glob(f"*-{suffix}.nii.gz"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 '*-{suffix}.nii.gz' in {patient_dir}, found {len(matches)}"
        )
    return matches[0]


def load_volume(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    vol = img.get_fdata(dtype=np.float32)
    return vol


def normalise_volume_to_minus1_1(vol: np.ndarray) -> np.ndarray:
    """
    Per-volume normalisation:
    - use non-zero voxels only
    - percentile clip 1..99
    - z-score
    - clip z to [-3, 3]
    - divide by 3 -> [-1, 1]
    Background is set to -1 so it displays as black in the official repo.
    """
    vol = vol.astype(np.float32, copy=False)
    out = np.full_like(vol, -1.0, dtype=np.float32)

    mask = vol > 0
    if not np.any(mask):
        return out

    vals = vol[mask]
    lo, hi = np.percentile(vals, [1.0, 99.0])
    vals = np.clip(vals, lo, hi)

    mean = float(vals.mean())
    std = float(vals.std())
    if std < 1e-8:
        std = 1.0

    norm_vals = (np.clip(vol[mask], lo, hi) - mean) / std
    norm_vals = np.clip(norm_vals, -3.0, 3.0) / 3.0
    out[mask] = norm_vals.astype(np.float32)

    return out


def resize_2d(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if arr.shape == (out_h, out_w):
        return arr.astype(np.float32, copy=False)

    zoom_factors = (out_h / arr.shape[0], out_w / arr.shape[1])
    resized = zoom(arr, zoom_factors, order=1)

    # guard against rounding edge-cases
    if resized.shape != (out_h, out_w):
        fixed = np.zeros((out_h, out_w), dtype=np.float32)
        h = min(out_h, resized.shape[0])
        w = min(out_w, resized.shape[1])
        fixed[:h, :w] = resized[:h, :w]
        resized = fixed

    return resized.astype(np.float32, copy=False)


def get_axial_slice_range(num_slices: int, z_start: float, z_end: float):
    start_idx = int(np.floor(num_slices * z_start))
    end_idx = int(np.ceil(num_slices * z_end))

    start_idx = max(0, min(start_idx, num_slices - 1))
    end_idx = max(start_idx + 1, min(end_idx, num_slices))

    return range(start_idx, end_idx)


def main():
    parser = argparse.ArgumentParser(description="Convert BraTS T1n/T2w volumes into paired per-slice NPZ files.")
    parser.add_argument("--raw_root", type=str, required=True, help="Root folder containing BraTS patient directories.")
    parser.add_argument("--out_root", type=str, required=True, help="Output root for per-patient NPZ slices.")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--z_start", type=float, default=0.20, help="Start fraction of axial slices to keep.")
    parser.add_argument("--z_end", type=float, default=0.80, help="End fraction of axial slices to keep.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not (0.0 <= args.z_start < args.z_end <= 1.0):
        raise ValueError("Require 0 <= z_start < z_end <= 1")

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    patient_dirs = [p for p in sorted(raw_root.iterdir()) if p.is_dir()]
    if not patient_dirs:
        raise RuntimeError(f"No patient directories found in: {raw_root}")

    total_saved = 0

    for patient_dir in tqdm(patient_dirs, desc="Patients"):
        patient_id = patient_dir.name

        try:
            t1n_path = load_single_modality(patient_dir, "t1n")
            t2w_path = load_single_modality(patient_dir, "t2w")
        except FileNotFoundError as e:
            print(f"[WARN] Skipping {patient_id}: {e}")
            continue

        vol_A = load_volume(t1n_path)
        vol_B = load_volume(t2w_path)

        if vol_A.shape != vol_B.shape:
            raise ValueError(
                f"Shape mismatch for {patient_id}: "
                f"T1n {vol_A.shape} vs T2w {vol_B.shape}"
            )

        vol_A = normalise_volume_to_minus1_1(vol_A)
        vol_B = normalise_volume_to_minus1_1(vol_B)

        z_range = get_axial_slice_range(vol_A.shape[2], args.z_start, args.z_end)
        patient_out_dir = out_root / patient_id
        patient_out_dir.mkdir(parents=True, exist_ok=True)

        saved_for_patient = 0
        for z in z_range:
            slice_A = vol_A[:, :, z]
            slice_B = vol_B[:, :, z]

            # skip fully blank slices only
            if np.count_nonzero(slice_A) == 0 and np.count_nonzero(slice_B) == 0:
                continue

            slice_A = resize_2d(slice_A, args.height, args.width)
            slice_B = resize_2d(slice_B, args.height, args.width)

            slice_A = np.clip(slice_A, -1.0, 1.0).astype(np.float32, copy=False)
            slice_B = np.clip(slice_B, -1.0, 1.0).astype(np.float32, copy=False)

            out_path = patient_out_dir / f"slice_{z:03d}.npz"
            if out_path.exists() and not args.overwrite:
                continue

            np.savez_compressed(
                out_path,
                A=slice_A,
                B=slice_B,
                z=np.int16(z),
            )
            saved_for_patient += 1
            total_saved += 1

        print(f"[OK] {patient_id}: saved {saved_for_patient} slices")

    print(f"Done. Saved {total_saved} NPZ slices to: {out_root}")


if __name__ == "__main__":
    main()