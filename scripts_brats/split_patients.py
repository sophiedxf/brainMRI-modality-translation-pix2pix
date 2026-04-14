import argparse
import random
from pathlib import Path


def find_patient_ids(raw_root: Path):
    patient_ids = []
    for patient_dir in sorted(raw_root.iterdir()):
        if not patient_dir.is_dir():
            continue

        has_t1n = any(patient_dir.glob("*-t1n.nii.gz"))
        has_t2w = any(patient_dir.glob("*-t2w.nii.gz"))
        if has_t1n and has_t2w:
            patient_ids.append(patient_dir.name)

    return patient_ids


def write_list(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item}\n")


def main():
    parser = argparse.ArgumentParser(description="Create patient-level train/val/test splits for BraTS2023.")
    parser.add_argument("--raw_root", type=str, required=True, help="Root folder containing BraTS patient directories.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output folder for train.txt / val.txt / test.txt")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)

    patient_ids = find_patient_ids(raw_root)
    if not patient_ids:
        raise RuntimeError(f"No valid patient folders found in: {raw_root}")

    rng = random.Random(args.seed)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(round(n * args.train_ratio))
    n_val = int(round(n * args.val_ratio))

    # ensure all patients are assigned exactly once
    if n_train + n_val > n:
        n_val = max(0, n - n_train)

    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train + n_val]
    test_ids = patient_ids[n_train + n_val:]

    if len(train_ids) == 0 or len(test_ids) == 0:
        raise RuntimeError(
            f"Split produced an empty train or test set: "
            f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
        )

    write_list(out_dir / "train.txt", train_ids)
    write_list(out_dir / "val.txt", val_ids)
    write_list(out_dir / "test.txt", test_ids)

    print(f"Found {n} patients")
    print(f"Train: {len(train_ids)}")
    print(f"Val:   {len(val_ids)}")
    print(f"Test:  {len(test_ids)}")
    print(f"Saved split files to: {out_dir}")


if __name__ == "__main__":
    main()