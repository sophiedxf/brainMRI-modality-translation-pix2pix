import random
from pathlib import Path

import numpy as np
import torch

from data.base_dataset import BaseDataset


class BratsnpzDataset(BaseDataset):
    """
    Custom paired dataset for BraTS NPZ slices.

    Expected structure:
        dataroot/
            splits/
                train.txt
                val.txt
                test.txt
            slices_npz/
                <patient_id>/
                    slice_###.npz
                    ...
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument(
            "--split_dir",
            type=str,
            default=None,
            help="Directory containing train.txt / val.txt / test.txt. Default: <dataroot>/splits",
        )
        parser.add_argument(
            "--npz_dir",
            type=str,
            default=None,
            help="Directory containing per-patient NPZ slice folders. Default: <dataroot>/slices_npz",
        )

        # sensible defaults for this MRI setup
        parser.set_defaults(
            input_nc=1,
            output_nc=1,
            load_size=256,
            crop_size=256,
            preprocess="none",
            no_flip=True,
        )
        return parser

    def __init__(self, opt):
        super().__init__(opt)

        self.phase = opt.phase
        self.split_dir = Path(opt.split_dir) if opt.split_dir else Path(opt.dataroot) / "splits"
        self.npz_dir = Path(opt.npz_dir) if opt.npz_dir else Path(opt.dataroot) / "slices_npz"

        split_file = self.split_dir / f"{self.phase}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r", encoding="utf-8") as f:
            patient_ids = [line.strip() for line in f if line.strip()]

        if not patient_ids:
            raise RuntimeError(f"No patient IDs found in split file: {split_file}")

        self.samples = []
        missing_patients = []

        for patient_id in patient_ids:
            patient_dir = self.npz_dir / patient_id
            if not patient_dir.exists():
                missing_patients.append(patient_id)
                continue

            npz_paths = sorted(patient_dir.glob("slice_*.npz"))
            for npz_path in npz_paths:
                self.samples.append(npz_path)

        if missing_patients:
            raise FileNotFoundError(
                f"Missing preprocessed patient folders in {self.npz_dir}: "
                f"{missing_patients[:10]}{' ...' if len(missing_patients) > 10 else ''}"
            )

        if not self.samples:
            raise RuntimeError(
                f"No NPZ slice files found for phase='{self.phase}' under {self.npz_dir}"
            )

        print(
            f"[BratsnpzDataset] phase={self.phase}, patients={len(patient_ids)}, "
            f"slices={len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def _load_npz(self, npz_path: Path):
        with np.load(npz_path) as data:
            A = data["A"].astype(np.float32)
            B = data["B"].astype(np.float32)

        if A.ndim != 2 or B.ndim != 2:
            raise ValueError(f"Expected 2D arrays in {npz_path}, got A{A.shape}, B{B.shape}")

        if A.shape != B.shape:
            raise ValueError(f"Shape mismatch in {npz_path}: A{A.shape}, B{B.shape}")

        A = np.clip(A, -1.0, 1.0)
        B = np.clip(B, -1.0, 1.0)

        return A, B

    def __getitem__(self, index):
        npz_path = self.samples[index]
        A, B = self._load_npz(npz_path)

        A = torch.from_numpy(np.ascontiguousarray(A)).unsqueeze(0)  # [1, H, W]
        B = torch.from_numpy(np.ascontiguousarray(B)).unsqueeze(0)  # [1, H, W]

        # Optional paired horizontal flip during training only.
        # Default is disabled via no_flip=True.
        if self.phase == "train" and not self.opt.no_flip:
            if random.random() > 0.5:
                A = torch.flip(A, dims=[2])
                B = torch.flip(B, dims=[2])

        sample_id = f"{npz_path.parent.name}_{npz_path.stem}"
        display_path = str(npz_path.parent / f"{sample_id}.npz")

        return {
            "A": A,
            "B": B,
            "A_paths": display_path,
            "B_paths": display_path,
        }