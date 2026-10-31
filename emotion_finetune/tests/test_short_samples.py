"""Verify the short-sample fix: a dataset with some short clips must PASS verification."""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent


def make_dataset(root: Path):
    rng = np.random.default_rng(0)
    for sub in ("motion", "audio", "smirk", "gaze"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    emotions = ["happy", "sad", "angry", "surprise", "fear"]
    for emotion in emotions:
        for i in range(4):
            # make every 4th sample too short (T < 34)
            T = 30 if i == 3 else 60
            stem = f"M003_{emotion}_level_1_{i:03d}"
            torch.save(torch.randn(T, 32), root / "motion" / f"{stem}.pt")
            np.save(root / "audio" / f"{stem}.npy", rng.standard_normal((T, 768)).astype(np.float32))
            torch.save({"pose_params": torch.zeros(T, 3), "cam": torch.zeros(T, 3)}, root / "smirk" / f"{stem}.pt")
            np.save(root / "gaze" / f"{stem}.npy", np.zeros((T, 2), dtype=np.float32))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "ds"
        make_dataset(root)
        cmd = [
            sys.executable,
            str(_REPO / "emotion_finetune" / "verify_emotion_dataset.py"),
            "--dataset_path", str(root),
            "--wav2vec_sec", "0.96",
            "--require_smirk_gaze", "1",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_REPO))
        out = proc.stdout + proc.stderr
        print(out)
        assert "short samples" in out.lower(), "expected a short-sample warning"
        assert "SKIPPED" in out, "expected SKIP message"
        assert proc.returncode == 0, f"verification should PASS (got exit {proc.returncode})"
        assert "DATASET VERIFIED" in out, "expected DATASET VERIFIED"
        print("=" * 60)
        print("SHORT-SAMPLE FIX VERIFIED: dataset with short clips passes")
        print("=" * 60)


if __name__ == "__main__":
    main()