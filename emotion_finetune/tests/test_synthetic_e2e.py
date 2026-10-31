"""End-to-end synthetic test: fake dataset -> dataset class -> one training step."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
for _p in (str(_REPO), str(_REPO / "IMTalker")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def make_fake_dataset(root: Path, per_emotion: int = 3, T: int = 60):
    rng = np.random.default_rng(0)
    for sub in ("motion", "audio", "smirk", "gaze"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    emotions = ["happy", "sad", "angry", "surprise", "fear"]
    for emotion in emotions:
        for i in range(per_emotion):
            stem = f"M003_{emotion}_level_1_{i:03d}"
            motion = torch.randn(T, 32) + torch.randn(32) * (emotions.index(emotion) + 1)
            torch.save(motion, root / "motion" / f"{stem}.pt")
            np.save(root / "audio" / f"{stem}.npy", rng.standard_normal((T, 768)).astype(np.float32))
            torch.save(
                {"pose_params": torch.zeros(T, 3), "cam": torch.zeros(T, 3)},
                root / "smirk" / f"{stem}.pt",
            )
            np.save(root / "gaze" / f"{stem}.npy", np.zeros((T, 2), dtype=np.float32))


def main():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "ds"
        make_fake_dataset(root)

        from emotion_finetune.emotion_dataset import (
            EmotionAudioMotionDataset,
            collate_emotion_batch,
            stratified_split,
        )

        opt = argparse.Namespace(
            dataset_path=str(root), wav2vec_sec=0.96, fps=25.0, num_prev_frames=10,
        )
        ds = EmotionAudioMotionDataset(opt, conditioning_mode="live")
        assert len(ds) == 15, len(ds)
        train_stems, val_stems = stratified_split(ds.samples, 0.2, seed=42)
        assert len(train_stems) + len(val_stems) == 15
        batch = collate_emotion_batch([ds[0], ds[1]])
        assert tuple(batch["m_now"].shape) == (2, 24, 32), batch["m_now"].shape
        assert tuple(batch["a_now"].shape) == (2, 24, 768), batch["a_now"].shape
        assert tuple(batch["emotion"].shape) == (2, 5)
        assert torch.allclose(batch["emotion"].sum(-1), torch.ones(2))
        print("dataset + collate + split: OK")

        # one real training step on CPU
        from emotion_finetune.train_emotion import (
            build_model_opt,
            emotion_training_losses,
        )
        from emotion_finetune.emotion_modules import upgrade_fm_with_emotion
        from generator.FM import FMGenerator
        import contextlib

        args = argparse.Namespace(
            fps=25.0, wav2vec_sec=0.96, num_prev_frames=10, conditioning_mode="live",
            lambda_anchor=0.5, lambda_margin=0.2, margin=0.05, seed=42,
        )
        model_opt = build_model_opt(args, "cpu")
        fm = FMGenerator(model_opt)
        # A fresh FMGenerator has DiT-style zero-init on the decoder output and
        # every block adaLN gate, so no gradient can flow to the conditioning
        # at random init.  The real generator.ckpt is trained (non-zero), so
        # simulate that here to make the gradient check meaningful.
        with torch.no_grad():
            fm.fmt.decoder.linear.weight.normal_(0, 0.02)
            fm.fmt.decoder.linear.bias.normal_(0, 0.02)
            fm.fmt.decoder.adaLN_modulation[-1].weight.normal_(0, 0.02)
            fm.fmt.decoder.adaLN_modulation[-1].bias.normal_(0, 0.02)
            for block in fm.fmt.blocks:
                block.adaLN_modulation[-1].weight.normal_(0, 0.02)
                block.adaLN_modulation[-1].bias.normal_(0, 0.02)
        upgrade_fm_with_emotion(fm, lora_rank=16, lora_alpha=32.0, verbose=False)
        fm.eval()
        loss, metrics = emotion_training_losses(fm, batch, args, contextlib.nullcontext())
        loss.backward()
        grads = [
            p.grad.abs().sum().item()
            for p in fm.emotion_projection.parameters()
        ]
        assert all(g > 0 for g in grads), "emotion projection received no gradient"
        print(
            f"training step: OK loss={metrics['loss']:.4f} fm={metrics['fm']:.4f} "
            f"anchor={metrics['anchor']:.6f} margin={metrics['margin']:.4f}"
        )
        assert metrics["anchor"] == 0.0, "anchor must be 0 at init (zero-init LoRA)"
        print("ALL SYNTHETIC TESTS PASSED")


if __name__ == "__main__":
    main()