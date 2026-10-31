"""Verify an emotion dataset against the IMTalker generator training contract.

Checks (against the ACTUAL training code, not just file existence):

  structure   motion/ audio/ smirk/ gaze/ present, stems match (with
              prefix-group audio pairing for mismatched sequence numbers)
  shapes      motion [T,32] float32 | audio [T,768] float32 |
              smirk pose_params/cam [T,3] | gaze [T,2] float32
  alignment   all modalities share the same T per sample
  length      T >= wav2vec_sec*fps + num_prev_frames (34 for the 0.96s live contract).
              Shorter samples are SKIPPED (a warning, not a failure) - the
              training dataset class skips them too.  Only a suspiciously large
              short fraction (> --max_short_fraction) is treated as a failure.
  labels      every stem parses to one of the 5 emotions; per-emotion counts
  values      finite tensors, plausible ranges
  separability per-emotion motion-latent statistics + nearest-centroid accuracy
  dataloader  a real EmotionAudioMotionDataset batch with correct shapes
  recompute   (optional, --recompute_check N) re-run the repo wav2vec2 on N
              raw wavs and compare against the stored audio/*.npy

Usage:
  python emotion_finetune/verify_emotion_dataset.py \
      --dataset_path /workspace/emotion_generator_dataset \
      [--wav2vec_sec 0.96] [--recompute_check 4 \
       --raw_audio_dir /workspace/mead_wavs \
       --wav2vec_model_path IMTalker/checkpoints/wav2vec2-base-960h]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "IMTalker")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from emotion_finetune.emotion_dataset import (
    EmotionAudioMotionDataset,
    build_prefix_pairing,
    collate_emotion_batch,
    parse_mead_stem,
)
from emotion_finetune.emotion_modules import EMOTIONS

FAILURES: list[str] = []
WARNINGS: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  [FAIL] {msg}")


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  [warn] {msg}")


def ok(msg: str) -> None:
    print(f"  [ok] {msg}")


def check_structure(root: Path) -> dict:
    print("\n[1/7] directory structure")
    dirs = {}
    for name in ("motion", "audio", "smirk", "gaze"):
        path = root / name
        dirs[name] = path
        if not path.is_dir():
            fail(f"missing directory: {path}")
        else:
            count = len(list(path.glob("*.pt"))) + len(list(path.glob("*.npy")))
            ok(f"{name}/: {count} files")
    return dirs


def check_samples(dirs: dict, args) -> list[dict]:
    print("\n[2/7] per-sample shapes / dtypes / alignment")
    motion_stems = sorted(p.stem for p in dirs["motion"].glob("*.pt"))
    audio_stems = sorted(p.stem for p in dirs["audio"].glob("*.npy"))
    pairing, pair_warnings = build_prefix_pairing(motion_stems, audio_stems)
    for warning in pair_warnings:
        warn(warning)
    unmatched = [s for s in motion_stems if s not in pairing]
    if unmatched:
        fail(f"{len(unmatched)} motion files have no audio match, e.g. {unmatched[:5]}")
    else:
        ok(f"all {len(motion_stems)} motion files matched to audio")

    required_len = int(float(args.wav2vec_sec) * float(args.fps)) + int(args.num_prev_frames)
    print(f"  required_len = {required_len} frames (wav2vec_sec={args.wav2vec_sec}, "
          f"fps={args.fps}, num_prev_frames={args.num_prev_frames})")

    records = []
    short_samples: list[tuple[str, int]] = []
    checked = 0
    for stem in motion_stems:
        info = parse_mead_stem(stem)
        if info is None:
            fail(f"{stem}: no emotion token in filename")
            continue
        audio_stem = pairing.get(stem)
        if audio_stem is None:
            continue
        try:
            motion = torch.load(dirs["motion"] / f"{stem}.pt", map_location="cpu")
            audio = np.load(dirs["audio"] / f"{audio_stem}.npy", mmap_mode="r")
            T = int(motion.shape[0])
            problems = []
            if not (motion.ndim == 2 and motion.shape[1] == 32):
                problems.append(f"motion shape {tuple(motion.shape)} != (T,32)")
            if motion.dtype != torch.float32:
                problems.append(f"motion dtype {motion.dtype} != float32")
            if not (audio.ndim == 2 and audio.shape[1] == 768):
                problems.append(f"audio shape {audio.shape} != (T,768)")
            if audio.dtype != np.float32:
                problems.append(f"audio dtype {audio.dtype} != float32")
            if int(audio.shape[0]) != T:
                problems.append(f"audio T={audio.shape[0]} != motion T={T}")
            if not torch.isfinite(motion).all():
                problems.append("motion contains NaN/Inf")
            if not np.isfinite(audio[:]).all():
                problems.append("audio contains NaN/Inf")
            smirk_path = dirs["smirk"] / f"{stem}.pt"
            gaze_path = dirs["gaze"] / f"{stem}.npy"
            if smirk_path.exists():
                smirk = torch.load(smirk_path, map_location="cpu")
                if "pose_params" not in smirk or tuple(smirk["pose_params"].shape) != (T, 3):
                    problems.append(f"smirk pose_params shape {tuple(smirk.get('pose_params', torch.empty(0)).shape)} != ({T},3)")
                if "cam" not in smirk or tuple(smirk["cam"].shape) != (T, 3):
                    problems.append(f"smirk cam shape {tuple(smirk.get('cam', torch.empty(0)).shape)} != ({T},3)")
            else:
                problems.append("smirk file missing")
            if gaze_path.exists():
                gaze = np.load(gaze_path, mmap_mode="r")
                if gaze.shape != (T, 2):
                    problems.append(f"gaze shape {gaze.shape} != ({T},2)")
            else:
                problems.append("gaze file missing")
            if problems:
                fail(f"{stem}: " + "; ".join(problems))
            elif T < required_len:
                short_samples.append((stem, T))
            else:
                records.append({"stem": stem, "emotion": info["emotion"], "T": T})
            checked += 1
            if checked >= int(args.max_check):
                break
        except Exception as exc:
            fail(f"{stem}: load error {exc}")
    if short_samples:
        fraction = len(short_samples) / max(1, len(motion_stems))
        examples = ", ".join(f"{s}(T={t})" for s, t in short_samples[:5])
        warn(
            f"{len(short_samples)} samples are shorter than required_len={required_len} "
            f"and will be SKIPPED during training (e.g. {examples}). "
            "These are naturally short clips; the training dataset class skips them automatically."
        )
        if fraction > float(args.max_short_fraction):
            fail(
                f"{fraction:.1%} of samples are too short (< {required_len} frames) -- "
                "this looks like a systematic problem (wrong fps? truncated preprocessing?), "
                "not a few short clips"
            )
    if records:
        ok(f"{len(records)} samples passed all shape/dtype/alignment checks "
           f"({len(short_samples)} short samples skipped)")
    return records

def check_labels(records: list[dict]) -> None:
    print("\n[3/7] emotion labels")
    counts = defaultdict(int)
    for record in records:
        counts[record["emotion"]] += 1
    for emotion in EMOTIONS:
        if counts.get(emotion, 0) == 0:
            fail(f"no samples for emotion '{emotion}'")
        else:
            ok(f"{emotion}: {counts[emotion]} samples")
    unknown = set(counts) - set(EMOTIONS)
    if unknown:
        warn(f"stems contain non-target emotions (ignored by training): {sorted(unknown)}")


def check_separability(dirs: dict, records: list[dict], args) -> None:
    print("\n[4/7] per-emotion motion-latent separability")
    rng = np.random.default_rng(0)
    per_emotion: dict[str, list[np.ndarray]] = defaultdict(list)
    sample_records = [r for r in records if r["emotion"] in EMOTIONS]
    rng.shuffle(sample_records)
    cap = int(args.separability_samples)
    picked = sample_records[:cap]
    for record in picked:
        motion = torch.load(dirs["motion"] / f"{record['stem']}.pt", map_location="cpu").numpy()
        feat = np.concatenate([motion.mean(axis=0), motion.std(axis=0)], axis=0)
        per_emotion[record["emotion"]].append(feat)
    if any(len(v) < 2 for v in per_emotion.values()) or len(per_emotion) < 2:
        warn("not enough samples per emotion for a separability estimate")
        return
    centroids = {e: np.stack(v).mean(axis=0) for e, v in per_emotion.items()}
    correct = total = 0
    for emotion, feats in per_emotion.items():
        for feat in feats:
            dists = {e: float(np.linalg.norm(feat - c)) for e, c in centroids.items()}
            if min(dists, key=dists.get) == emotion:
                correct += 1
            total += 1
    acc = correct / max(1, total)
    print(f"  nearest-centroid accuracy on {total} samples: {acc:.3f} "
          f"(chance = {1.0/len(per_emotion):.3f})")
    if acc < 0.35:
        warn("emotions are barely separable in the motion-latent space -- "
             "check that the videos really contain distinct expressions")
    else:
        ok("emotion classes are separable in the motion-latent space")
    for emotion in EMOTIONS:
        if emotion in per_emotion:
            feats = np.stack(per_emotion[emotion])
            print(f"  {emotion:9s} n={len(feats):4d} latent|mean|={np.linalg.norm(feats.mean(axis=0)):.3f} "
              f"avg_std={feats[:, 32:].mean():.4f}")


def check_dataloader(args) -> None:
    print("\n[5/7] real dataset + DataLoader batch")
    opt = argparse.Namespace(
        dataset_path=args.dataset_path,
        wav2vec_sec=float(args.wav2vec_sec),
        fps=float(args.fps),
        num_prev_frames=int(args.num_prev_frames),
    )
    dataset = EmotionAudioMotionDataset(
        opt,
        conditioning_mode=args.conditioning_mode,
        require_smirk_gaze=bool(args.require_smirk_gaze),
    )
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_emotion_batch)
    batch = next(iter(loader))
    clip = int(float(args.wav2vec_sec) * float(args.fps))
    prev = int(args.num_prev_frames)
    expectations = {
        "m_now": (2, clip, 32), "a_now": (2, clip, 768),
        "m_prev": (2, prev, 32), "a_prev": (2, prev, 768),
        "gaze_now": (2, clip, 2), "pose_now": (2, clip, 3), "cam_now": (2, clip, 3),
        "m_ref": (2, 32), "emotion": (2, len(EMOTIONS)),
    }
    for key, shape in expectations.items():
        actual = tuple(batch[key].shape)
        if actual != shape:
            fail(f"batch[{key}] shape {actual} != {shape}")
    emotion_sums = batch["emotion"].sum(dim=-1)
    if not torch.allclose(emotion_sums, torch.ones_like(emotion_sums)):
        fail("emotion vectors are not one-hot")
    ok(f"DataLoader batch OK: m_now {tuple(batch['m_now'].shape)}, "
       f"a_now {tuple(batch['a_now'].shape)}, emotions={batch['emotion_name']}")


def check_recompute(dirs: dict, records: list[dict], args) -> None:
    print("\n[6/7] wav2vec recompute check")
    if int(args.recompute_check) <= 0:
        print("  skipped (set --recompute_check N to enable)")
        return
    raw_audio_dir = Path(args.raw_audio_dir)
    if not raw_audio_dir.is_dir():
        fail(f"--recompute_check needs --raw_audio_dir with the source wavs (got {raw_audio_dir})")
        return
    from generator.wav2vec2 import Wav2VecModel
    from transformers import Wav2Vec2FeatureExtractor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_model_path, local_files_only=True)
    model = Wav2VecModel.from_pretrained(args.wav2vec_model_path, local_files_only=True).to(device).eval().float()
    import librosa

    checked = 0
    for record in records:
        if checked >= int(args.recompute_check):
            break
        wav_path = raw_audio_dir / f"{record['stem']}.wav"
        if not wav_path.exists():
            candidates = sorted(raw_audio_dir.glob(f"{record['stem'].rsplit('_', 1)[0]}_*.wav"))
            if not candidates:
                warn(f"{record['stem']}: no raw wav found for recompute check")
                continue
            wav_path = candidates[0]
        speech, _sr = librosa.load(str(wav_path), sr=16000)
        inputs = processor(speech, sampling_rate=16000, return_tensors="pt").input_values[0]
        T = record["T"]
        target_samples = int(T * 16000 / float(args.fps))
        if inputs.shape[0] < target_samples:
            inputs = torch.nn.functional.pad(inputs.unsqueeze(0), (0, target_samples - inputs.shape[0]), mode="replicate").squeeze(0)
        else:
            inputs = inputs[:target_samples]
        with torch.no_grad():
            out = model(inputs.unsqueeze(0).to(device), seq_len=T, output_hidden_states=False)
            feat = out.last_hidden_state.squeeze(0).float().cpu().numpy()
        stored = np.load(dirs["audio"] / f"{record['stem']}.npy", mmap_mode="r")
        if stored.shape != feat.shape:
            fail(f"{record['stem']}: recomputed {feat.shape} != stored {stored.shape}")
            continue
        cos = float(np.dot(feat.ravel(), stored[:].ravel()) /
                    (np.linalg.norm(feat) * np.linalg.norm(stored[:]) + 1e-9))
        max_diff = float(np.abs(feat - stored[:]).max())
        print(f"  {record['stem']}: cosine={cos:.5f} max|diff|={max_diff:.4f}")
        if cos < 0.999:
            fail(f"{record['stem']}: stored audio features do not match a fresh wav2vec2 pass "
                 f"(cosine={cos:.5f}) -- wrong extractor/model or misalignment")
        checked += 1
    if checked:
        ok(f"{checked} samples match a fresh wav2vec2 pass")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--wav2vec_sec", type=float, default=0.96)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--num_prev_frames", type=int, default=10)
    parser.add_argument("--conditioning_mode", choices=("live", "full"), default="live")
    parser.add_argument("--require_smirk_gaze", type=int, default=1)
    parser.add_argument("--max_check", type=int, default=100000)
    parser.add_argument("--max_short_fraction", type=float, default=0.5,
                        help="fail if more than this fraction of samples are too short (systematic problem)")
    parser.add_argument("--separability_samples", type=int, default=250)
    parser.add_argument("--recompute_check", type=int, default=0)
    parser.add_argument("--raw_audio_dir", default="")
    parser.add_argument("--wav2vec_model_path", default="")
    args = parser.parse_args()

    root = Path(args.dataset_path)
    print(f"verifying emotion dataset: {root}")
    if not root.is_dir():
        raise SystemExit(f"dataset path does not exist: {root}")

    dirs = check_structure(root)
    records = check_samples(dirs, args)
    check_labels(records)
    if records:
        check_separability(dirs, records, args)
    check_dataloader(args)
    check_recompute(dirs, records, args)

    print("\n[7/7] summary")
    print("=" * 70)
    if FAILURES:
        print(f"DATASET VERIFICATION FAILED: {len(FAILURES)} problem(s)")
        for msg in FAILURES[:25]:
            print(f"  - {msg}")
        raise SystemExit(1)
    print(f"DATASET VERIFIED ({len(WARNINGS)} warning(s)) -- ready for emotion training")
    print("=" * 70)


if __name__ == "__main__":
    main()