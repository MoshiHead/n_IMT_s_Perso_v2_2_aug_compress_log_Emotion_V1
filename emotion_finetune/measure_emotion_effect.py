"""Measure how much the emotion conditioning actually moves the generated motion.

For a dataset sample, this generates the motion latent with emotion=neutral
and with each emotion at several --emotion_scale values, then reports the L2
distance from neutral.  It answers two questions:

  1. Is the emotion effect real (distance > 0 even at scale=1)?
  2. Does increasing emotion_scale amplify it cleanly (distance grows ~linearly)?

If both are true, the model learned the right DIRECTION and you only need a
larger emotion_scale (or a retrain with --emotion_cond_scale) for visibility.

Usage:
  python emotion_finetune/measure_emotion_effect.py \
      --dataset_path /workspace/emotion_generator_dataset \
      --generator_path IMTalker/checkpoints/generator.ckpt \
      --renderer_path IMTalker/checkpoints/renderer.ckpt \
      --ref_path IMTalker/assets/3robert.jpeg \
      --base_lora_path checkpoints/live_winner/lora/....ckpt \
      --emotion_adapter_path exps/emotion_lora_v1/emotion_adapter_final.ckpt \
      --num_samples 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "IMTalker"), str(_REPO_ROOT / "IMTalker" / "generator")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from emotion_finetune.emotion_dataset import EmotionAudioMotionDataset
from emotion_finetune.emotion_modules import (
    EMOTIONS,
    clean_generator_state,
    clear_emotion,
    set_emotion,
    set_emotion_guidance,
    upgrade_fm_with_emotion,
)
from emotion_finetune.evaluate_emotion_model import build_opt, load_ref_x


def load_fm(args):
    from generator.FM import FMGenerator
    from generator.train_lora import apply_lora_to_model

    fm = FMGenerator(build_opt(args))
    ckpt = torch.load(args.generator_path, map_location="cpu")
    fm.load_state_dict(clean_generator_state(ckpt), strict=False)
    if args.base_lora_path:
        apply_lora_to_model(fm, rank=int(args.base_lora_rank), alpha=float(args.base_lora_alpha), dropout=0.0)
        lora_ckpt = torch.load(args.base_lora_path, map_location="cpu")
        fm.load_state_dict(clean_generator_state(lora_ckpt), strict=False)
    upgrade_fm_with_emotion(
        fm,
        adapter_path=args.emotion_adapter_path,
        lora_rank=int(args.emotion_lora_rank),
        lora_alpha=float(args.emotion_lora_alpha),
        verbose=False,
    )
    fm.to(args.device).eval()
    return fm


@torch.no_grad()
def gen_motion(fm, a_feat, ref_x, args, emotion, guidance):
    if emotion is None:
        clear_emotion(fm)
    else:
        set_emotion(fm, emotion, scale=1.0)
        set_emotion_guidance(fm, guidance)
    data = {"a_feat": a_feat.unsqueeze(0), "ref_x": ref_x}
    motion = fm.sample(data, a_cfg_scale=float(args.a_cfg_scale), nfe=int(args.nfe), seed=int(args.seed))
    clear_emotion(fm)
    return motion.squeeze(0).float().cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--generator_path", required=True)
    parser.add_argument("--renderer_path", required=True)
    parser.add_argument("--ref_path", required=True)
    parser.add_argument("--base_lora_path", default="")
    parser.add_argument("--base_lora_rank", type=int, default=64)
    parser.add_argument("--base_lora_alpha", type=float, default=128.0)
    parser.add_argument("--emotion_adapter_path", required=True)
    parser.add_argument("--emotion_lora_rank", type=int, default=16)
    parser.add_argument("--emotion_lora_alpha", type=float, default=32.0)
    parser.add_argument("--wav2vec_sec", type=float, default=0.96)
    parser.add_argument("--a_cfg_scale", type=float, default=1.13)
    parser.add_argument("--nfe", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--scales", default="1,1.5,2,2.5,3",
                        help="emotion_guidance values to sweep (output-space, in-distribution)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    scales = [float(s) for s in str(args.scales).split(",") if s.strip()]

    data_opt = argparse.Namespace(
        dataset_path=args.dataset_path, wav2vec_sec=float(args.wav2vec_sec),
        fps=25.0, num_prev_frames=10,
    )
    probe = EmotionAudioMotionDataset(data_opt, conditioning_mode="live", require_smirk_gaze=False)
    rng = np.random.default_rng(0)
    indices = rng.choice(len(probe.samples), size=min(int(args.num_samples), len(probe.samples)), replace=False)
    samples = [probe.samples[int(i)] for i in indices]

    fm = load_fm(args)
    ref_x = load_ref_x(args)
    clip = int(float(args.wav2vec_sec) * 25.0)

    print("=" * 74)
    print("EMOTION EFFECT MAGNITUDE (L2 distance of generated motion from neutral)")
    print("=" * 74)
    header = "sample".ljust(28) + "emotion".ljust(10) + "".join(f"g{s:g}".rjust(10) for s in scales)
    print(header)
    print("-" * len(header))

    all_rows = []
    for sample in samples:
        audio_feat = torch.from_numpy(np.load(sample["audio_path"], mmap_mode="r").copy()).float()[:clip].to(args.device)
        neutral = gen_motion(fm, audio_feat, ref_x, args, None, 1.0)
        neutral_norm = neutral.norm().item()
        for emotion in EMOTIONS:
            row = []
            for scale in scales:
                motion = gen_motion(fm, audio_feat, ref_x, args, emotion, scale)
                dist = (motion - neutral).norm().item()
                row.append(dist)
            all_rows.append((emotion, row, neutral_norm))
            print(sample["stem"][:28].ljust(28) + emotion.ljust(10) + "".join(f"{d:10.3f}" for d in row))
        print(f"{'':28}{'(neutral |m|=' + format(neutral_norm, '.2f') + ')':>18}")

    print("-" * len(header))
    print("mean over samples:")
    by_emotion = {e: [] for e in EMOTIONS}
    for emotion, row, _ in all_rows:
        by_emotion[emotion].append(row)
    for emotion in EMOTIONS:
        arr = np.array(by_emotion[emotion]).mean(axis=0)
        print("  " + emotion.ljust(10) + "".join(f"{d:10.3f}" for d in arr))

    print("=" * 74)
    base = np.array([r[1][0] for r in all_rows]).mean()
    big = np.array([r[1][-1] for r in all_rows]).mean()
    print(f"mean distance at guidance={scales[0]:g}: {base:.3f}   at guidance={scales[-1]:g}: {big:.3f}")
    if base < 0.5:
        print("-> the guidance=1 effect is SMALL (sub-visible). Amplify with --emotion_guidance")
        print("   (output-space, in-distribution) at inference, or retrain with --emotion_cond_scale.")
    if big > 1.5 * base:
        print("-> the effect GROWS ~linearly with guidance (in-distribution), so guidance amplification")
        print("   strengthens the expression WITHOUT the out-of-distribution distortion of emotion_scale.")
    print("=" * 74)


if __name__ == "__main__":
    main()