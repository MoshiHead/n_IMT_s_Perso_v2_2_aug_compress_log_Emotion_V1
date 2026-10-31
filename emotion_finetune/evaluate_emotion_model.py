"""Quantitative evaluation of the trained emotion adapter.

Generates MOTION LATENTS (no video rendering -> fast) for held-out dataset
samples under 7 settings and reports:

  1. neutral preservation : original stack vs upgraded stack with emotion=0
     (relative L2 + velocity correlation; expect ~0)
  2. emotion distinctiveness : nearest-centroid classification of generated
     motion against centroids computed from the TRAINING split
  3. lip-sync retention : correlation between per-frame motion-delta energy
     and audio-feature-delta energy, per setting (should stay comparable)
  4. temporal stability : mean jerk (2nd difference) per setting

Usage:
  python emotion_finetune/evaluate_emotion_model.py \
      --dataset_path /workspace/emotion_generator_dataset \
      --generator_path IMTalker/checkpoints/generator.ckpt \
      --renderer_path IMTalker/checkpoints/renderer.ckpt \
      --base_lora_path checkpoints/live_winner/lora/....ckpt \
      --emotion_adapter_path exps/emotion_lora_v1/emotion_adapter_final.ckpt \
      --ref_path IMTalker/assets/3robert.jpeg --num_samples 24
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "IMTalker")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from emotion_finetune.emotion_dataset import EmotionAudioMotionDataset, stratified_split
from emotion_finetune.emotion_modules import (
    EMOTIONS,
    clean_generator_state,
    clear_emotion,
    set_emotion,
    upgrade_fm_with_emotion,
)

SETTINGS = ["original", "neutral"] + list(EMOTIONS)


def build_opt(args) -> argparse.Namespace:
    return argparse.Namespace(
        fps=25.0,
        rank=args.device,
        wav2vec_sec=float(args.wav2vec_sec),
        num_prev_frames=10,
        only_last_features=True,
        skip_fm_audio_encoder=True,
        audio_adapter_mode="none",
        no_pose_cam=True,
        ode_atol=1e-5,
        ode_rtol=1e-5,
        torchdiffeq_ode_method="euler",
        dim_c=32,
        dim_w=32,
        dim_motion=32,
        dim_h=512,
        mlp_ratio=4.0,
        fmt_depth=8,
        num_heads=8,
        audio_dropout_prob=0.1,
        seed=int(args.seed),
        fix_noise_seed=True,
        static_pose_cam=False,
        debug_session=False,
    )


def load_fm(args, with_emotion: bool):
    from generator.FM import FMGenerator
    from generator.train_lora import apply_lora_to_model

    fm = FMGenerator(build_opt(args))
    ckpt = torch.load(args.generator_path, map_location="cpu")
    fm.load_state_dict(clean_generator_state(ckpt), strict=False)
    if args.base_lora_path:
        apply_lora_to_model(fm, rank=int(args.base_lora_rank), alpha=float(args.base_lora_alpha), dropout=0.0)
        lora_ckpt = torch.load(args.base_lora_path, map_location="cpu")
        fm.load_state_dict(clean_generator_state(lora_ckpt), strict=False)
    if with_emotion:
        upgrade_fm_with_emotion(
            fm,
            adapter_path=args.emotion_adapter_path,
            lora_rank=int(args.emotion_lora_rank),
            lora_alpha=float(args.emotion_lora_alpha),
            verbose=False,
        )
    fm.to(args.device).eval()
    return fm


def load_ref_x(args):
    from renderer.models import IMTRenderer

    renderer_args = argparse.Namespace(swin_res_threshold=128, num_heads=8, window_size=8)
    renderer = IMTRenderer(renderer_args)
    ckpt = torch.load(args.renderer_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    cleaned = {k.replace("gen.", "", 1): v for k, v in state.items() if k.startswith("gen.")}
    renderer.load_state_dict(cleaned, strict=False)
    renderer.to(args.device).eval()
    img = Image.open(args.ref_path).convert("RGB").resize((512, 512), Image.LANCZOS)
    tensor = T.ToTensor()(img).unsqueeze(0).to(args.device)
    with torch.no_grad():
        ref_x = renderer.latent_token_encoder(tensor).float()
    del renderer
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return ref_x


@torch.no_grad()
def generate_motion(fm, a_feat: torch.Tensor, ref_x: torch.Tensor, args) -> torch.Tensor:
    data = {"a_feat": a_feat.unsqueeze(0), "ref_x": ref_x}
    motion = fm.sample(data, a_cfg_scale=float(args.a_cfg_scale), nfe=int(args.nfe), seed=int(args.seed))
    return motion.squeeze(0).float().cpu()


def motion_features(motion: torch.Tensor) -> np.ndarray:
    arr = motion.numpy()
    return np.concatenate([arr.mean(axis=0), arr.std(axis=0)], axis=0)


def jerk(motion: torch.Tensor) -> float:
    if motion.shape[0] < 3:
        return 0.0
    return float((motion[2:] - 2 * motion[1:-1] + motion[:-2]).abs().mean())


def sync_score(motion: torch.Tensor, audio: torch.Tensor) -> float:
    dm = (motion[1:] - motion[:-1]).norm(dim=-1).numpy()
    da = (audio[1:] - audio[:-1]).norm(dim=-1).numpy()
    n = min(len(dm), len(da))
    if n < 4 or dm.std() < 1e-8 or da.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(dm[:n], da[:n])[0, 1])

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
    parser.add_argument("--emotion_scale", type=float, default=1.0)
    parser.add_argument("--wav2vec_sec", type=float, default=0.96)
    parser.add_argument("--a_cfg_scale", type=float, default=1.13)
    parser.add_argument("--nfe", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=24)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_json", default="")
    args = parser.parse_args()

    # ---- held-out samples ---------------------------------------------------
    data_opt = argparse.Namespace(
        dataset_path=args.dataset_path,
        wav2vec_sec=float(args.wav2vec_sec),
        fps=25.0,
        num_prev_frames=10,
    )
    probe = EmotionAudioMotionDataset(data_opt, conditioning_mode="live", require_smirk_gaze=False)
    train_stems, val_stems = stratified_split(probe.samples, args.val_fraction, seed=int(args.seed))
    rng = np.random.default_rng(0)
    rng.shuffle(val_stems)
    eval_stems = val_stems[: int(args.num_samples)]
    by_stem = {s["stem"]: s for s in probe.samples}
    print(f"[eval] {len(eval_stems)} held-out samples from {len(val_stems)} val stems")

    # ---- models ---------------------------------------------------------------
    fm_original = load_fm(args, with_emotion=False)
    fm_emotion = load_fm(args, with_emotion=True)
    ref_x = load_ref_x(args)

    # ---- training-data centroids (for distinctiveness) ------------------------
    train_features: dict[str, list[np.ndarray]] = {e: [] for e in EMOTIONS}
    for stem in train_stems[:400]:
        sample = by_stem[stem]
        motion = torch.load(sample["motion_path"], map_location="cpu").float()
        train_features[sample["emotion"]].append(motion_features(motion))
    centroids = {
        e: np.stack(v).mean(axis=0) for e, v in train_features.items() if v
    }

    # ---- generation loop ------------------------------------------------------
    neutral_rel_errors: list[float] = []
    neutral_vel_corr: list[float] = []
    per_setting_sync: dict[str, list[float]] = {s: [] for s in SETTINGS}
    per_setting_jerk: dict[str, list[float]] = {s: [] for s in SETTINGS}
    classify_correct = classify_total = 0
    per_emotion_correct: dict[str, list[int]] = {e: [0, 0] for e in EMOTIONS}

    for stem in eval_stems:
        sample = by_stem[stem]
        audio_feat = torch.from_numpy(
            np.load(sample["audio_path"], mmap_mode="r").copy()
        ).float()
        clip = int(float(args.wav2vec_sec) * 25.0)
        audio_feat = audio_feat[:clip].to(args.device)
        true_emotion = sample["emotion"]

        motions: dict[str, torch.Tensor] = {}
        motions["original"] = generate_motion(fm_original, audio_feat, ref_x, args)
        clear_emotion(fm_emotion)
        motions["neutral"] = generate_motion(fm_emotion, audio_feat, ref_x, args)
        for emotion in EMOTIONS:
            set_emotion(fm_emotion, emotion, scale=float(args.emotion_scale))
            motions[emotion] = generate_motion(fm_emotion, audio_feat, ref_x, args)
        clear_emotion(fm_emotion)

        # 1) neutral preservation
        diff = (motions["neutral"] - motions["original"])
        rel = float(diff.norm() / (motions["original"].norm() + 1e-9))
        neutral_rel_errors.append(rel)
        v_orig = motions["original"][1:] - motions["original"][:-1]
        v_neut = motions["neutral"][1:] - motions["neutral"][:-1]
        cos = float(F.cosine_similarity(v_orig.flatten(), v_neut.flatten(), dim=0))
        neutral_vel_corr.append(cos)

        # 2) distinctiveness: classify the TRUE-emotion render
        feat = motion_features(motions[true_emotion])
        dists = {e: float(np.linalg.norm(feat - c)) for e, c in centroids.items()}
        predicted = min(dists, key=dists.get)
        classify_total += 1
        per_emotion_correct[true_emotion][1] += 1
        if predicted == true_emotion:
            classify_correct += 1
            per_emotion_correct[true_emotion][0] += 1

        # 3+4) sync + jerk per setting
        audio_cpu = audio_feat.cpu()
        for setting in SETTINGS:
            per_setting_sync[setting].append(sync_score(motions[setting], audio_cpu))
            per_setting_jerk[setting].append(jerk(motions[setting]))

    # ---- report -----------------------------------------------------------------
    report = {
        "num_samples": len(eval_stems),
        "neutral_preservation": {
            "relative_l2_mean": float(np.mean(neutral_rel_errors)),
            "relative_l2_max": float(np.max(neutral_rel_errors)),
            "velocity_cosine_mean": float(np.mean(neutral_vel_corr)),
        },
        "emotion_distinctiveness": {
            "nearest_centroid_accuracy": classify_correct / max(1, classify_total),
            "chance": 1.0 / len(EMOTIONS),
            "per_emotion": {
                e: {"correct": c, "total": t, "accuracy": (c / t if t else 0.0)}
                for e, (c, t) in per_emotion_correct.items()
            },
        },
        "lip_sync_retention": {
            s: float(np.mean(per_setting_sync[s])) for s in SETTINGS
        },
        "temporal_stability_jerk": {
            s: float(np.mean(per_setting_jerk[s])) for s in SETTINGS
        },
    }

    print("\n" + "=" * 70)
    print("EMOTION MODEL EVALUATION")
    print("=" * 70)
    print(f"samples: {report['num_samples']}")
    np_report = report["neutral_preservation"]
    print(f"\n[1] neutral preservation (original vs emotion=0)")
    print(f"    relative L2   : mean={np_report['relative_l2_mean']:.5f} max={np_report['relative_l2_max']:.5f}  (want ~0)")
    print(f"    velocity cos  : {np_report['velocity_cosine_mean']:.5f}  (want ~1)")
    ed = report["emotion_distinctiveness"]
    print(f"\n[2] emotion distinctiveness (nearest-centroid on generated motion)")
    print(f"    accuracy: {ed['nearest_centroid_accuracy']:.3f}  (chance {ed['chance']:.3f})")
    for emotion, stats in ed["per_emotion"].items():
        print(f"      {emotion:9s}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.3f}")
    print(f"\n[3] lip-sync retention (motion/audio delta correlation)")
    for setting in SETTINGS:
        print(f"      {setting:9s}: {report['lip_sync_retention'][setting]:.4f}")
    print(f"\n[4] temporal stability (mean jerk, lower = smoother)")
    for setting in SETTINGS:
        print(f"      {setting:9s}: {report['temporal_stability_jerk'][setting]:.5f}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, indent=2))
        print(f"\nreport written: {args.out_json}")


if __name__ == "__main__":
    main()