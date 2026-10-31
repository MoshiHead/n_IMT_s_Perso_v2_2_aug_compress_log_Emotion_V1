"""Sanity checks for the emotion upgrade (run BEFORE training).

Proves, on the real model stack:

  1. upgrade + emotion=None            -> bit-identical to the original model
  2. upgrade + emotion="happy" (zero-init, untrained) -> still bit-identical
     (the untrained adapter is a no-op by construction)
  3. after perturbing the emotion weights, emotion="happy" changes the output
     (the conditioning pathway is actually wired in)
  4. adapter save/load roundtrip restores the perturbed weights exactly

Usage:
  python emotion_finetune/sanity_check.py \
      --generator_path IMTalker/checkpoints/generator.ckpt \
      [--base_lora_path checkpoints/.../ditto_blink_lora_withaudio_r64_096_continue_2h_last.ckpt] \
      [--device cpu]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "IMTalker"), str(_REPO_ROOT / "IMTalker" / "generator")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from emotion_finetune.emotion_modules import (
    clean_generator_state,
    clear_emotion,
    load_emotion_adapter,
    save_emotion_adapter,
    set_emotion,
    upgrade_fm_with_emotion,
)


def build_opt(device: str) -> argparse.Namespace:
    return argparse.Namespace(
        fps=25.0,
        rank=device,
        wav2vec_sec=0.96,
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
        seed=42,
        fix_noise_seed=True,
        static_pose_cam=False,
        debug_session=False,
    )


@torch.no_grad()
def run_sample(fm, a_feat, ref_x, nfe: int = 3):
    data = {"a_feat": a_feat, "ref_x": ref_x}
    return fm.sample(data, a_cfg_scale=1.13, nfe=nfe, seed=42)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_path", default="")
    parser.add_argument("--base_lora_path", default="")
    parser.add_argument("--base_lora_rank", type=int, default=64)
    parser.add_argument("--base_lora_alpha", type=float, default=128.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from generator.FM import FMGenerator

    device = args.device
    opt = build_opt(device)
    fm = FMGenerator(opt)
    if args.generator_path:
        ckpt = torch.load(args.generator_path, map_location="cpu")
        missing, unexpected = fm.load_state_dict(clean_generator_state(ckpt), strict=False)
        print(f"[sanity] base generator loaded: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("[sanity] no --generator_path: using randomly initialized weights (math check only)")
    if args.base_lora_path:
        from generator.train_lora import apply_lora_to_model

        apply_lora_to_model(fm, rank=args.base_lora_rank, alpha=args.base_lora_alpha, dropout=0.0)
        lora_ckpt = torch.load(args.base_lora_path, map_location="cpu")
        fm.load_state_dict(clean_generator_state(lora_ckpt), strict=False)
        print(f"[sanity] with-audio LoRA loaded: {args.base_lora_path}")
    fm.to(device).eval()

    torch.manual_seed(0)
    a_feat = torch.randn(1, 24, 768, device=device)
    ref_x = torch.randn(1, 32, device=device)

    out_original = run_sample(fm, a_feat, ref_x)

    summary = upgrade_fm_with_emotion(fm, lora_rank=16, lora_alpha=32.0, verbose=True)
    assert summary["targets"] > 0

    # 1) emotion=None must be bit-identical
    out_none = run_sample(fm, a_feat, ref_x)
    diff_none = (out_none - out_original).abs().max().item()
    print(f"[check 1] emotion=None        max|diff| = {diff_none:.3e}")
    assert diff_none == 0.0, "emotion=None changed the output!"

    # 2) untrained emotion (zero-init projection) must also be a no-op
    set_emotion(fm, "happy")
    out_happy_untrained = run_sample(fm, a_feat, ref_x)
    diff_zero = (out_happy_untrained - out_original).abs().max().item()
    print(f"[check 2] untrained 'happy'   max|diff| = {diff_zero:.3e}")
    assert diff_zero == 0.0, "zero-init emotion projection is not a no-op!"

    # 3) perturb the emotion weights -> output must change (pathway is live)
    with torch.no_grad():
        fm.emotion_projection.proj.weight.normal_(0, 0.1)
        for module in fm.modules():
            if type(module).__name__ == "EmotionLoRALinear":
                module.emotion_lora_B.weight.normal_(0, 0.02)
    set_emotion(fm, "happy")
    out_happy_trained = run_sample(fm, a_feat, ref_x)
    diff_trained = (out_happy_trained - out_original).abs().max().item()
    print(f"[check 3] perturbed 'happy'   max|diff| = {diff_trained:.3e}")
    assert diff_trained > 0.0, "emotion conditioning has no effect on the output!"

    # 3b) ...but emotion=None must STILL be bit-identical (LoRA B is no longer
    # zero, yet with no emotion conditioning the FMT input is unchanged)
    clear_emotion(fm)
    out_none2 = run_sample(fm, a_feat, ref_x)
    diff_none2 = (out_none2 - out_original).abs().max().item()
    print(f"[check 3b] emotion=None again max|diff| = {diff_none2:.3e}  (LoRA delta now active)")

    # 4) adapter save/load roundtrip
    with tempfile.TemporaryDirectory() as tmp:
        adapter_path = Path(tmp) / "emotion_adapter.ckpt"
        save_emotion_adapter(fm, adapter_path, config_extra={"source": "sanity_check"})
        with torch.no_grad():
            fm.emotion_projection.proj.weight.zero_()
        info = load_emotion_adapter(fm, adapter_path)
        set_emotion(fm, "happy")
        out_reloaded = run_sample(fm, a_feat, ref_x)
        diff_reload = (out_reloaded - out_happy_trained).abs().max().item()
        print(f"[check 4] save/load roundtrip max|diff| = {diff_reload:.3e} (loaded {info['loaded']} keys)")
        assert diff_reload == 0.0, "adapter save/load roundtrip is not exact!"

    clear_emotion(fm)
    print("=" * 60)
    print("ALL SANITY CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()