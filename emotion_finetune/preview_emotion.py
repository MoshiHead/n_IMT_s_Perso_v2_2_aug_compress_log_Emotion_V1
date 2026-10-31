"""Offline emotion preview: render the same reference + audio with the
original model (neutral) and with each of the 5 trained emotions, then write
a side-by-side comparison video.

Uses the repository's own InferenceAgent (generator/generate.py) so the
render path is identical to the original offline inference.

Usage:
  python emotion_finetune/preview_emotion.py \
      --ref_path IMTalker/assets/3robert.jpeg \
      --aud_path some_audio.wav \
      --generator_path IMTalker/checkpoints/generator.ckpt \
      --renderer_path IMTalker/checkpoints/renderer.ckpt \
      --wav2vec_model_path IMTalker/checkpoints/wav2vec2-base-960h \
      --base_lora_path checkpoints/live_winner/lora/ditto_blink_lora_withaudio_r64_096_continue_2h_last.ckpt \
      --emotion_adapter_path exps/emotion_lora_v1/emotion_adapter_final.ckpt \
      --out_dir outputs/emotion_preview --name robot --crop
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "IMTalker"), str(_REPO_ROOT / "IMTalker" / "generator")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from emotion_finetune.emotion_modules import (
    EMOTIONS,
    clean_generator_state,
    clear_emotion,
    set_emotion,
    upgrade_fm_with_emotion,
)


def build_agent_opt(args) -> argparse.Namespace:
    """Namespace compatible with generator/generate.py InferenceAgent."""
    return argparse.Namespace(
        rank=args.device,
        ngpus=1,
        fps=25,
        sampling_rate=16000,
        input_size=512,
        input_nc=3,
        audio_marcing=2,
        attention_window=5,
        audio_dropout_prob=0.1,
        ref_dropout_prob=0.1,
        emotion_dropout_prob=0.1,
        style_dim=512,
        dim_a=512,
        dim_h=512,
        dim_e=7,
        dim_motion=32,
        dim_c=32,
        dim_w=32,
        fmt_depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        no_learned_pe=False,
        depth=2,
        latent_dim=32,
        swin_res_threshold=128,
        window_size=8,
        drop_path=0.1,
        low_res_depth=2,
        num_prev_frames=10,
        wav2vec_sec=float(args.wav2vec_sec),
        only_last_features=True,
        skip_fm_audio_encoder=False,
        audio_adapter_mode="none",
        no_pose_cam=False,
        ode_atol=1e-5,
        ode_rtol=1e-5,
        torchdiffeq_ode_method="euler",
        a_cfg_scale=float(args.a_cfg_scale),
        nfe=int(args.nfe),
        seed=int(args.seed),
        fix_noise_seed=True,
        static_pose_cam=False,
        debug_session=False,
        renderer_path=args.renderer_path,
        generator_path=args.generator_path,
        wav2vec_model_path=args.wav2vec_model_path,
        res_dir=args.out_dir,
        enable_eye_blink_composite=False,
        blink_motion_path="",
        render_sub_batch=8,
        eye_left_x=0.36,
        eye_right_x=0.64,
        eye_center_y=0.405,
        eye_radius_x=0.145,
        eye_radius_y=0.070,
        eye_feather=0.10,
    )


def load_agent(args):
    from generator.generate import InferenceAgent
    from generator.train_lora import apply_lora_to_model

    opt = build_agent_opt(args)
    agent = InferenceAgent(opt)
    if args.base_lora_path:
        apply_lora_to_model(
            agent.fm,
            rank=int(args.base_lora_rank),
            alpha=float(args.base_lora_alpha),
            dropout=0.0,
        )
        lora_ckpt = torch.load(args.base_lora_path, map_location="cpu")
        missing, unexpected = agent.fm.load_state_dict(clean_generator_state(lora_ckpt), strict=False)
        print(f"[preview] with-audio LoRA loaded: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if args.emotion_adapter_path:
        upgrade_fm_with_emotion(
            agent.fm,
            adapter_path=args.emotion_adapter_path,
            lora_rank=int(args.emotion_lora_rank),
            lora_alpha=float(args.emotion_lora_alpha),
        )
    agent.fm.to(args.device).eval()
    return agent


def render_all(agent, args) -> list[tuple[str, str]]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    renders: list[tuple[str, str]] = []
    settings = ["neutral"] + list(EMOTIONS)
    for setting in settings:
        if setting == "neutral":
            if getattr(agent.fm, "_emotion_upgraded", False):
                clear_emotion(agent.fm)
        else:
            set_emotion(agent.fm, setting, scale=float(args.emotion_scale))
        out_path = out_dir / f"{args.name}_{setting}.mp4"
        print(f"[preview] rendering {setting} -> {out_path}", flush=True)
        agent.run_inference(
            str(out_path), args.ref_path, args.aud_path,
            pose_path=None, gaze_path=None,
            a_cfg_scale=float(args.a_cfg_scale), nfe=int(args.nfe),
            crop=bool(args.crop), seed=int(args.seed),
        )
        renders.append((setting, str(out_path)))
    if getattr(agent.fm, "_emotion_upgraded", False):
        clear_emotion(agent.fm)
    return renders


def write_comparison(renders: list[tuple[str, str]], out_path: str) -> None:
    inputs: list[str] = []
    filters: list[str] = []
    for index, (label, path) in enumerate(renders):
        inputs.extend(["-i", path])
        filters.append(
            f"[{index}:v]scale=512:512,setsar=1,"
            f"drawtext=text='{label}':x=12:y=12:fontsize=22:"
            "fontcolor=white:box=1:boxcolor=black@0.45:boxborderw=6"
            f"[v{index}]"
        )
    filter_complex = (
        ";".join(filters)
        + ";"
        + "".join(f"[v{index}]" for index in range(len(renders)))
        + f"xstack=inputs={len(renders)}:layout=0_0|512_0|1024_0|0_512|512_512|1024_512[v]"
    )
    subprocess.check_call(
        [
            "ffmpeg", "-y", *inputs,
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-shortest", out_path,
        ]
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ref_path", required=True)
    p.add_argument("--aud_path", required=True)
    p.add_argument("--renderer_path", required=True)
    p.add_argument("--generator_path", required=True)
    p.add_argument("--wav2vec_model_path", required=True)
    p.add_argument("--base_lora_path", default="")
    p.add_argument("--base_lora_rank", type=int, default=64)
    p.add_argument("--base_lora_alpha", type=float, default=128.0)
    p.add_argument("--emotion_adapter_path", default="")
    p.add_argument("--emotion_lora_rank", type=int, default=16)
    p.add_argument("--emotion_lora_alpha", type=float, default=32.0)
    p.add_argument("--emotion_scale", type=float, default=1.0)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--name", default="preview")
    p.add_argument("--wav2vec_sec", type=float, default=0.96)
    p.add_argument("--a_cfg_scale", type=float, default=1.13)
    p.add_argument("--nfe", type=int, default=5)
    p.add_argument("--seed", type=int, default=25)
    p.add_argument("--crop", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_comparison", action="store_true")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    agent = load_agent(args)
    renders = render_all(agent, args)
    if not args.no_comparison:
        comparison = str(Path(args.out_dir) / f"{args.name}_comparison.mp4")
        write_comparison(renders, comparison)
        print(f"[preview] comparison video: {comparison}", flush=True)
    print("[preview] done", flush=True)


if __name__ == "__main__":
    main()