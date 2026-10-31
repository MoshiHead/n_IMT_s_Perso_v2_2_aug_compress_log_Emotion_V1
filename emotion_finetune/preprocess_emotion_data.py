"""Preprocess raw robot-face videos + MEAD wavs into the IMTalker generator
training format (motion + audio modalities).

Pairing: videos and wavs are matched by (speaker, emotion, level) prefix -
the trailing sequence numbers may differ (e.g. M003_happy_level_1_001.mp4
was generated from M003_happy_level_1_030.wav).  Inside each prefix group
both sides are sorted by sequence number and paired positionally.

Outputs (per paired sample, named after the VIDEO stem):

  <output>/motion/<video_stem>.pt    Tensor[T, 32]   (renderer motion latents)
  <output>/audio/<video_stem>.npy    ndarray[T, 768] (wav2vec2 last hidden, 25 fps)

This produces the two modalities the live-matched emotion training actually
conditions on.  If you also need smirk/gaze, run the original preprocessing
notebooks (01/02) on the paired staging directory this script can optionally
write (--paired_dir).

Usage:
  python emotion_finetune/preprocess_emotion_data.py \
      --video_dir /workspace/robot_videos \
      --audio_dir /workspace/mead_wavs \
      --output_dir /workspace/emotion_generator_dataset \
      --renderer_path IMTalker/checkpoints/renderer.ckpt \
      --wav2vec_model_path IMTalker/checkpoints/wav2vec2-base-960h \
      [--paired_dir /workspace/paired] [--device cuda] [--overwrite]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "IMTalker"), str(_REPO_ROOT / "IMTalker" / "generator")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from emotion_finetune.emotion_dataset import build_prefix_pairing, parse_mead_stem

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def count_frames(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    n = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        n += 1
    cap.release()
    return n


@torch.no_grad()
def extract_motion(renderer, video_path: Path, device: str, batch_size: int = 16) -> torch.Tensor:
    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    cap = cv2.VideoCapture(str(video_path))
    latents, buf = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf.append(transform(Image.fromarray(frame_rgb)))
        if len(buf) == batch_size:
            latents.append(renderer.mot_encode(torch.stack(buf).to(device)).cpu())
            buf = []
    if buf:
        latents.append(renderer.mot_encode(torch.stack(buf).to(device)).cpu())
    cap.release()
    if not latents:
        raise RuntimeError(f"no frames read from {video_path}")
    motion = torch.cat(latents, dim=0)
    assert motion.ndim == 2 and motion.shape[1] == 32, f"unexpected motion shape {motion.shape}"
    return motion


@torch.no_grad()
def extract_audio_features(wav2vec, processor, wav_path: Path, num_frames: int, fps: float, device: str) -> np.ndarray:
    import librosa

    speech, _sr = librosa.load(str(wav_path), sr=16000)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt").input_values[0]
    target_samples = int(num_frames * 16000 / fps)
    if inputs.shape[0] < target_samples:
        inputs = F.pad(inputs.unsqueeze(0), (0, target_samples - inputs.shape[0]), mode="replicate").squeeze(0)
    else:
        inputs = inputs[:target_samples]
    out = wav2vec(inputs.unsqueeze(0).to(device), seq_len=num_frames, output_hidden_states=False)
    feat = out.last_hidden_state.squeeze(0).float().cpu().numpy()
    assert feat.shape == (num_frames, 768), f"unexpected feature shape {feat.shape}"
    return feat.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--renderer_path", required=True)
    parser.add_argument("--wav2vec_model_path", required=True)
    parser.add_argument("--paired_dir", default="", help="optional: write paired video/wav symlinks here")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    motion_dir = output_dir / "motion"
    audio_out_dir = output_dir / "audio"
    motion_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(p for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    wavs = sorted(audio_dir.glob("*.wav"))
    if not videos:
        raise SystemExit(f"no videos found in {video_dir}")
    if not wavs:
        raise SystemExit(f"no wav files found in {audio_dir}")

    video_stems = [p.stem for p in videos]
    audio_stems = [p.stem for p in wavs]
    pairing, warnings = build_prefix_pairing(video_stems, audio_stems)
    for warning in warnings:
        print(f"[pairing][warn] {warning}")
    video_by_stem = {p.stem: p for p in videos}
    wav_by_stem = {p.stem: p for p in wavs}
    pairs = [
        (video_by_stem[v], wav_by_stem[a])
        for v, a in sorted(pairing.items())
        if v in video_by_stem and a in wav_by_stem
    ]
    skipped = [v for v in video_stems if v not in pairing]
    print(f"[pairing] {len(pairs)} video<->audio pairs; {len(skipped)} videos unmatched")
    if skipped:
        print(f"[pairing] unmatched videos (first 5): {skipped[:5]}")
    no_emotion = [v for v, _ in pairs if parse_mead_stem(v.stem) is None]
    if no_emotion:
        print(f"[pairing][warn] {len(no_emotion)} paired videos have no emotion token, e.g. {no_emotion[:5]}")

    if args.paired_dir:
        paired = Path(args.paired_dir)
        (paired / "videos").mkdir(parents=True, exist_ok=True)
        (paired / "wavs").mkdir(parents=True, exist_ok=True)
        for video_path, wav_path in pairs:
            for src, dst_dir in ((video_path, paired / "videos"), (wav_path, paired / "wavs")):
                dst = dst_dir / src.name
                if not dst.exists():
                    try:
                        dst.symlink_to(src.resolve())
                    except OSError:
                        import shutil

                        shutil.copy2(src, dst)
        print(f"[pairing] paired staging written to {paired} (run notebooks 01/02 there for smirk/gaze)")

    # --- models ---------------------------------------------------------------
    from renderer.models import IMTRenderer
    from generator.wav2vec2 import Wav2VecModel
    from transformers import Wav2Vec2FeatureExtractor

    device = args.device
    renderer_args = argparse.Namespace(swin_res_threshold=128, num_heads=8, window_size=8)
    renderer = IMTRenderer(renderer_args)
    ckpt = torch.load(args.renderer_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    cleaned = {k.replace("gen.", "", 1): v for k, v in state.items() if k.startswith("gen.")}
    renderer.load_state_dict(cleaned, strict=False)
    renderer.to(device).eval()
    print(f"[model] renderer loaded: {args.renderer_path}")

    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_model_path, local_files_only=True)
    wav2vec = Wav2VecModel.from_pretrained(args.wav2vec_model_path, local_files_only=True).to(device).eval().float()
    for param in wav2vec.parameters():
        param.requires_grad_(False)
    print(f"[model] wav2vec2 loaded: {args.wav2vec_model_path}")

    made = failed = 0
    for index, (video_path, wav_path) in enumerate(pairs, start=1):
        stem = video_path.stem
        motion_out = motion_dir / f"{stem}.pt"
        audio_out = audio_out_dir / f"{stem}.npy"
        if not args.overwrite and motion_out.exists() and audio_out.exists():
            continue
        try:
            motion = extract_motion(renderer, video_path, device, args.batch_size)
            feat = extract_audio_features(wav2vec, processor, wav_path, int(motion.shape[0]), args.fps, device)
            torch.save(motion, str(motion_out))
            np.save(str(audio_out), feat)
            made += 1
        except Exception as exc:
            print(f"[FAIL] {stem}: {exc}")
            failed += 1
        if index == 1 or index % 50 == 0:
            print(f"[progress] {index}/{len(pairs)} made={made} failed={failed}", flush=True)

    print("=" * 70)
    print(f"PREPROCESSING DONE: {made} samples written to {output_dir} (failures: {failed})")
    print("next: run emotion_finetune/verify_emotion_dataset.py")
    print("=" * 70)


if __name__ == "__main__":
    main()