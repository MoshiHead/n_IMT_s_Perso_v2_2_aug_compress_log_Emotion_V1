"""Emotion dataset for IMTalker generator fine-tuning.

Follows the exact file contract of the original training pipeline
(generator/dataset.py: AudioMotionSmirkGazeDataset):

    <root>/motion/<stem>.pt    Tensor[T, 32]   (renderer motion latents)
    <root>/audio/<stem>.npy    ndarray[T, 768] (wav2vec2 last hidden, 25 fps)
    <root>/smirk/<stem>.pt     dict{pose_params (T,3), cam (T,3), ...}
    <root>/gaze/<stem>.npy     ndarray[T, 2]   (yaw, pitch radians)

and adds the emotion label parsed from MEAD-style stems:

    M003_happy_level_1_001   -> speaker=M003 emotion=happy level=1 seq=001

Video and audio sequence numbers may differ (the robot videos were generated
from MEAD audio with independent numbering), so audio files that do not share
the video stem are matched by (speaker, emotion, level) prefix group: both
groups are sorted by sequence number and paired positionally.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .emotion_modules import EMOTIONS, emotion_to_vector, normalize_emotion_name


def parse_mead_stem(stem: str) -> dict | None:
    """Parse a MEAD-style stem into speaker/emotion/prefix/seq.

    Returns None when no known emotion token is present.
    """
    parts = str(stem).split("_")
    emo_idx = None
    emotion = None
    for i, part in enumerate(parts):
        normalized = normalize_emotion_name(part)
        if normalized is not None and normalized != "neutral":
            emo_idx = i
            emotion = normalized
            break
    if emo_idx is None:
        return None
    speaker = parts[0] if parts else ""
    if emo_idx + 2 < len(parts) and parts[emo_idx + 1] == "level":
        prefix = "_".join(parts[: emo_idx + 3])          # speaker_emotion_level_N
        seq = parts[-1] if len(parts) > emo_idx + 3 else ""
    else:
        prefix = "_".join(parts[: emo_idx + 1])          # speaker_emotion
        seq = parts[-1] if len(parts) > emo_idx + 1 else ""
    return {"speaker": speaker, "emotion": emotion, "prefix": prefix, "seq": seq}


def _seq_key(seq: str):
    return (0, int(seq)) if str(seq).isdigit() else (1, str(seq))


def build_prefix_pairing(motion_stems: list[str], audio_stems: list[str]) -> tuple[dict, list[str]]:
    """Pair motion stems with audio stems inside (speaker, emotion, level) groups.

    Exact-stem matches win first; the remainder are paired positionally after
    sorting each prefix group by sequence number.  Returns (mapping, warnings).
    """
    warnings: list[str] = []
    audio_set = set(audio_stems)
    mapping: dict[str, str] = {}

    leftover_motion = []
    for stem in motion_stems:
        if stem in audio_set:
            mapping[stem] = stem
        else:
            leftover_motion.append(stem)

    used_audio = set(mapping.values())
    motion_groups: dict[str, list[tuple[str, str]]] = {}
    audio_groups: dict[str, list[tuple[str, str]]] = {}
    for stem in leftover_motion:
        info = parse_mead_stem(stem)
        if info is not None:
            motion_groups.setdefault(info["prefix"], []).append((info["seq"], stem))
    for stem in audio_stems:
        if stem in used_audio:
            continue
        info = parse_mead_stem(stem)
        if info is not None:
            audio_groups.setdefault(info["prefix"], []).append((info["seq"], stem))

    for prefix, motions in sorted(motion_groups.items()):
        audios = audio_groups.get(prefix, [])
        motions = sorted(motions, key=lambda item: _seq_key(item[0]))
        audios = sorted(audios, key=lambda item: _seq_key(item[0]))
        if len(motions) != len(audios):
            warnings.append(
                f"prefix {prefix}: {len(motions)} motion vs {len(audios)} audio files; "
                f"pairing the first {min(len(motions), len(audios))} after sorting"
            )
        for (_, m_stem), (_, a_stem) in zip(motions, audios):
            mapping[m_stem] = a_stem
    return mapping, warnings


class EmotionAudioMotionDataset(Dataset):
    """AudioMotionSmirkGazeDataset + emotion labels.

    conditioning_mode:
      "live"  -> gaze/pose/cam are returned as zeros, matching the live
                 PersonaPlex path (FMGenerator with no_pose_cam=True produces
                 the same exact-zero conditioning; the values are ignored).
      "full"  -> real smirk pose/cam and gaze values (original IMTalker
                 training recipe).
    """

    def __init__(
        self,
        opt,
        stems: list[str] | None = None,
        conditioning_mode: str = "live",
        emotions: list[str] | None = None,
        require_smirk_gaze: bool = True,
    ):
        super().__init__()
        self.opt = opt
        self.emotions = list(emotions or EMOTIONS)
        self.conditioning_mode = str(conditioning_mode)
        self.num_frames_for_clip = int(float(opt.wav2vec_sec) * float(opt.fps))
        self.num_prev_frames = int(opt.num_prev_frames)
        self.required_len = self.num_frames_for_clip + self.num_prev_frames

        root = Path(opt.dataset_path)
        self.motion_dir = root / "motion"
        self.audio_dir = root / "audio"
        self.smirk_dir = root / "smirk"
        self.gaze_dir = root / "gaze"

        if stems is None:
            stems = sorted(p.stem for p in self.motion_dir.glob("*.pt"))
        audio_stems = sorted(p.stem for p in self.audio_dir.glob("*.npy")) if self.audio_dir.is_dir() else []
        pairing, pair_warnings = build_prefix_pairing(list(stems), audio_stems)
        for warning in pair_warnings:
            print(f"[dataset][pairing] {warning}")

        self.samples = []
        skipped = {"emotion": 0, "audio": 0, "smirk": 0, "gaze": 0, "short": 0, "error": 0}
        for stem in stems:
            info = parse_mead_stem(stem)
            if info is None or info["emotion"] not in self.emotions:
                skipped["emotion"] += 1
                continue
            audio_stem = pairing.get(stem)
            if audio_stem is None:
                skipped["audio"] += 1
                continue
            motion_path = self.motion_dir / f"{stem}.pt"
            audio_path = self.audio_dir / f"{audio_stem}.npy"
            smirk_path = self.smirk_dir / f"{stem}.pt"
            gaze_path = self.gaze_dir / f"{stem}.npy"
            if require_smirk_gaze and not smirk_path.exists():
                skipped["smirk"] += 1
                continue
            if require_smirk_gaze and not gaze_path.exists():
                skipped["gaze"] += 1
                continue
            try:
                motion = torch.load(motion_path, map_location="cpu")
                audio = np.load(audio_path, mmap_mode="r")
                min_len = min(int(motion.shape[0]), int(audio.shape[0]))
                if require_smirk_gaze:
                    smirk = torch.load(smirk_path, map_location="cpu")
                    gaze = np.load(gaze_path, mmap_mode="r")
                    min_len = min(min_len, int(smirk["pose_params"].shape[0]), int(gaze.shape[0]))
                if min_len < self.required_len:
                    skipped["short"] += 1
                    continue
            except Exception as exc:
                print(f"[dataset][warn] {stem}: {exc}")
                skipped["error"] += 1
                continue
            self.samples.append(
                {
                    "stem": stem,
                    "audio_stem": audio_stem,
                    "emotion": info["emotion"],
                    "motion_path": str(motion_path),
                    "audio_path": str(audio_path),
                    "smirk_path": str(smirk_path) if smirk_path.exists() else "",
                    "gaze_path": str(gaze_path) if gaze_path.exists() else "",
                }
            )
        if not self.samples:
            raise RuntimeError(f"no valid samples under {root} (skipped={skipped})")
        counts = {e: 0 for e in self.emotions}
        for sample in self.samples:
            counts[sample["emotion"]] += 1
        print(f"[dataset] {len(self.samples)} samples | per-emotion={counts} | skipped={skipped}")

    def __len__(self):
        return len(self.samples)
    def _load_clip(self, index: int) -> dict:
        item = self.samples[index]
        motion = torch.load(item["motion_path"], map_location="cpu").float()
        audio = torch.from_numpy(np.load(item["audio_path"], mmap_mode="r").copy()).float()

        min_len = min(int(motion.shape[0]), int(audio.shape[0]))
        if self.conditioning_mode == "full" and item["smirk_path"] and item["gaze_path"]:
            smirk = torch.load(item["smirk_path"], map_location="cpu")
            pose = smirk["pose_params"].float()
            cam = smirk["cam"].float()
            gaze = torch.from_numpy(np.load(item["gaze_path"], mmap_mode="r").copy()).float()
            min_len = min(min_len, int(pose.shape[0]), int(gaze.shape[0]))
        else:
            pose = torch.zeros(min_len, 3)
            cam = torch.zeros(min_len, 3)
            gaze = torch.zeros(min_len, 2)

        start_idx = random.randint(0, min_len - self.required_len)
        end_idx = start_idx + self.required_len

        motion_seg = motion[start_idx:end_idx]
        audio_seg = audio[start_idx:end_idx]
        pose_seg = pose[start_idx:end_idx]
        cam_seg = cam[start_idx:end_idx]
        gaze_seg = gaze[start_idx:end_idx]

        prev = self.num_prev_frames
        ref_idx = random.randint(0, int(motion_seg.shape[0]) - 1)

        return {
            "m_now": motion_seg[prev:],
            "a_now": audio_seg[prev:],
            "gaze_now": gaze_seg[prev:],
            "pose_now": pose_seg[prev:],
            "cam_now": cam_seg[prev:],
            "m_prev": motion_seg[:prev],
            "a_prev": audio_seg[:prev],
            "gaze_prev": gaze_seg[:prev],
            "pose_prev": pose_seg[:prev],
            "cam_prev": cam_seg[:prev],
            "m_ref": motion_seg[ref_idx],
            "emotion": emotion_to_vector(item["emotion"], emotions=self.emotions),
            "emotion_name": item["emotion"],
            "stem": item["stem"],
        }

    def __getitem__(self, index: int) -> dict:
        try:
            return self._load_clip(index)
        except Exception as exc:
            print(f"[dataset][error] index {index}: {exc} -- resampling")
            return self.__getitem__(random.randint(0, len(self) - 1))


def collate_emotion_batch(batch: list[dict]) -> dict:
    """Default-collate compatible; keeps string fields as lists."""
    out = {}
    tensor_keys = [
        "m_now", "a_now", "gaze_now", "pose_now", "cam_now",
        "m_prev", "a_prev", "gaze_prev", "pose_prev", "cam_prev",
        "m_ref", "emotion",
    ]
    for key in tensor_keys:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    out["emotion_name"] = [item["emotion_name"] for item in batch]
    out["stem"] = [item["stem"] for item in batch]
    return out


def stratified_split(samples: list[dict], val_fraction: float, seed: int = 42) -> tuple[list[str], list[str]]:
    """Per-emotion stratified train/val split of dataset sample dicts."""
    rng = random.Random(seed)
    by_emotion: dict[str, list[str]] = {}
    for sample in samples:
        by_emotion.setdefault(sample["emotion"], []).append(sample["stem"])
    train_stems: list[str] = []
    val_stems: list[str] = []
    for emotion, stems in sorted(by_emotion.items()):
        stems = sorted(stems)
        rng.shuffle(stems)
        n_val = max(1, int(round(len(stems) * float(val_fraction)))) if len(stems) > 1 else 0
        val_stems.extend(stems[:n_val])
        train_stems.extend(stems[n_val:])
    return sorted(train_stems), sorted(val_stems)