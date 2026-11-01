"""Emotion conditioning modules for the IMTalker FM generator.

Design summary
--------------
The IMTalker FlowMatchingTransformer (FMT) conditions every block through

    c = t_embedder(t) + c_embedder(a + pose + cam + gaze)

where a/pose/cam/gaze are per-frame dim_c (32) conditioning terms.  In the
live PersonaPlex pipeline pose/cam/gaze are exact zeros, so the live
conditioning is audio-only.  This module adds the emotion as a peer of the
timestep embedding, in the FMT's hidden (dim_h) conditioning space:

    c = t_embedder(t) + c_embedder(a + pose + cam + gaze) + emotion

with

    emotion = emotion_projection(one_hot_5)      # Linear(5 -> 512, bias=False)

Injecting into the hidden space (the same mechanism DiT uses for class
labels) gives the emotion a direct, full-strength pathway into every block's
adaLN modulation, instead of being compressed through the frozen 32-dim
c_embedder bottleneck.

Properties (all verified by emotion_finetune/sanity_check.py):

1. emotion_projection is zero-initialized and has NO bias, so an all-zero
   emotion vector ("neutral") contributes exactly zero to the conditioning
   forever -- even after training.
2. The emotion term is injected through a patched FMT.forward that is a
   verbatim copy of the original with one addition, so with
   current_emotion_cond=None the computation is bit-identical to the
   unmodified model.
3. Emotion LoRA adapters (EmotionLoRALinear) wrap the frozen base (+ optional
   frozen with-audio LoRA) with a second, separately-namespaced delta
   (emotion_lora_A/emotion_lora_B).  Zero-initialized B means the upgraded
   model is bit-identical to the live stack before training.
4. The emotion term is present in BOTH classifier-free-guidance branches
   (forward_with_cfg passes it through unchanged), so existing audio-CFG
   semantics are preserved.
"""

from __future__ import annotations

import re
import types
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Emotion vocabulary
# ---------------------------------------------------------------------------

EMOTIONS = ["happy", "sad", "angry", "surprise", "fear"]
NEUTRAL = "neutral"

# Filename / label aliases mapped onto the canonical EMOTIONS list.  MEAD
# uses "surprised"; the project convention is "surprise".
EMOTION_ALIASES = {
    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",
    "sad": "sad",
    "sadness": "sad",
    "angry": "angry",
    "anger": "angry",
    "surprise": "surprise",
    "surprised": "surprise",
    "fear": "fear",
    "fearful": "fear",
    "afraid": "fear",
}


def normalize_emotion_name(name) -> str | None:
    """Map a raw label token to a canonical emotion name, or None if unknown."""
    if name is None:
        return None
    token = str(name).strip().lower()
    if token in ("", "none", NEUTRAL):
        return NEUTRAL
    return EMOTION_ALIASES.get(token)


def emotion_to_vector(emotion, scale: float = 1.0, emotions: list[str] | None = None) -> torch.Tensor:
    """One-hot (or soft) emotion vector of shape [len(emotions)].

    "neutral" (or None) returns the all-zero vector, which is the exact
    no-conditioning signal by construction.
    """
    emotions = list(emotions or EMOTIONS)
    vec = torch.zeros(len(emotions), dtype=torch.float32)
    if isinstance(emotion, torch.Tensor):
        if emotion.shape[-1] != len(emotions):
            raise ValueError(f"emotion tensor dim {emotion.shape[-1]} != {len(emotions)}")
        return emotion.float() * float(scale)
    name = normalize_emotion_name(emotion)
    if name in (None, NEUTRAL):
        return vec
    if name not in emotions:
        raise ValueError(f"unknown emotion {emotion!r}; expected one of {emotions} or neutral")
    vec[emotions.index(name)] = float(scale)
    return vec


# ---------------------------------------------------------------------------
# Emotion projection (zero-init, bias-free => neutral is exactly zero forever)
# ---------------------------------------------------------------------------

class EmotionProjection(nn.Module):
    """Linear(emotion_dim -> dim_h, bias=False) with zero-initialized weight.

    The emotion is injected into the FMT's hidden (dim_h) conditioning space,
    added alongside the timestep embedding -- the same mechanism DiT uses for
    class labels.  This gives the emotion a direct, full-strength pathway into
    every block's adaLN modulation instead of being compressed through the
    frozen 32-dim c_embedder bottleneck (which was only trained to map
    audio/pose/cam/gaze and attenuates the unfamiliar emotion direction).

    bias=False is deliberate: with an all-zero input the output is exactly
    zero regardless of the trained weight, so "neutral" can never drift away
    from the original model through this pathway.
    """

    def __init__(self, emotion_dim: int = 5, dim_h: int = 512):
        super().__init__()
        self.emotion_dim = int(emotion_dim)
        self.dim_h = int(dim_h)
        self.proj = nn.Linear(self.emotion_dim, self.dim_h, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, emotion: torch.Tensor) -> torch.Tensor:
        return self.proj(emotion)


# ---------------------------------------------------------------------------
# Emotion LoRA (stacks on top of frozen Linear or frozen LoRALinear)
# ---------------------------------------------------------------------------

def _module_io_features(module: nn.Module) -> tuple[int, int]:
    """(in_features, out_features) for nn.Linear or any wrapper exposing .base."""
    if hasattr(module, "in_features") and hasattr(module, "out_features"):
        return int(module.in_features), int(module.out_features)
    base = getattr(module, "base", None)
    if base is not None:
        return _module_io_features(base)
    raise TypeError(f"cannot determine in/out features of {type(module).__name__}")


class EmotionLoRALinear(nn.Module):
    """Second LoRA delta stacked on an already-frozen module.

    The wrapped module may be a plain nn.Linear or the repo's LoRALinear
    (generator/train_lora.py).  The delta uses its own key namespace
    (emotion_lora_A / emotion_lora_B) so it never collides with the existing
    lora_A/lora_B keys of the with-audio LoRA.  B is zero-initialized, so the
    wrapper is an exact identity at initialization.

    delta_enabled=False bypasses the delta (used by the neutral-anchor loss
    to recover the frozen reference function without a second model copy).
    """

    def __init__(self, base: nn.Module, rank: int = 16, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        in_features, out_features = _module_io_features(base)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / max(1, self.rank)
        self.delta_enabled = True
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.emotion_lora_A = nn.Linear(in_features, self.rank, bias=False)
        self.emotion_lora_B = nn.Linear(self.rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.emotion_lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.emotion_lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.delta_enabled:
            out = out + self.emotion_lora_B(self.emotion_lora_A(self.dropout(x))) * self.scale
        return out

# ---------------------------------------------------------------------------
# Emotion LoRA target selection (mirrors the proven with-audio LoRA tiers)
# ---------------------------------------------------------------------------

_BLOCK_TARGET_RE = re.compile(
    r"^fmt\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:fc1|fc2)|adaLN_modulation\.1)$"
)
_EXTRA_TARGETS = {"fmt.decoder.linear", "fmt.decoder.adaLN_modulation.1"}


def emotion_lora_target_names(model: nn.Module) -> list[str]:
    """Module paths that receive an emotion LoRA wrapper.

    Same target set as the proven with-audio LoRA (FMT attention qkv/proj,
    MLP fc1/fc2, adaLN modulation, decoder) minus the audio/pose/gaze/cam
    projections: the audio projection already carries the with-audio LoRA
    (lip-sync) and the emotion pathway is the adaLN conditioning, not the
    audio frontend.
    """
    names = []
    for name, _module in model.named_modules():
        if _BLOCK_TARGET_RE.match(name) or name in _EXTRA_TARGETS:
            names.append(name)
    return names


def _get_child(root: nn.Module, path: str) -> nn.Module:
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _set_child(root: nn.Module, path: str, value: nn.Module) -> None:
    parts = path.split(".")
    parent = _get_child(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    key = parts[-1]
    if key.isdigit():
        parent[int(key)] = value
    else:
        setattr(parent, key, value)


def apply_emotion_lora_to_model(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.0,
) -> list[str]:
    """Wrap every emotion-LoRA target with EmotionLoRALinear (frozen base)."""
    targets = emotion_lora_target_names(model)
    if not targets:
        raise RuntimeError("no emotion LoRA targets found -- is this an FMGenerator?")
    for name in targets:
        base = _get_child(model, name)
        _set_child(model, name, EmotionLoRALinear(base, rank=rank, alpha=alpha, dropout=dropout))
    return targets


# ---------------------------------------------------------------------------
# Patched FMT forward (verbatim copy of FlowMatchingTransformer.forward from
# IMTalker/generator/FMT.py plus the emotion conditioning term)
# ---------------------------------------------------------------------------

def emotion_fmt_forward(
    self,
    t,
    x,
    a,
    prev_x,
    prev_a,
    ref_x,
    gaze,
    prev_gaze,
    pose,
    prev_pose,
    cam,
    prev_cam,
    train: bool = True,
    **kwargs,
) -> torch.Tensor:
    """FlowMatchingTransformer.forward + additive emotion conditioning.

    The body is a verbatim copy of the original forward; the only change is
    the `emotion_cond` addition before c_embedder.  With
    self.current_emotion_cond=None the computation is bit-identical to the
    unmodified model.
    """
    t = self.t_embedder(t).unsqueeze(1)
    a = self.sequence_embedder(a, dropout_prob=self.opt.audio_dropout_prob, train=train)
    pose = self.sequence_embedder(pose, dropout_prob=self.opt.audio_dropout_prob, train=train)
    cam = self.sequence_embedder(cam, dropout_prob=self.opt.audio_dropout_prob, train=train)
    gaze = self.sequence_embedder(gaze, dropout_prob=self.opt.audio_dropout_prob, train=train)

    if prev_x is not None:
        prev_x = self.sequence_embedder(prev_x, dropout_prob=0.5, train=train)
        prev_a = self.sequence_embedder(prev_a, dropout_prob=0.5, train=train)
        prev_pose = self.sequence_embedder(prev_pose, dropout_prob=0.5, train=train)
        prev_cam = self.sequence_embedder(prev_cam, dropout_prob=0.5, train=train)
        prev_gaze = self.sequence_embedder(prev_gaze, dropout_prob=0.5, train=train)

        x = torch.cat([prev_x, x], dim=1)
        a = torch.cat([prev_a, a], dim=1)
        pose = torch.cat([prev_pose, pose], dim=1)
        cam = torch.cat([prev_cam, cam], dim=1)
        gaze = torch.cat([prev_gaze, gaze], dim=1)

    ref_x = ref_x[:, None, ...].repeat(1, x.shape[1], 1)
    x = torch.cat([ref_x, x], dim=-1)
    x = self.x_embedder(x)

    # Calculate RoPE
    rotary_pos_emb = self.rotary_emb(x, seq_len=x.shape[1])

    c = self.c_embedder(a + pose + cam + gaze)
    c = t + c
    # Emotion is injected into the hidden (dim_h) conditioning space, as a
    # peer of the timestep embedding, NOT through the frozen 32-dim c_embedder.
    emotion_cond = getattr(self, "current_emotion_cond", None)
    if emotion_cond is not None:
        c = c + emotion_cond

    for block in self.blocks:
        x = block(x, c, rotary_pos_emb=rotary_pos_emb)

    return self.decoder(x, c)


# ---------------------------------------------------------------------------
# Emotion state helpers (runtime attribute on the FMT instance)
# ---------------------------------------------------------------------------

def _require_upgraded(fm: nn.Module) -> None:
    if not getattr(fm, "_emotion_upgraded", False):
        raise RuntimeError("FMGenerator has not been upgraded; call upgrade_fm_with_emotion() first")


def emotion_forward_with_cfg(
    self,
    t,
    x,
    a,
    prev_x,
    prev_a,
    ref_x,
    gaze,
    prev_gaze,
    pose,
    prev_pose,
    cam,
    prev_cam,
    a_cfg_scale: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """CFG with a dedicated, output-space emotion guidance term.

    The stock ``forward_with_cfg`` broadcasts a single emotion vector to BOTH
    guidance branches, which (a) puts the emotion into the null-audio
    unconditional branch the model never saw in training, and (b) forces any
    amplification to scale the *input* embedding, which quickly goes
    out-of-distribution (super-linear distortion).

    Here every evaluation stays in-distribution and the emotion is amplified in
    *velocity* space instead:

        v = v00 + a_cfg * (v10 - v00) + e_cfg * (v11 - v10)

    with  v00 = FMT(null_audio, neutral), v10 = FMT(audio, neutral) and
    v11 = FMT(audio, emotion).  ``e_cfg`` (``_emotion_cfg_scale``) is therefore
    an exactly-linear, always-in-distribution emotion strength: 1.0 reproduces
    the trained emotion, larger values strengthen it cleanly.  With no active
    emotion this reduces to the stock path (bit-identical).
    """
    original = getattr(self, "_original_forward_with_cfg", None)
    emotion_cond = getattr(self, "current_emotion_cond", None)
    e_cfg = float(getattr(self, "_emotion_cfg_scale", 1.0))
    if original is None or emotion_cond is None:
        return original(
            t=t, x=x, a=a, prev_x=prev_x, prev_a=prev_a, ref_x=ref_x,
            gaze=gaze, prev_gaze=prev_gaze, pose=pose, prev_pose=prev_pose,
            cam=cam, prev_cam=prev_cam, a_cfg_scale=a_cfg_scale, **kwargs,
        )

    saved = self.current_emotion_cond
    try:
        # neutral reference velocity v10 = FMT(audio, neutral)
        self.current_emotion_cond = None
        v10 = self.forward(
            t=t, x=x, a=a, prev_x=prev_x, prev_a=prev_a, ref_x=ref_x,
            gaze=gaze, prev_gaze=prev_gaze, pose=pose, prev_pose=prev_pose,
            cam=cam, prev_cam=prev_cam, train=False,
        )
        # emotional velocity v11 = FMT(audio, emotion)
        self.current_emotion_cond = saved
        v11 = self.forward(
            t=t, x=x, a=a, prev_x=prev_x, prev_a=prev_a, ref_x=ref_x,
            gaze=gaze, prev_gaze=prev_gaze, pose=pose, prev_pose=prev_pose,
            cam=cam, prev_cam=prev_cam, train=False,
        )
        if a_cfg_scale != 1.0:
            # audio CFG base v00 + a_cfg*(v10 - v00); v00 = FMT(null_audio, neutral)
            self.current_emotion_cond = None
            null_a = torch.zeros_like(a)
            v00 = self.forward(
                t=t, x=x, a=null_a, prev_x=prev_x, prev_a=prev_a, ref_x=ref_x,
                gaze=gaze, prev_gaze=prev_gaze, pose=pose, prev_pose=prev_pose,
                cam=cam, prev_cam=prev_cam, train=False,
            )
            base = v00 + a_cfg_scale * (v10 - v00)
        else:
            base = v10
    finally:
        self.current_emotion_cond = saved

    return base + e_cfg * (v11 - v10)


def set_emotion_guidance(fm: nn.Module, scale: float) -> None:
    """Set the output-space emotion guidance strength (e_cfg).

    1.0 reproduces the trained emotion; >1 amplifies it in velocity space
    (clean, in-distribution) instead of scaling the input embedding.
    """
    _require_upgraded(fm)
    fm.fmt._emotion_cfg_scale = float(scale)


def set_emotion(fm: nn.Module, emotion, scale: float = 1.0) -> torch.Tensor:
    """Project an emotion (name or [5]/[B,5] tensor) and store it on the FMT.

    The projection runs WITH autograd enabled so the emotion_projection
    weights receive gradients during training.  The stored tensor has shape
    [B, 1, dim_c] and broadcasts over the time axis inside the FMT.
    """
    _require_upgraded(fm)
    vec = emotion_to_vector(emotion, scale=scale, emotions=list(getattr(fm, "_emotion_names", EMOTIONS)))
    weight = fm.emotion_projection.proj.weight
    vec = vec.to(device=weight.device, dtype=weight.dtype)
    cond = fm.emotion_projection(vec)
    # emotion_cond_scale strengthens the (otherwise sub-visible) emotion signal
    # consistently between training and inference; stored in the adapter config.
    cond_scale = float(getattr(fm, "_emotion_cond_scale", 1.0))
    if cond_scale != 1.0:
        cond = cond * cond_scale
    if cond.ndim == 1:
        cond = cond.view(1, 1, -1)
    else:
        cond = cond.unsqueeze(1)
    fm.fmt.current_emotion_cond = cond
    return cond


def clear_emotion(fm: nn.Module) -> None:
    """Remove the emotion conditioning (exact original behavior)."""
    _require_upgraded(fm)
    fm.fmt.current_emotion_cond = None
    fm.fmt._emotion_cfg_scale = 1.0


def set_emotion_lora_enabled(fm: nn.Module, enabled: bool) -> int:
    """Toggle every EmotionLoRALinear delta (neutral-anchor reference path)."""
    count = 0
    for module in fm.modules():
        if isinstance(module, EmotionLoRALinear):
            module.delta_enabled = bool(enabled)
            count += 1
    return count

# ---------------------------------------------------------------------------
# State-dict helpers / adapter save-load
# ---------------------------------------------------------------------------

def emotion_lora_state_dict(fm: nn.Module) -> dict:
    """Full state-dict keys of the emotion LoRA deltas only."""
    return {
        key: value.detach().cpu()
        for key, value in fm.state_dict().items()
        if ".emotion_lora_A." in key or ".emotion_lora_B." in key
    }


def emotion_trainable_parameters(fm: nn.Module) -> list[nn.Parameter]:
    """All parameters owned by the emotion modules (projection + LoRA)."""
    params = list(fm.emotion_projection.parameters())
    for module in fm.modules():
        if isinstance(module, EmotionLoRALinear):
            params.extend(module.emotion_lora_A.parameters())
            params.extend(module.emotion_lora_B.parameters())
    return params


def save_emotion_adapter(
    fm: nn.Module,
    path,
    config_extra: dict | None = None,
    override_params: dict | None = None,
) -> None:
    """Save the portable emotion adapter (projection + LoRA deltas).

    override_params optionally replaces matching entries (used to export EMA
    weights instead of the raw training weights).
    """
    _require_upgraded(fm)
    proj_sd = {k: v.detach().cpu() for k, v in fm.emotion_projection.state_dict().items()}
    lora_sd = emotion_lora_state_dict(fm)
    if override_params:
        for key, value in override_params.items():
            # proj_sd keys are relative ("proj.weight") while EMA keys are
            # full ("emotion_projection.proj.weight") -- strip the prefix.
            if key.startswith("emotion_projection."):
                rel = key[len("emotion_projection."):]
                if rel in proj_sd:
                    proj_sd[rel] = value.detach().cpu()
            if key in lora_sd:
                lora_sd[key] = value.detach().cpu()
    config = {
        "format": "imtalker_emotion_adapter_v1",
        "emotion_dim": int(fm.emotion_projection.emotion_dim),
        "dim_h": int(fm.opt.dim_h),
        "emotions": list(getattr(fm, "_emotion_names", EMOTIONS)),
        "lora_rank": int(getattr(fm, "_emotion_lora_rank", 0)),
        "lora_alpha": float(getattr(fm, "_emotion_lora_alpha", 0.0)),
        "emotion_cond_scale": float(getattr(fm, "_emotion_cond_scale", 1.0)),
        "base": "IMTalker FMGenerator (+ optional frozen with-audio LoRA)",
    }
    if config_extra:
        config.update(config_extra)
    payload = {"config": config, "emotion_projection": proj_sd, "emotion_lora": lora_sd}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def load_emotion_adapter(fm: nn.Module, path) -> dict:
    """Load a saved emotion adapter into an upgraded FMGenerator."""
    _require_upgraded(fm)
    payload = torch.load(str(path), map_location="cpu")
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    proj_sd = payload.get("emotion_projection", {})
    lora_sd = payload.get("emotion_lora", {})
    if not proj_sd and not lora_sd:
        raise RuntimeError(f"{path} does not look like an emotion adapter checkpoint")
    expected_dim = int(config.get("emotion_dim", fm.emotion_projection.emotion_dim))
    if expected_dim != int(fm.emotion_projection.emotion_dim):
        raise RuntimeError(
            f"adapter emotion_dim={expected_dim} but model has {fm.emotion_projection.emotion_dim}"
        )
    expected_dim_h = int(config.get("dim_h", fm.emotion_projection.dim_h))
    if expected_dim_h != int(fm.emotion_projection.dim_h):
        raise RuntimeError(
            f"adapter dim_h={expected_dim_h} but model has {fm.emotion_projection.dim_h} "
            "(old 32-dim adapters are incompatible with the 512-dim injection)"
        )
    model_sd = fm.state_dict()
    loaded, skipped = 0, []
    with torch.no_grad():
        for key, value in proj_sd.items():
            full = f"emotion_projection.{key}"
            if full in model_sd and model_sd[full].shape == value.shape:
                model_sd[full].copy_(value.to(model_sd[full].device, model_sd[full].dtype))
                loaded += 1
            else:
                skipped.append(full)
        for key, value in lora_sd.items():
            if key in model_sd and model_sd[key].shape == value.shape:
                model_sd[key].copy_(value.to(model_sd[key].device, model_sd[key].dtype))
                loaded += 1
            else:
                skipped.append(key)
    if loaded == 0:
        raise RuntimeError(f"no adapter keys matched the model (skipped {len(skipped)}): {path}")
    return {"loaded": loaded, "skipped": skipped, "config": config, "path": str(path)}


# ---------------------------------------------------------------------------
# Checkpoint cleaning (identical semantics to the live server's loader)
# ---------------------------------------------------------------------------

def clean_generator_state(ckpt: dict) -> dict:
    raw = ckpt.get("ema_state_dict") or ckpt.get("state_dict", ckpt.get("model", ckpt))
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]
    return {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# The one-call upgrade
# ---------------------------------------------------------------------------

def upgrade_fm_with_emotion(
    fm: nn.Module,
    emotion_dim: int = 5,
    emotions: list[str] | None = None,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.0,
    adapter_path=None,
    emotion_name: str = NEUTRAL,
    emotion_scale: float = 1.0,
    emotion_cond_scale: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Turn a stock FMGenerator into an emotion-conditioned generator.

    Steps: freeze all existing parameters, patch the FMT forward, add the
    zero-init emotion projection, wrap the emotion LoRA targets, optionally
    load a trained adapter, optionally set a constant emotion.

    With adapter_path=None and emotion_name="neutral" the upgraded model is
    bit-identical to the input model (verified by sanity_check.py).
    """
    if getattr(fm, "_emotion_upgraded", False):
        raise RuntimeError("upgrade_fm_with_emotion() called twice on the same model")
    emotions = list(emotions or EMOTIONS)
    if int(emotion_dim) != len(emotions):
        raise ValueError(f"emotion_dim={emotion_dim} but {len(emotions)} emotion names given")

    for param in fm.parameters():
        param.requires_grad_(False)

    # 1) patch the FMT forward + CFG (instance-level; the class stays untouched)
    fm.fmt._original_forward_with_cfg = fm.fmt.forward_with_cfg
    fm.fmt.forward = types.MethodType(emotion_fmt_forward, fm.fmt)
    fm.fmt.forward_with_cfg = types.MethodType(emotion_forward_with_cfg, fm.fmt)
    fm.fmt.current_emotion_cond = None
    fm.fmt._emotion_cfg_scale = 1.0

    # 2) zero-init emotion projection (trainable)
    fm.emotion_projection = EmotionProjection(emotion_dim, int(fm.opt.dim_h))

    # 3) emotion LoRA on the FMT blocks + decoder (trainable A/B)
    targets = apply_emotion_lora_to_model(fm, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)

    fm._emotion_upgraded = True
    fm._emotion_names = emotions
    fm._emotion_lora_rank = int(lora_rank)
    fm._emotion_lora_alpha = float(lora_alpha)
    fm._emotion_cond_scale = float(emotion_cond_scale)

    # move the freshly created modules onto the model's device
    device = next(fm.parameters()).device
    fm.emotion_projection.to(device)
    for module in fm.modules():
        if isinstance(module, EmotionLoRALinear):
            module.emotion_lora_A.to(device)
            module.emotion_lora_B.to(device)

    # 4) optional trained adapter (its config may override the cond scale)
    load_info = None
    if adapter_path:
        load_info = load_emotion_adapter(fm, adapter_path)
        cfg_scale = (load_info.get("config") or {}).get("emotion_cond_scale")
        if cfg_scale is not None:
            fm._emotion_cond_scale = float(cfg_scale)

    # 5) optional constant emotion
    active = NEUTRAL
    normalized = normalize_emotion_name(emotion_name)
    if normalized not in (None, NEUTRAL):
        set_emotion(fm, normalized, scale=emotion_scale)
        active = normalized

    trainable = sum(p.numel() for p in emotion_trainable_parameters(fm))
    summary = {
        "targets": len(targets),
        "lora_rank": int(lora_rank),
        "lora_alpha": float(lora_alpha),
        "trainable_params": trainable,
        "adapter": load_info,
        "active_emotion": active,
        "emotion_scale": float(emotion_scale),
    }
    if verbose:
        print(
            f"[emotion] upgrade complete: targets={len(targets)} rank={lora_rank} "
            f"alpha={lora_alpha} trainable={trainable:,} active={active} scale={emotion_scale}",
            flush=True,
        )
        if load_info is not None:
            print(
                f"[emotion] adapter loaded: {load_info['path']} "
                f"keys={load_info['loaded']} skipped={len(load_info['skipped'])}",
                flush=True,
            )
    return summary