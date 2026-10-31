"""Emotion-conditioned fine-tuning for the IMTalker FM generator.

Self-contained package: no existing repository file is modified.  The emotion
conditioning enters the FlowMatchingTransformer through a patched instance
forward (see emotion_modules.emotion_fmt_forward), so the stock FMGenerator /
FlowMatchingTransformer classes stay untouched.
"""

from .emotion_modules import (
    EMOTIONS,
    NEUTRAL,
    EmotionLoRALinear,
    EmotionProjection,
    apply_emotion_lora_to_model,
    clear_emotion,
    emotion_fmt_forward,
    emotion_lora_state_dict,
    emotion_to_vector,
    load_emotion_adapter,
    normalize_emotion_name,
    save_emotion_adapter,
    set_emotion,
    set_emotion_lora_enabled,
    upgrade_fm_with_emotion,
)

__all__ = [
    "EMOTIONS",
    "NEUTRAL",
    "EmotionLoRALinear",
    "EmotionProjection",
    "apply_emotion_lora_to_model",
    "clear_emotion",
    "emotion_fmt_forward",
    "emotion_lora_state_dict",
    "emotion_to_vector",
    "load_emotion_adapter",
    "normalize_emotion_name",
    "save_emotion_adapter",
    "set_emotion",
    "set_emotion_lora_enabled",
    "upgrade_fm_with_emotion",
]