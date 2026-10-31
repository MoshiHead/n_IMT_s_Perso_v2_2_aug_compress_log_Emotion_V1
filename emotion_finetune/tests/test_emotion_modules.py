"""Local CPU tests for the emotion modules (no GPU, no checkpoints needed)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "IMTalker")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from emotion_finetune.emotion_dataset import build_prefix_pairing, parse_mead_stem
from emotion_finetune.emotion_modules import (
    EMOTIONS,
    EmotionLoRALinear,
    EmotionProjection,
    emotion_to_vector,
    normalize_emotion_name,
)


class TestEmotionVocabulary(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(normalize_emotion_name("happy"), "happy")
        self.assertEqual(normalize_emotion_name("surprised"), "surprise")
        self.assertEqual(normalize_emotion_name("SAD"), "sad")
        self.assertEqual(normalize_emotion_name("neutral"), "neutral")
        self.assertIsNone(normalize_emotion_name("contempt"))

    def test_vector(self):
        vec = emotion_to_vector("happy")
        self.assertEqual(vec.shape, (5,))
        self.assertEqual(vec[0].item(), 1.0)
        self.assertEqual(vec.sum().item(), 1.0)
        neutral = emotion_to_vector("neutral")
        self.assertEqual(neutral.abs().sum().item(), 0.0)
        scaled = emotion_to_vector("fear", scale=0.5)
        self.assertAlmostEqual(scaled[EMOTIONS.index("fear")].item(), 0.5)


class TestMeadStemParsing(unittest.TestCase):
    def test_parse(self):
        info = parse_mead_stem("M003_happy_level_1_001")
        self.assertEqual(info["speaker"], "M003")
        self.assertEqual(info["emotion"], "happy")
        self.assertEqual(info["prefix"], "M003_happy_level_1")
        self.assertEqual(info["seq"], "001")
        info2 = parse_mead_stem("M009_fear_level_1_016")
        self.assertEqual(info2["emotion"], "fear")
        info3 = parse_mead_stem("M011_surprised_level_3_030")
        self.assertEqual(info3["emotion"], "surprise")
        self.assertIsNone(parse_mead_stem("random_file_name"))

    def test_prefix_pairing_mismatched_sequences(self):
        motion = ["M003_happy_level_1_001", "M003_happy_level_1_002", "M003_sad_level_2_010"]
        audio = ["M003_happy_level_1_030", "M003_happy_level_1_031", "M003_sad_level_2_027"]
        mapping, warnings = build_prefix_pairing(motion, audio)
        self.assertEqual(len(mapping), 3)
        self.assertEqual(mapping["M003_happy_level_1_001"], "M003_happy_level_1_030")
        self.assertEqual(mapping["M003_happy_level_1_002"], "M003_happy_level_1_031")
        self.assertEqual(mapping["M003_sad_level_2_010"], "M003_sad_level_2_027")

    def test_prefix_pairing_exact_wins(self):
        motion = ["M003_happy_level_1_001"]
        audio = ["M003_happy_level_1_001", "M003_happy_level_1_030"]
        mapping, _ = build_prefix_pairing(motion, audio)
        self.assertEqual(mapping["M003_happy_level_1_001"], "M003_happy_level_1_001")

    def test_prefix_pairing_count_mismatch(self):
        motion = ["M003_happy_level_1_001", "M003_happy_level_1_002"]
        audio = ["M003_happy_level_1_030"]
        mapping, warnings = build_prefix_pairing(motion, audio)
        self.assertEqual(len(mapping), 1)
        self.assertTrue(warnings)


class TestEmotionProjection(unittest.TestCase):
    def test_zero_init_noop(self):
        proj = EmotionProjection(5, 32)
        out = proj(emotion_to_vector("happy"))
        self.assertEqual(out.abs().max().item(), 0.0)

    def test_bias_free_neutral_forever(self):
        proj = EmotionProjection(5, 32)
        with torch.no_grad():
            proj.proj.weight.normal_(0, 1.0)  # simulate trained weights
        neutral = proj(emotion_to_vector("neutral"))
        self.assertEqual(neutral.abs().max().item(), 0.0)
        happy = proj(emotion_to_vector("happy"))
        self.assertGreater(happy.abs().max().item(), 0.0)


class TestEmotionLoRALinear(unittest.TestCase):
    def test_zero_init_identity(self):
        base = nn.Linear(64, 64)
        wrapped = EmotionLoRALinear(base, rank=16, alpha=32.0)
        x = torch.randn(4, 64)
        self.assertTrue(torch.equal(wrapped(x), base(x)))

    def test_delta_toggle(self):
        base = nn.Linear(64, 64)
        wrapped = EmotionLoRALinear(base, rank=16, alpha=32.0)
        with torch.no_grad():
            wrapped.emotion_lora_B.weight.normal_(0, 0.1)
        x = torch.randn(4, 64)
        self.assertFalse(torch.allclose(wrapped(x), base(x)))
        wrapped.delta_enabled = False
        self.assertTrue(torch.equal(wrapped(x), base(x)))

    def test_wraps_lora_linear(self):
        from generator.train_lora import LoRALinear

        base = LoRALinear(nn.Linear(64, 64), rank=8, alpha=16)
        wrapped = EmotionLoRALinear(base, rank=16, alpha=32.0)
        x = torch.randn(4, 64)
        self.assertTrue(torch.equal(wrapped(x), base(x)))


if __name__ == "__main__":
    unittest.main(verbosity=2)