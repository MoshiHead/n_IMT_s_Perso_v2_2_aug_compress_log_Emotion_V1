"""Launch the AH AudioPace live server with emotion conditioning.

This wrapper loads the UNMODIFIED AH server module
(IMTalker/liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py),
patches its _load_fm() to apply the emotion upgrade after the base generator
and with-audio LoRA are loaded, and then runs the server's own main().

Configuration is via environment variables (no CLI changes needed):

  EMOTION_ADAPTER_PATH   path to a trained emotion_adapter.ckpt.
                         Empty  -> server behaves exactly like the stock AH server.
  AVATAR_EMOTION         neutral | happy | sad | angry | surprise | fear
                         (constant emotion for the whole conversation)
  EMOTION_SCALE          input-embedding scale; KEEP AT 1.0 (the trained magnitude).
                         >1 drives the model out-of-distribution and distorts the face.
  EMOTION_GUIDANCE       output-space emotion strength (default 1.0). The recommended
                         way to get a stronger expression: exactly linear, always
                         in-distribution, so it stays natural at 1.5-3.
  EMOTION_LORA_RANK      must match the adapter's training rank (default 16)
  EMOTION_LORA_ALPHA     must match the adapter's training alpha (default 32)

All server CLI arguments are forwarded unchanged (the server parses sys.argv
itself), so the launch command is identical to the stock AH launch.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    imtalker_dir = repo_root / "IMTalker"
    for path in (str(imtalker_dir), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    server_path = imtalker_dir / "liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py"
    if not server_path.is_file():
        raise SystemExit(f"AH server not found: {server_path}")

    spec = importlib.util.spec_from_file_location("ah_live_server", server_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ah_live_server"] = module
    spec.loader.exec_module(module)

    original_load_fm = module._load_fm

    def _load_fm_with_emotion(args, device):
        fm = original_load_fm(args, device)
        adapter_path = os.environ.get("EMOTION_ADAPTER_PATH", "").strip()
        emotion_name = os.environ.get("AVATAR_EMOTION", "neutral").strip() or "neutral"
        emotion_scale = float(os.environ.get("EMOTION_SCALE", "1.0") or "1.0")
        emotion_guidance = float(os.environ.get("EMOTION_GUIDANCE", "1.0") or "1.0")
        lora_rank = int(os.environ.get("EMOTION_LORA_RANK", "16") or "16")
        lora_alpha = float(os.environ.get("EMOTION_LORA_ALPHA", "32") or "32")

        if adapter_path:
            from emotion_finetune.emotion_modules import (
                set_emotion_guidance,
                upgrade_fm_with_emotion,
            )

            summary = upgrade_fm_with_emotion(
                fm,
                adapter_path=adapter_path,
                emotion_name=emotion_name,
                emotion_scale=emotion_scale,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )
            # clean output-space amplification (in-distribution, exactly linear)
            if emotion_guidance != 1.0 and summary["active_emotion"] != "neutral":
                set_emotion_guidance(fm, emotion_guidance)
            print(
                f"[emotion] live emotion enabled: adapter={adapter_path} "
                f"avatar_emotion={summary['active_emotion']} scale={emotion_scale} "
                f"guidance={emotion_guidance} targets={summary['targets']} rank={lora_rank}",
                flush=True,
            )
        elif emotion_name.lower() not in ("", "neutral", "none"):
            print(
                f"[emotion] WARNING: AVATAR_EMOTION={emotion_name} but "
                f"EMOTION_ADAPTER_PATH is not set -- running neutral",
                flush=True,
            )
        else:
            print(
                "[emotion] disabled (EMOTION_ADAPTER_PATH unset) -- "
                "behavior identical to the stock AH server",
                flush=True,
            )
        return fm

    module._load_fm = _load_fm_with_emotion
    module.main()


if __name__ == "__main__":
    main()