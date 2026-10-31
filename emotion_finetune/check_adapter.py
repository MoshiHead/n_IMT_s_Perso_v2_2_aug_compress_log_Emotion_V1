"""Quick diagnostic: does a trained emotion adapter contain non-trivial weights?

Prints the per-emotion embedding norms and the emotion LoRA B weight norms.
If these are all near zero, training did not learn (the adapter is a no-op).
If they are clearly non-zero and DISTINCT per emotion, the adapter should
produce visible emotional differences.

Usage:
  python emotion_finetune/check_adapter.py --adapter_path exps/emotion_lora_v1/emotion_adapter_final.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

EMOTIONS = ["happy", "sad", "angry", "surprise", "fear"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    args = parser.parse_args()

    payload = torch.load(args.adapter_path, map_location="cpu")
    config = payload.get("config", {})
    proj = payload.get("emotion_projection", {})
    lora = payload.get("emotion_lora", {})
    emotions = config.get("emotions", EMOTIONS)

    print("=" * 64)
    print(f"adapter: {args.adapter_path}")
    print(f"config : {config}")
    print("=" * 64)

    weight = proj.get("proj.weight")
    if weight is None:
        print("[error] no emotion_projection.proj.weight in adapter")
        return
    print(f"\nemotion embedding (proj.weight {tuple(weight.shape)}):")
    print(f"  dim_h = {weight.shape[0]}  (512 = new hidden-space injection, 32 = old bottleneck)")
    if weight.shape[0] == 32:
        print("  [warn] this is an OLD 32-dim adapter -- retrain with the fixed code!")
    norms = weight.norm(dim=0)
    for i, emotion in enumerate(emotions):
        if i < norms.shape[0]:
            bar = "#" * int(min(50, norms[i].item() * 20))
            print(f"  {emotion:9s} norm={norms[i].item():8.4f} {bar}")
    print(f"  max pairwise embedding distance:")
    import itertools
    max_d = 0.0
    for i, j in itertools.combinations(range(min(len(emotions), weight.shape[1])), 2):
        d = (weight[:, i] - weight[:, j]).norm().item()
        max_d = max(max_d, d)
    print(f"    {max_d:.4f}  (want clearly > 0 for distinct emotions)")

    b_norms = [v.norm().item() for k, v in lora.items() if "emotion_lora_B" in k]
    if b_norms:
        import numpy as np
        print(f"\nemotion LoRA B weights ({len(b_norms)} modules):")
        print(f"  mean norm = {np.mean(b_norms):.5f}  max = {np.max(b_norms):.5f}  min = {np.min(b_norms):.5f}")
        nonzero = sum(1 for n in b_norms if n > 1e-3)
        print(f"  modules with |B| > 1e-3: {nonzero}/{len(b_norms)}")

    print("\n" + "=" * 64)
    emb_ok = weight.shape[0] == 512 and max_d > 0.5
    lora_ok = b_norms and (max(b_norms) > 1e-2)
    if emb_ok and lora_ok:
        print("VERDICT: adapter has non-trivial, distinct emotion weights -- "
              "it should produce visible emotions.")
    elif weight.shape[0] == 32:
        print("VERDICT: OLD 32-dim adapter (weak injection) -- RETRAIN with the fixed code.")
    elif max_d <= 0.5:
        print("VERDICT: emotion embeddings are too similar / too small -- "
              "the model did not learn distinct emotions. Retrain (check margin loss decreases).")
    else:
        print("VERDICT: LoRA deltas are near zero -- emotion pathway undertrained. Retrain.")
    print("=" * 64)


if __name__ == "__main__":
    main()