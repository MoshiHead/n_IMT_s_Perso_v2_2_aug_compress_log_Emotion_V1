# IMTalker Emotion Fine-Tuning

Teach the **existing pretrained IMTalker generator** five emotional facial
expressions (**happy, sad, angry, surprise, fear**) from a custom robot-face
dataset, while preserving the original neutral behavior and audio-driven lip
synchronization. No existing repository file is modified.

## How it works

The IMTalker FlowMatchingTransformer (FMT) conditions every block through

```
c = t_embedder(t) + c_embedder(a + pose + cam + gaze)
```

In the live PersonaPlex pipeline `pose`/`cam`/`gaze` are exact zeros, so the
live conditioning is audio-only. This package adds the emotion as a **peer of
the timestep embedding**, in the FMT's hidden (512-dim) conditioning space:

```
c = t_embedder(t) + c_embedder(a + pose + cam + gaze) + emotion
emotion = EmotionProjection(one_hot_5)        # Linear(5 -> 512, bias=False), zero-init
```

Injecting into the hidden space (the same mechanism DiT uses for class
labels) gives the emotion a direct, full-strength pathway into every block's
adaLN modulation. (An earlier version injected into the 32-dim sum *before*
the frozen `c_embedder`, which attenuated the emotion and let the model
ignore it -- do not use adapters trained with that version.)

* **Zero-init, bias-free emotion projection** - an all-zero (neutral) emotion
  vector contributes exactly zero, forever (even after training). The
  untrained adapter is a no-op.
* **Emotion LoRA (rank 16)** on the FMT attention/MLP/adaLN + decoder, stacked
  on the frozen base + with-audio LoRA with a separate key namespace
  (`emotion_lora_A/B`). Zero-init B => the upgraded model is bit-identical to
  the live stack before training.
* **Neutral-anchor distillation loss** - pins the `emotion=0` slice of the
  model to the frozen original, so neutral behavior is preserved *without*
  neutral training videos.
* **Emotion-margin loss** - permuted emotion labels must fit worse, forcing
  the model to actually use the conditioning.

Only ~1.5M parameters are trained; the base generator, with-audio LoRA,
wav2vec and renderer stay frozen.

## Files

| file | purpose |
|---|---|
| `emotion_modules.py` | emotion projection, `EmotionLoRALinear`, patched FMT forward, upgrade/save/load |
| `emotion_dataset.py` | MEAD-stem parsing, (speaker,emotion,level) prefix pairing, dataset |
| `verify_emotion_dataset.py` | verify a dataset against the training contract |
| `preprocess_emotion_data.py` | raw videos + wavs -> motion/audio (with prefix pairing) |
| `train_emotion.py` | training (single- or multi-GPU, AMP, EMA, resume) |
| `sanity_check.py` | bit-exactness proof (run before training) |
| `preview_emotion.py` | offline 6-way render (neutral + 5 emotions) |
| `evaluate_emotion_model.py` | quantitative evaluation |
| `measure_emotion_effect.py` | diagnostic: emotion effect size on the motion latent vs scale |
| `check_adapter.py` | diagnostic: inspect a trained adapter (norms, distinctiveness) |
| `launch_emotion_live.py` | live-server wrapper (applies the adapter to the stock AH server) |
| `generate_notebooks.py` | regenerates the two RunPod notebooks |
| `tests/` | local CPU tests |

## Quick start (on a GPU pod)

1. **Train** with `RunPod_Emotion_Training.ipynb` (installs deps, downloads
   checkpoints, verifies the dataset, sanity-checks, trains, previews, evaluates).
2. **Run live** with `RunPod_Emotion_Live_Inference.ipynb` - set
   `AVATAR_EMOTION` to `happy`/`sad`/`angry`/`surprise`/`fear` (constant
   expression for the conversation) or `neutral` (original behavior).

## Command-line usage

```bash
# verify a preprocessed dataset
python emotion_finetune/verify_emotion_dataset.py \
    --dataset_path /workspace/emotion_generator_dataset --wav2vec_sec 0.96

# sanity check (bit-exactness)
python emotion_finetune/sanity_check.py \
    --generator_path IMTalker/checkpoints/generator.ckpt \
    --base_lora_path checkpoints/live_winner/lora/ditto_blink_lora_withaudio_r64_096_continue_2h_last.ckpt

# train (single GPU)
python emotion_finetune/train_emotion.py \
    --dataset_path /workspace/emotion_generator_dataset \
    --generator_path IMTalker/checkpoints/generator.ckpt \
    --base_lora_path checkpoints/live_winner/lora/ditto_blink_lora_withaudio_r64_096_continue_2h_last.ckpt \
    --exp_root exps --exp_name emotion_lora_v1

# train (multi GPU)
torchrun --nproc_per_node=2 emotion_finetune/train_emotion.py ... (same args)

# preview + evaluate
python emotion_finetune/preview_emotion.py --ref_path IMTalker/assets/3robert.jpeg \
    --aud_path audio.wav --generator_path ... --renderer_path ... \
    --wav2vec_model_path ... --base_lora_path ... \
    --emotion_adapter_path exps/emotion_lora_v1/emotion_adapter_final.ckpt \
    --out_dir outputs/preview --name robot --crop

# run live (constant emotion)
EMOTION_ADAPTER_PATH=exps/emotion_lora_v1/emotion_adapter_final.ckpt \
AVATAR_EMOTION=happy bash run_live_emotion.sh
```

## Dataset format

The training reads the IMTalker generator format (same as
`generator/dataset.py`), with the emotion label parsed from MEAD-style stems:

```
<dataset>/
  motion/M003_happy_level_1_001.pt    Tensor[T, 32]   (renderer motion latents)
  audio/M003_happy_level_1_001.npy    ndarray[T, 768] (wav2vec2 last hidden, 25 fps)
  smirk/M003_happy_level_1_001.pt     pose_params (T,3), cam (T,3)
  gaze/M003_happy_level_1_001.npy     ndarray[T, 2]
```

Video and audio sequence numbers may differ (`..._001.mp4` vs `..._030.wav`);
they are matched by `(speaker, emotion, level)` prefix. Minimum clip length is
`wav2vec_sec*fps + num_prev_frames` = 34 frames for the 0.96 s live contract.

## Neutral preservation

Neutral is preserved by construction **and** by training:

1. `emotion=0` -> zero contribution (bias-free projection), verified
   bit-exact by `sanity_check.py`.
2. The anchor loss keeps the `emotion=0` slice pinned to the frozen original
   during training (watch `val_anchor` stay near 0).
3. `evaluate_emotion_model.py` measures the residual neutral drift after
   training (relative L2 should be near 0).

## Emotion strength (cond scale)

The emotion embedding enters the 512-dim conditioning as an additive term.
Depending on the data, the magnitude the model settles on can be **too small to
be visible** on the rendered face even though training metrics look correct
(`margin` -> 0, distinct adapter weights). Two levers control the strength:

- `--emotion_cond_scale` (training, default 1.0): multiplies the emotion
  embedding during training and is **baked into the adapter config**, so
  inference applies it automatically. Use e.g. `4` so the model learns a
  natively stronger signal. Changing it changes the effective conditioning, so
  **start a fresh `--exp_name`** rather than resuming a checkpoint trained with
  a different value.
- `--emotion_emb_lr_mult` (training, default 1.0): trains the small emotion
  embedding with a higher LR than the LoRA (e.g. `10`), so it reaches a usable
  magnitude faster.
- `--emotion_scale` / `EMOTION_SCALE` (inference, default 1.0): an extra
  on-the-fly multiplier, no retrain needed. Good for a quick check of whether
  an already-trained adapter learned the right direction.

## Troubleshooting: emotions visible but too subtle

This is the expected outcome when the training data's expressions are mild
(e.g. MEAD `level_1`). The model learns the data's intensity faithfully, and the
flow-matching loss **cannot reward stronger-than-data expressions** - so no
training setting will push past it. The fix is to **amplify the learned emotion
direction at inference**. Because the emotion enters the conditioning linearly
(`c += emotion_projection(one_hot)`, and adaLN is linear in `c`), scaling it
extrapolates the expression naturally (equivalent to emotion CFG here):

```bash
# sweep several strengths in one run; pick the strongest natural-looking scale
python emotion_finetune/preview_emotion.py ... --emotion_scales 1,2,3
# then use the chosen scale live, e.g. EMOTION_SCALE=2 bash run_live_emotion.sh
```

Start around 2; increase until the expression is strong but before it distorts
(usually 2-4). The same `EMOTION_SCALE` env var applies to the live server.

## Troubleshooting: preview videos all look neutral

If training converged (margin ~0) and `check_adapter.py` reports distinct,
non-trivial weights, but the rendered emotions are invisible:

1. **Measure the effect size** (no retraining):

   ```bash
   python emotion_finetune/measure_emotion_effect.py \
       --dataset_path /workspace/emotion_generator_dataset \
       --generator_path IMTalker/checkpoints/generator.ckpt \
       --renderer_path IMTalker/checkpoints/renderer.ckpt \
       --ref_path IMTalker/assets/3robert.jpeg \
       --base_lora_path checkpoints/live_winner/lora/ditto_blink_lora_withaudio_r64_096_continue_2h_last.ckpt \
       --emotion_adapter_path exps/emotion_lora_v1/emotion_adapter_final.ckpt \
       --num_samples 4 --scales 1,2,4,8
   ```

   If the L2 distance from neutral is tiny at scale 1 but grows with scale, the
   model learned the right **direction** and only the **magnitude** is small.

2. **Quick confirmation**: re-render the preview with `--emotion_scale 2` (then
   4). If emotions appear, the adapter is fine - it just needs amplification.

3. **Robust fix**: retrain from a fresh `--exp_name` with
   `--emotion_cond_scale 4 --emotion_emb_lr_mult 10` (the RunPod notebook
   defaults). The learned emotion will be visible at inference scale 1.0.

4. If even scale 8 shows nothing, the direction itself is wrong - check the
   evaluation report's distinctiveness and the dataset separability line from
   `verify_emotion_dataset.py` (the motion latents must actually carry the
   expression).