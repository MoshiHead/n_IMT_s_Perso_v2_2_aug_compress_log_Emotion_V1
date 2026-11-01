"""Generate the two RunPod notebooks for the emotion fine-tuning pipeline.

Run:  python emotion_finetune/generate_notebooks.py

Writes:
  RunPod_Emotion_Training.ipynb
  RunPod_Emotion_Live_Inference.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_nb(cells: list[dict], path: Path) -> None:
    nb = notebook(cells)
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    json.loads(path.read_text(encoding="utf-8"))  # validate
    print(f"wrote {path.name}: {len(cells)} cells")

# ===========================================================================
# Training notebook
# ===========================================================================

TRAIN_INTRO_MD = """# IMTalker Emotion Fine-Tuning - RunPod Training

Fine-tune the **existing pretrained IMTalker generator** to produce five
emotional facial expressions (**happy, sad, angry, surprise, fear**) from your
robot-face dataset, while preserving the original neutral behavior and
audio-driven lip synchronization.

## Approach (combined emotion-conditioning + LoRA)

The IMTalker FlowMatchingTransformer conditions every block through
`c = t + c_embedder(a + pose + cam + gaze)`. In the live PersonaPlex pipeline
pose/cam/gaze are exact zeros, so the live conditioning is audio-only. This
adds the emotion as a peer of the timestep embedding, in the hidden (512-dim)
conditioning space (the same mechanism DiT uses for class labels):

```
c = t + c_embedder(a + pose + cam + gaze) + emotion
emotion = EmotionProjection(one_hot_5)     # Linear(5 -> 512, bias=False), zero-init
```

* **Zero-init, bias-free emotion projection** - an all-zero (neutral) emotion
  vector contributes exactly zero, forever. The untrained adapter is a no-op.
* **Emotion LoRA (rank 16)** on the FMT attention/MLP/adaLN + decoder, stacked
  on the frozen base + with-audio LoRA with a separate key namespace.
* **Neutral-anchor distillation loss** - pins the `emotion=0` slice of the
  model to the frozen original, so neutral behavior is preserved *without*
  needing neutral training videos.
* **Emotion-margin loss** - permuted emotion labels must fit worse, so the
  model is forced to actually use the conditioning (no superficial averaging).

Only ~1.5M parameters are trained; the base generator, with-audio LoRA,
wav2vec and renderer stay frozen.

## What this notebook does

1. Installs dependencies + downloads pretrained checkpoints
2. Verifies your preprocessed emotion dataset against the training contract
3. Runs a bit-exactness sanity check (proves neutral preservation by design)
4. Trains the emotion adapter (single- or multi-GPU)
5. Renders a 6-way preview (neutral + 5 emotions) and runs quantitative evaluation

## Prerequisites

* A RunPod GPU pod (RTX 5090 / any CUDA GPU; multi-GPU supported)
* Your preprocessed emotion dataset on the pod (`motion/ audio/ smirk/ gaze/`
  in the IMTalker generator format, MEAD-style filenames)
* A Hugging Face token (for the base IMTalker checkpoints)
"""

TRAIN_PARAMS_CODE = """import os

# --- Project location -------------------------------------------------------
PROJECT_ROOT = "/workspace/speech2avatar"
GIT_REPO_URL = "https://github.com/nash-raf/speech2avatar.git"  # leave blank if already cloned
GIT_BRANCH = "main"
VENV_DIR = "/workspace/emotion_train_venv"

# --- Emotion dataset (preprocessed, IMTalker generator format) ----------------
# Must contain: motion/<stem>.pt  audio/<stem>.npy  smirk/<stem>.pt  gaze/<stem>.npy
# with MEAD-style stems like M003_happy_level_1_001 (emotion parsed from the name).
DATASET_DIR = "/workspace/emotion_generator_dataset"

# Optional raw data (only used if DATASET_DIR is missing/empty): the notebook
# will pair videos<->wavs by (speaker, emotion, level) prefix and preprocess.
RAW_VIDEO_DIR = ""   # e.g. /workspace/robot_videos  (M003_happy_level_1_001.mp4)
RAW_AUDIO_DIR = ""   # e.g. /workspace/mead_wavs     (M003_happy_level_1_030.wav)

# --- Pretrained checkpoints (downloaded in Step 4 if missing) -----------------
GENERATOR_PATH = ""   # default: {PROJECT_ROOT}/IMTalker/checkpoints/generator.ckpt
RENDERER_PATH = ""    # default: {PROJECT_ROOT}/IMTalker/checkpoints/renderer.ckpt
WAV2VEC_MODEL_PATH = ""  # default: {PROJECT_ROOT}/IMTalker/checkpoints/wav2vec2-base-960h
BASE_LORA_PATH = ""   # default: {PROJECT_ROOT}/checkpoints/live_winner/lora/ditto_blink_lora_withaudio_r64_096_continue_2h_last.ckpt

HF_IMTALKER_REPO = "cbsjtu01/IMTalker"
HF_LORA_REPO = "niloy629/hdtf_preprocess"
HF_LORA_FILE = "live_winner/lora/ditto_blink_lora_withaudio_r64_096_continue_2h_last.ckpt"

# --- Training hyper-parameters ------------------------------------------------
EXP_ROOT = "/workspace/emotion_exps"
EXP_NAME = "emotion_lora_v1"

EMOTIONS = "happy,sad,angry,surprise,fear"
EMOTION_LORA_RANK = 16
EMOTION_LORA_ALPHA = 32.0
# Strengthen the emotion signal so the learned expression is VISIBLE (the default
# 1.0 learns the right direction but at a sub-visible magnitude on the motion latent).
EMOTION_COND_SCALE = 4.0     # multiplier on the emotion embedding (baked into the adapter)
EMOTION_EMB_LR_MULT = 10.0   # faster LR for the small emotion embedding vs the LoRA

BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
MAX_EPOCHS = 40
MAX_STEPS = 0              # 0 = no cap
VAL_FRACTION = 0.05
NUM_WORKERS = 4

LAMBDA_ANCHOR = 0.5        # neutral preservation strength
LAMBDA_MARGIN = 0.2        # emotion-usage strength (0 disables)
MARGIN = 0.05

AMP_MODE = "auto"          # auto | bf16 | fp16 | off
CONDITIONING_MODE = "live" # live = audio+emotion only (matches deployment)
WAV2VEC_SEC = 0.96         # live contract (24 frames + 10 prev)
SAVE_EVERY_STEPS = 1000
LOG_EVERY_STEPS = 25
SEED = 42

# --- Preview / evaluation -----------------------------------------------------
TEST_REF = ""              # default: {PROJECT_ROOT}/IMTalker/assets/3robert.jpeg
TEST_AUDIO = ""            # a .wav to render the preview with (any speech)
TEST_EMOTION_GUIDANCE = 1.0  # single output-space strength (used only if the sweep below is "")
TEST_EMOTION_GUIDANCES = "1,1.5,2,2.5,3"  # strength sweep: renders each emotion at every
                               # listed guidance into guidance_<g>/ subfolders so you can pick
                               # the strongest NATURAL look. Guidance amplifies the emotion in
                               # velocity space (exactly linear, in-distribution), so it does NOT
                               # distort the way scaling the input embedding did.
EVAL_NUM_SAMPLES = 24

EXPECTED_TORCH_VERSION = "2.8.0+cu128"

# --- Derived ------------------------------------------------------------------
IMTALKER_DIR = f"{PROJECT_ROOT}/IMTalker"
EMOTION_DIR = f"{PROJECT_ROOT}/emotion_finetune"
EXP_DIR = f"{EXP_ROOT}/{EXP_NAME}"
VENV_PYTHON = f"{VENV_DIR}/bin/python"
VENV_ACTIVATE = f"source {VENV_DIR}/bin/activate"

os.makedirs("/workspace", exist_ok=True)
print("Parameters loaded.")
print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"DATASET_DIR  = {DATASET_DIR}")
print(f"EXP_DIR      = {EXP_DIR}")
"""
TRAIN_UTILS_CODE = """import hashlib
import json as _json
import os
import socket
import subprocess
import time


def run(cmd, cwd=None, env=None, check=True, retries=1, retry_delay=8, quiet=False):
    last_returncode = None
    for attempt in range(1, retries + 1):
        if not quiet:
            print(f"$ {cmd}" + (f"   [attempt {attempt}/{retries}]" if retries > 1 else ""))
        proc = subprocess.Popen(
            cmd, shell=True, executable="/bin/bash", cwd=cwd,
            env=env or os.environ.copy(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="")
        proc.wait()
        last_returncode = proc.returncode
        if last_returncode == 0:
            return 0
        print(f"[warn] command failed with exit code {last_returncode}")
        if attempt < retries:
            print(f"[recovery] retrying in {retry_delay}s...")
            time.sleep(retry_delay)
    if check:
        raise RuntimeError(f"Command failed after {retries} attempt(s) (exit {last_returncode}): {cmd}")
    return last_returncode


def get_torch_info():
    probe = (
        "import torch, json;"
        "print(json.dumps({"
        "'version': torch.__version__,"
        "'cuda_available': torch.cuda.is_available(),"
        "'cuda_version': torch.version.cuda,"
        "'device_name': (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),"
        "'device_count': torch.cuda.device_count()"
        "}))"
    )
    out = subprocess.run([VENV_PYTHON, "-c", probe], capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout)
        print(out.stderr)
        raise RuntimeError("Failed to query torch/CUDA state inside the venv")
    return _json.loads(out.stdout.strip().splitlines()[-1])


def assert_torch_pin(auto_fix=True):
    info = get_torch_info()
    print(f"[info] torch={info['version']} cuda={info['cuda_available']} "
          f"cuda_version={info['cuda_version']} device={info['device_name']} "
          f"count={info['device_count']}")
    if info["version"] != EXPECTED_TORCH_VERSION:
        print(f"[warn] torch drifted to '{info['version']}' (expected '{EXPECTED_TORCH_VERSION}')")
        if not auto_fix:
            raise RuntimeError(f"torch pin violated: {info['version']}")
        print("[recovery] reinstalling pinned torch (cu128, no-deps)...")
        run(
            f"{VENV_ACTIVATE} && pip install --force-reinstall --no-deps "
            f"torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 "
            f"--index-url https://download.pytorch.org/whl/cu128",
            retries=3,
        )
        info = get_torch_info()
        if info["version"] != EXPECTED_TORCH_VERSION:
            raise RuntimeError(f"torch pin recovery failed; still got {info['version']}")
    print(f"[ok] torch pin confirmed: {info['version']}")
    return info


def sha256_of(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def hf_download(repo, filename, local_dir, repo_type="model", retries=3):
    repo_type_flag = f"--repo-type {repo_type}" if repo_type != "model" else ""
    os.makedirs(local_dir, exist_ok=True)
    cmd = (
        f"{VENV_ACTIVATE} && export HF_HUB_ENABLE_HF_TRANSFER=1 && "
        f"hf download {repo} {filename} {repo_type_flag} --local-dir {local_dir}"
    )
    run(cmd, retries=retries, retry_delay=15)


def tail_file(path, n=40):
    if not os.path.exists(path):
        print(f"(no {path} yet)")
        return
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()
    print("".join(lines[-n:]))


print("Utilities loaded.")
"""

TRAIN_GPU_CODE = """# GPU detection - works with any RunPod GPU configuration.
smi = subprocess.run("nvidia-smi", shell=True, executable="/bin/bash",
                     capture_output=True, text=True)
print(smi.stdout)
if smi.returncode != 0:
    raise RuntimeError("nvidia-smi failed - this notebook needs a CUDA GPU pod")

import re as _re
N_GPU = len(_re.findall(r"\\d+\\s+(?:NVIDIA|GeForce|RTX|A\\d+|H\\d+|L\\d+|Tesla)", smi.stdout)) or \\
      smi.stdout.count("MiB /")
N_GPU = max(1, N_GPU)
print(f"[gpu] detected {N_GPU} GPU(s)")
if N_GPU > 1:
    print(f"[gpu] multi-GPU training will use torchrun --nproc_per_node={N_GPU}")
else:
    print("[gpu] single-GPU training")
"""
TRAIN_CLONE_CODE = r"""# Step 0 - clone (if needed) and validate the project structure.
if GIT_REPO_URL and not os.path.isdir(os.path.join(PROJECT_ROOT, "IMTalker")):
    run(f"git clone {GIT_REPO_URL} {PROJECT_ROOT}")
    if GIT_BRANCH:
        run(f"git checkout {GIT_BRANCH}", cwd=PROJECT_ROOT)

REQUIRED_STRUCTURE = [
    "IMTalker",
    "IMTalker/generator/FM.py",
    "IMTalker/generator/FMT.py",
    "IMTalker/generator/train_lora.py",
    "IMTalker/renderer/models.py",
    "emotion_finetune",
    "emotion_finetune/emotion_modules.py",
    "emotion_finetune/emotion_dataset.py",
    "emotion_finetune/train_emotion.py",
    "emotion_finetune/verify_emotion_dataset.py",
    "emotion_finetune/sanity_check.py",
    "emotion_finetune/preview_emotion.py",
    "emotion_finetune/evaluate_emotion_model.py",
    "emotion_finetune/measure_emotion_effect.py",
    "emotion_finetune/check_adapter.py",
]
missing = [p for p in REQUIRED_STRUCTURE if not os.path.exists(os.path.join(PROJECT_ROOT, p))]
if missing:
    raise RuntimeError(
        "project structure incomplete; missing:\n  " + "\n  ".join(missing)
        + "\nMake sure the emotion_finetune package is committed to the repository."
    )
print("[ok] project structure validated")
"""

TRAIN_SYSDEPS_CODE = """# Step 1 - system dependencies (matches live.md Fresh Pod Setup).
run(
    "export DEBIAN_FRONTEND=noninteractive && apt-get update && "
    "apt-get install -y python3.11 python3.11-venv ffmpeg git",
    retries=2,
)
"""

TRAIN_VENV_CODE = f"""# Step 2 - Python 3.11 venv + pinned PyTorch (cu128, required for RTX 5090 / sm_120).
run(f"python3.11 -m venv {{VENV_DIR}}")
run(f"{{VENV_ACTIVATE}} && python -m pip install --upgrade pip wheel")
run(f"{{VENV_ACTIVATE}} && python -m pip install setuptools==80.9.0")
run(
    f"{{VENV_ACTIVATE}} && pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 "
    f"--index-url https://download.pytorch.org/whl/cu128",
    retries=3,
)
torch_info = assert_torch_pin()
if not torch_info["cuda_available"]:
    raise RuntimeError("CUDA is not available inside the venv - cannot train on CPU")
"""

TRAIN_PYDEPS_CODE = """# Step 3 - Python dependencies (IMTalker requirements + training extras).
run(f"{VENV_ACTIVATE} && pip install -r {IMTALKER_DIR}/requirement.txt", retries=2)
run(
    f'{VENV_ACTIVATE} && pip install "huggingface_hub[cli]" hf_transfer tensorboard '
    f'einops librosa face_alignment==1.4.1',
    retries=2,
)
print("[ok] python dependencies installed")
"""

TRAIN_HFAUTH_CODE = """# Step 4 - Hugging Face authentication (needed for cbsjtu01/IMTalker + niloy629 datasets).
import getpass

token = os.environ.get("HF_TOKEN", "").strip()
if not token:
    token = getpass.getpass("Hugging Face token (read access): ").strip()
os.environ["HF_TOKEN"] = token
run(f"{VENV_ACTIVATE} && hf auth login --token $HF_TOKEN", quiet=True)
print("[ok] Hugging Face authenticated")
"""

TRAIN_ASSETS_CODE = """# Step 5 - download pretrained checkpoints (base generator, renderer, wav2vec, with-audio LoRA).
IMTALKER_CKPT_DIR = f"{IMTALKER_DIR}/checkpoints"
CHECKPOINT_DIR = f"{PROJECT_ROOT}/checkpoints"

if not GENERATOR_PATH:
    GENERATOR_PATH = f"{IMTALKER_CKPT_DIR}/generator.ckpt"
if not RENDERER_PATH:
    RENDERER_PATH = f"{IMTALKER_CKPT_DIR}/renderer.ckpt"
if not WAV2VEC_MODEL_PATH:
    WAV2VEC_MODEL_PATH = f"{IMTALKER_CKPT_DIR}/wav2vec2-base-960h"
if not BASE_LORA_PATH:
    BASE_LORA_PATH = f"{CHECKPOINT_DIR}/{HF_LORA_FILE}"

for fname in [
    "config.yaml",
    "renderer.ckpt",
    "generator.ckpt",
    "wav2vec2-base-960h/config.json",
    "wav2vec2-base-960h/pytorch_model.bin",
    "wav2vec2-base-960h/preprocessor_config.json",
    "wav2vec2-base-960h/feature_extractor_config.json",
]:
    target = f"{IMTALKER_CKPT_DIR}/{fname}"
    if not os.path.exists(target):
        hf_download(HF_IMTALKER_REPO, fname, IMTALKER_CKPT_DIR)
    else:
        print(f"[ok] present: {fname}")

if not os.path.exists(BASE_LORA_PATH):
    hf_download(HF_LORA_REPO, HF_LORA_FILE, CHECKPOINT_DIR, repo_type="dataset")
else:
    print(f"[ok] present: with-audio LoRA")

for path, label in [
    (GENERATOR_PATH, "generator"), (RENDERER_PATH, "renderer"),
    (WAV2VEC_MODEL_PATH, "wav2vec"), (BASE_LORA_PATH, "with-audio LoRA"),
]:
    if not os.path.exists(path):
        raise RuntimeError(f"missing {label}: {path}")
    print(f"[ok] {label}: {path}")
"""
TRAIN_DATASET_MD = """## Step 6 - Dataset: locate, pair (if raw), and verify

The training needs the IMTalker generator format:

```
DATASET_DIR/
  motion/<stem>.pt    Tensor[T, 32]   (renderer motion latents)
  audio/<stem>.npy    ndarray[T, 768] (wav2vec2 last hidden, 25 fps)
  smirk/<stem>.pt     pose_params (T,3), cam (T,3)
  gaze/<stem>.npy     ndarray[T, 2]
```

If you already preprocessed your robot videos (the expected case), point
`DATASET_DIR` at it. If you only have raw videos + MEAD wavs, set
`RAW_VIDEO_DIR`/`RAW_AUDIO_DIR` in the parameters cell: the notebook pairs
each video with its audio by **(speaker, emotion, level) prefix** (sequence
numbers may differ, e.g. `M003_happy_level_1_001.mp4` <->
`M003_happy_level_1_030.wav`) and runs the motion/audio preprocessing.

The verification step then checks the data against the **actual training
code contract** (shapes, dtypes, temporal alignment, minimum length, emotion
label coverage, latent separability) - not just file existence.
"""

TRAIN_DATASET_CODE = """# Dataset resolution: preprocessed (expected) or raw -> pair + preprocess.
REQUIRE_SMIRK_GAZE = 1


def _dataset_ready(path):
    return os.path.isdir(os.path.join(path, "motion")) and \\
           len(list(__import__("pathlib").Path(path, "motion").glob("*.pt"))) > 0


if _dataset_ready(DATASET_DIR):
    print(f"[dataset] using preprocessed dataset: {DATASET_DIR}")
elif RAW_VIDEO_DIR and RAW_AUDIO_DIR:
    print("[dataset] preprocessed dataset not found - building from raw videos + wavs")
    print(f"  videos: {RAW_VIDEO_DIR}")
    print(f"  wavs  : {RAW_AUDIO_DIR}")
    DATASET_DIR = "/workspace/emotion_generator_dataset"
    run(
        f"{VENV_ACTIVATE} && python {EMOTION_DIR}/preprocess_emotion_data.py "
        f"--video_dir {RAW_VIDEO_DIR} --audio_dir {RAW_AUDIO_DIR} "
        f"--output_dir {DATASET_DIR} "
        f"--renderer_path {RENDERER_PATH} "
        f"--wav2vec_model_path {WAV2VEC_MODEL_PATH} "
        f"--fps 25 --device cuda",
        cwd=PROJECT_ROOT,
        retries=1,
    )
    REQUIRE_SMIRK_GAZE = 0  # minimal preprocessing produces motion+audio only
else:
    raise RuntimeError(
        f"No preprocessed dataset at {DATASET_DIR} and RAW_VIDEO_DIR/RAW_AUDIO_DIR "
        "are not set. Either preprocess your robot videos first (see the README) "
        "or set the raw directories in the parameters cell."
    )
print(f"[dataset] DATASET_DIR = {DATASET_DIR} (require_smirk_gaze={REQUIRE_SMIRK_GAZE})")
"""

TRAIN_VERIFY_CODE = """# Dataset verification against the training contract.
run(
    f"{VENV_ACTIVATE} && python {EMOTION_DIR}/verify_emotion_dataset.py "
    f"--dataset_path {DATASET_DIR} "
    f"--wav2vec_sec {WAV2VEC_SEC} --fps 25 --num_prev_frames 10 "
    f"--conditioning_mode {CONDITIONING_MODE} "
    f"--require_smirk_gaze {REQUIRE_SMIRK_GAZE}",
    cwd=PROJECT_ROOT,
)
print("[ok] dataset verified")
"""

TRAIN_SANITY_MD = """## Step 7 - Bit-exactness sanity check

Before training, prove on the **real model stack** that:

1. the emotion-upgraded model with `emotion=None` is **bit-identical** to the
   original (neutral preservation by construction),
2. the untrained (zero-init) adapter is a no-op,
3. the emotion conditioning pathway is actually wired in (perturbing the
   weights changes the output),
4. adapter save/load is an exact roundtrip.
"""

TRAIN_SANITY_CODE = """# Bit-exactness sanity check on the real stack.
run(
    f"{VENV_ACTIVATE} && python {EMOTION_DIR}/sanity_check.py "
    f"--generator_path {GENERATOR_PATH} "
    f"--base_lora_path {BASE_LORA_PATH} "
    f"--device cuda",
    cwd=PROJECT_ROOT,
)
print("[ok] sanity checks passed - neutral behavior is preserved by construction")
"""
TRAIN_LAUNCH_MD = """## Step 8 - Train the emotion adapter

Trains **only** the emotion projection + emotion LoRA (~1.5M params) on top of
the frozen live stack. Losses:

* flow-matching L1 + velocity (the original IMTalker objective),
* **anchor** (`lambda_anchor`): keeps `emotion=0` equal to the frozen original,
* **margin** (`lambda_margin`): permuted emotions must fit worse.

Checkpoints are written to `EXP_DIR/checkpoints/` (`last.ckpt` auto-resumes),
and a portable `emotion_adapter_latest.ckpt` (EMA weights) is exported after
every save. Re-running this cell resumes from `last.ckpt`.
"""

TRAIN_LAUNCH_CODE = r"""import subprocess
from pathlib import Path

Path(EXP_DIR).mkdir(parents=True, exist_ok=True)
train_log = f"{EXP_DIR}/train_console.log"
pid_file = f"{EXP_DIR}/train.pid"

common_args = (
    f"--dataset_path {DATASET_DIR} "
    f"--generator_path {GENERATOR_PATH} "
    f"--base_lora_path {BASE_LORA_PATH} "
    f"--exp_root {EXP_ROOT} --exp_name {EXP_NAME} "
    f"--emotions {EMOTIONS} "
    f"--emotion_lora_rank {EMOTION_LORA_RANK} --emotion_lora_alpha {EMOTION_LORA_ALPHA} "
    f"--emotion_cond_scale {EMOTION_COND_SCALE} --emotion_emb_lr_mult {EMOTION_EMB_LR_MULT} "
    f"--wav2vec_sec {WAV2VEC_SEC} --conditioning_mode {CONDITIONING_MODE} "
    f"--require_smirk_gaze {REQUIRE_SMIRK_GAZE} "
    f"--batch_size {BATCH_SIZE} --lr {LR} --weight_decay {WEIGHT_DECAY} "
    f"--warmup_steps {WARMUP_STEPS} --max_epochs {MAX_EPOCHS} --max_steps {MAX_STEPS} "
    f"--val_fraction {VAL_FRACTION} --num_workers {NUM_WORKERS} "
    f"--lambda_anchor {LAMBDA_ANCHOR} --lambda_margin {LAMBDA_MARGIN} --margin {MARGIN} "
    f"--amp {AMP_MODE} --seed {SEED} "
    f"--save_every_steps {SAVE_EVERY_STEPS} --log_every_steps {LOG_EVERY_STEPS} "
    f"--resume auto"
)

if N_GPU > 1:
    cmd = (
        f"{VENV_ACTIVATE} && torchrun --nproc_per_node={N_GPU} "
        f"{EMOTION_DIR}/train_emotion.py {common_args}"
    )
else:
    cmd = f"{VENV_ACTIVATE} && python {EMOTION_DIR}/train_emotion.py {common_args}"

print("launching:", cmd, "\\n")
log_handle = open(train_log, "a")
proc = subprocess.Popen(
    cmd, shell=True, executable="/bin/bash", cwd=PROJECT_ROOT,
    stdout=log_handle, stderr=subprocess.STDOUT,
    start_new_session=True,
)
Path(pid_file).write_text(str(proc.pid))
print(f"[ok] training started | pid={proc.pid}")
print(f"     console log: {train_log}")
print(f"     text log   : {EXP_DIR}/train_log.txt")
print("run the MONITOR cell below to follow progress.")
"""

TRAIN_MONITOR_CODE = r"""# MONITOR - re-run anytime to check progress.
import os
from pathlib import Path

pid_file = Path(f"{EXP_DIR}/train.pid")
if pid_file.exists():
    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, 0)
        alive = True
    except OSError:
        alive = False
    print(f"training process pid={pid}: {'RUNNING' if alive else 'not running'}")

print("\\n----- train_log.txt (last 25) -----")
tail_file(f"{EXP_DIR}/train_log.txt", 25)

ckpts = sorted(Path(EXP_DIR, "checkpoints").glob("*.ckpt"),
               key=lambda p: p.stat().st_mtime) if Path(EXP_DIR, "checkpoints").is_dir() else []
print(f"\\ncheckpoints ({len(ckpts)}):", [c.name for c in ckpts[-4:]])
adapter = Path(EXP_DIR, "emotion_adapter_latest.ckpt")
if adapter.exists():
    print(f"latest exported adapter: {adapter} ({adapter.stat().st_size/1e6:.1f} MB)")
"""

TRAIN_STOP_CODE = """# STOP training (graceful SIGTERM; last.ckpt is kept and auto-resumed on re-launch).
STOP_TRAINING = False   # set to True and re-run this cell to actually stop

if not STOP_TRAINING:
    print("set STOP_TRAINING = True and re-run this cell to stop training.")
else:
    import os, signal
    from pathlib import Path
    pid = int(Path(f"{EXP_DIR}/train.pid").read_text().strip())
    os.killpg(os.getpgid(pid), signal.SIGTERM)
    print(f"stop signal sent to process group of pid {pid}")
"""
TRAIN_PREVIEW_MD = """## Step 9 - Preview + quantitative evaluation

**Preview** renders the same reference image + audio with the original model
(neutral) and each of the 5 trained emotions, then writes a 6-panel comparison
video. **Evaluation** generates motion latents for held-out samples and
reports neutral preservation, emotion distinctiveness, lip-sync retention and
temporal stability.

Run these on `emotion_adapter_latest.ckpt` (any time during training) or
`emotion_adapter_final.ckpt` (after training).

**If the preview videos all look neutral**, retrain with `EMOTION_COND_SCALE = 4.0`
(the default) so the signal is strong enough to converge, then re-run.

**If the emotions are visible but too subtle** (the common case): the model has
faithfully learned the *intensity present in the training data* (robot `level_1`
expressions are mild), and the flow-matching loss cannot reward stronger-than-data
expressions. The fix is to **amplify the learned direction at inference** - but the
mechanism matters:

* Do **NOT** raise `emotion_scale` (the input-embedding scale). The measurement shows
  it grows *super-linearly* and explodes (8 -> 290), which is exactly the distortion /
  rotating-face / broken-lip-sync failure. Keep `emotion_scale = 1.0`.
* **DO** raise `emotion_guidance` (the output-space strength). It extrapolates the
  *velocity* difference between the emotional and neutral pass, so every forward pass
  stays at the trained (in-distribution) magnitude and the effect grows **exactly
  linearly** - a strong expression that stays natural and keeps lip-sync.

The preview cell already sweeps `TEST_EMOTION_GUIDANCES = "1,1.5,2,2.5,3"`; pick the
strongest guidance that still looks natural and set it as `EMOTION_GUIDANCE` in the
live notebook. The optional diagnostic below quantifies the effect size at each
guidance value.
"""


TRAIN_PREVIEW_CODE = """# Render the 6-way preview video (neutral + happy + sad + angry + surprise + fear).
from pathlib import Path

ADAPTER = f"{EXP_DIR}/emotion_adapter_final.ckpt"
if not Path(ADAPTER).exists():
    ADAPTER = f"{EXP_DIR}/emotion_adapter_latest.ckpt"
if not TEST_REF:
    TEST_REF = f"{IMTALKER_DIR}/assets/3robert.jpeg"
if not TEST_AUDIO:
    TEST_AUDIO = f"{IMTALKER_DIR}/assets/audio_3.wav"

PREVIEW_OUT = f"{EXP_DIR}/preview"
run(
    f"{VENV_ACTIVATE} && python {EMOTION_DIR}/preview_emotion.py "
    f"--ref_path {TEST_REF} --aud_path {TEST_AUDIO} "
    f"--generator_path {GENERATOR_PATH} --renderer_path {RENDERER_PATH} "
    f"--wav2vec_model_path {WAV2VEC_MODEL_PATH} "
    f"--base_lora_path {BASE_LORA_PATH} "
    f"--emotion_adapter_path {ADAPTER} "
    f"--emotion_lora_rank {EMOTION_LORA_RANK} --emotion_lora_alpha {EMOTION_LORA_ALPHA} "
    + (f"--emotion_guidances {TEST_EMOTION_GUIDANCES} " if str(TEST_EMOTION_GUIDANCES).strip()
       else f"--emotion_guidance {TEST_EMOTION_GUIDANCE} ")
    f"--out_dir {PREVIEW_OUT} --name robot --crop --device cuda",
    cwd=PROJECT_ROOT,
)
print(f"[ok] preview videos in {PREVIEW_OUT}")
print(f"     comparison: {PREVIEW_OUT}/robot_comparison.mp4")
"""

TRAIN_EVAL_CODE = """# Quantitative evaluation on held-out samples.
from pathlib import Path

ADAPTER = f"{EXP_DIR}/emotion_adapter_final.ckpt"
if not Path(ADAPTER).exists():
    ADAPTER = f"{EXP_DIR}/emotion_adapter_latest.ckpt"
if not TEST_REF:
    TEST_REF = f"{IMTALKER_DIR}/assets/3robert.jpeg"

run(
    f"{VENV_ACTIVATE} && python {EMOTION_DIR}/evaluate_emotion_model.py "
    f"--dataset_path {DATASET_DIR} "
    f"--generator_path {GENERATOR_PATH} --renderer_path {RENDERER_PATH} "
    f"--ref_path {TEST_REF} "
    f"--base_lora_path {BASE_LORA_PATH} "
    f"--emotion_adapter_path {ADAPTER} "
    f"--emotion_lora_rank {EMOTION_LORA_RANK} --emotion_lora_alpha {EMOTION_LORA_ALPHA} "
    f"--wav2vec_sec {WAV2VEC_SEC} "
    f"--num_samples {EVAL_NUM_SAMPLES} --val_fraction {VAL_FRACTION} "
    f"--out_json {EXP_DIR}/eval_report.json --device cuda",
    cwd=PROJECT_ROOT,
)
"""

TRAIN_MEASURE_CODE = """# OPTIONAL diagnostic: measure how much the emotion conditioning moves the
# generated motion latent at several emotion_guidance values.  Guidance grows the
# effect ~linearly and in-distribution (no distortion), unlike the input scale.
from pathlib import Path

ADAPTER = f"{EXP_DIR}/emotion_adapter_final.ckpt"
if not Path(ADAPTER).exists():
    ADAPTER = f"{EXP_DIR}/emotion_adapter_latest.ckpt"
if not TEST_REF:
    TEST_REF = f"{IMTALKER_DIR}/assets/3robert.jpeg"

run(
    f"{VENV_ACTIVATE} && python {EMOTION_DIR}/measure_emotion_effect.py "
    f"--dataset_path {DATASET_DIR} "
    f"--generator_path {GENERATOR_PATH} --renderer_path {RENDERER_PATH} "
    f"--ref_path {TEST_REF} "
    f"--base_lora_path {BASE_LORA_PATH} "
    f"--emotion_adapter_path {ADAPTER} "
    f"--emotion_lora_rank {EMOTION_LORA_RANK} --emotion_lora_alpha {EMOTION_LORA_ALPHA} "
    f"--wav2vec_sec {WAV2VEC_SEC} --num_samples 4 --scales 1,1.5,2,2.5,3 --device cuda",
    cwd=PROJECT_ROOT,
)
"""

TRAIN_NEXT_MD = """## Next steps

1. **Inspect the preview video** (`EXP_DIR/preview/robot_comparison.mp4`):
   each emotion should look visibly distinct while lip-sync stays intact.
2. **Check the evaluation report** (`EXP_DIR/eval_report.json`):
   neutral relative-L2 should be near 0, emotion distinctiveness well above
   chance (0.2), lip-sync retention comparable across settings.
3. **Run the live server** with the trained adapter using
   `RunPod_Emotion_Live_Inference.ipynb` - set `AVATAR_EMOTION` to
   happy/sad/angry/surprise/fear and the avatar keeps that expression for the
   whole conversation; `neutral` (or no adapter) is the original behavior.

The portable adapter is `EXP_DIR/emotion_adapter_final.ckpt` (EMA weights).
Upload it to Hugging Face or copy it to your inference pod.
"""


def build_training_notebook() -> list[dict]:
    return [
        md(TRAIN_INTRO_MD),
        md("## Parameters\n\nEdit these before running."),
        code(TRAIN_PARAMS_CODE),
        md("## Utilities"),
        code(TRAIN_UTILS_CODE),
        md("## GPU detection"),
        code(TRAIN_GPU_CODE),
        md("## Step 0 - Clone / validate the project"),
        code(TRAIN_CLONE_CODE),
        md("## Step 1 - System dependencies"),
        code(TRAIN_SYSDEPS_CODE),
        md("## Step 2 - Python 3.11 venv + pinned PyTorch"),
        code(TRAIN_VENV_CODE),
        md("## Step 3 - Python dependencies"),
        code(TRAIN_PYDEPS_CODE),
        md("## Step 4 - Hugging Face authentication"),
        code(TRAIN_HFAUTH_CODE),
        md("## Step 5 - Download pretrained checkpoints"),
        code(TRAIN_ASSETS_CODE),
        md(TRAIN_DATASET_MD),
        code(TRAIN_DATASET_CODE),
        code(TRAIN_VERIFY_CODE),
        md(TRAIN_SANITY_MD),
        code(TRAIN_SANITY_CODE),
        md(TRAIN_LAUNCH_MD),
        code(TRAIN_LAUNCH_CODE),
        code(TRAIN_MONITOR_CODE),
        code(TRAIN_STOP_CODE),
        md(TRAIN_PREVIEW_MD),
        code(TRAIN_PREVIEW_CODE),
        code(TRAIN_EVAL_CODE),
        code(TRAIN_MEASURE_CODE),
        md(TRAIN_NEXT_MD),
    ]

# ===========================================================================
# Inference notebook
# ===========================================================================

INFER_INTRO_MD = """# PersonaPlex + IMTalker Live Avatar with Emotion - RunPod

Runs the **emotion-enabled** live avatar server. This is the stock AH
AudioPace winner server (PersonaPlex + IMTalker) started through a thin
wrapper (`emotion_finetune/launch_emotion_live.py`) that applies your trained
emotion adapter to the FM generator.

Set `AVATAR_EMOTION` to control the avatar's expression for the whole
conversation:

| `AVATAR_EMOTION` | behavior |
|---|---|
| `neutral` (default) | original IMTalker behavior (bit-identical) |
| `happy` | consistent happy expression |
| `sad` | consistent sad expression |
| `angry` | consistent angry expression |
| `surprise` | consistent surprised expression |
| `fear` | consistent fearful expression |

The emotion enters the IMTalker conditioning as a fifth additive term; the
PersonaPlex reply generation, the RMS gate, the timing contract, and the
renderer are all unchanged. With no adapter configured the server is
bit-identical to the stock AH server.

**Prerequisite:** a trained `emotion_adapter.ckpt` from
`RunPod_Emotion_Training.ipynb` (or leave `EMOTION_ADAPTER_PATH` empty to run
the stock neutral server).
"""

INFER_PARAMS_CODE = """import os

# --- Project location -------------------------------------------------------
PROJECT_ROOT = "/workspace/speech2avatar"
GIT_REPO_URL = "https://github.com/nash-raf/speech2avatar.git"  # leave blank if already cloned
GIT_BRANCH = "main"
VENV_DIR = "/workspace/preprocess_5090"

# --- Service ------------------------------------------------------------------
HOST = "0.0.0.0"
PORT = 8998

RUNPOD_POD_ID = os.environ.get("RUNPOD_POD_ID", "")
PUBLIC_URL = f"https://{RUNPOD_POD_ID}-{PORT}.proxy.runpod.net/" if RUNPOD_POD_ID else None

# --- Emotion --------------------------------------------------------------------
# Local path to a trained emotion adapter, OR an HF spec "repo_id:filename"
# (downloaded in Step 9). Leave EMPTY to run the stock neutral server.
EMOTION_ADAPTER_PATH = ""          # e.g. f"{PROJECT_ROOT}/checkpoints/emotion_adapter_final.ckpt"
EMOTION_ADAPTER_HF = ""            # e.g. "your-hf-user/imtalker-emotion:emotion_adapter_final.ckpt"
AVATAR_EMOTION = "neutral"         # neutral | happy | sad | angry | surprise | fear
EMOTION_GUIDANCE = "2.0"           # output-space emotion strength; if the expression is too
                                   # subtle raise this (1.5-3). Exactly linear + in-distribution,
                                   # so it stays natural. (Do NOT use the input scale - it distorts.)
EMOTION_LORA_RANK = "16"           # must match the adapter's training rank
EMOTION_LORA_ALPHA = "32"          # must match the adapter's training alpha

# --- Hugging Face sources (must match live.md) -----------------------------------
HF_ADAPTER_REPO = "niloy629/hdtf_preprocess"
HF_ADAPTER_FILE = "personaplex_lookahead_rms_adapter/checkpoints/personaplex_lookahead096_future048_rms50_adapter.pt"
HF_SILENCE_FILE = "personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt"
HF_LORA_REPO = "niloy629/hdtf_preprocess"
HF_LORA_FILE = "live_winner/lora/ditto_blink_lora_withaudio_r64_096_continue_2h_last.ckpt"
HF_LORA_SHA256 = "f34de1bd748857bc802102da578046bb10bd5f664460607d9c785ff35922e52f"
HF_IMTALKER_REPO = "cbsjtu01/IMTalker"
HF_PERSONAPLEX_REPO = "brianmatzelle/personaplex-7b-v1-bnb-4bit"
HF_PERSONAPLEX_MIMI_REPO = "nvidia/personaplex-7b-v1"

EXPECTED_TORCH_VERSION = "2.8.0+cu128"

# --- Optional run_live_emotion.sh overrides (leave "" for script defaults) --------
REF_PATH = ""
A_CFG_SCALE = ""
NFE = ""
VOICE_PROMPT = ""
DISABLE_LORA = ""
ENABLE_SEARCH = False
WEB_SEARCH_ENABLED = False
WEB_SEARCH_API_KEY = os.environ.get("WEB_SEARCH_API_KEY", "")
ROUTER_THRESHOLD = "0.40"
ROUTER_RULES = "1"

STARTUP_TIMEOUT_SEC = 900
POLL_INTERVAL_SEC = 5

# --- Derived paths ------------------------------------------------------------------
IMTALKER_DIR = f"{PROJECT_ROOT}/IMTalker"
PERSONAPLEX_DIR = f"{PROJECT_ROOT}/personaplex"
CHECKPOINT_DIR = f"{PROJECT_ROOT}/checkpoints"
PERSONAPLEX_BNB4_DIR = f"{CHECKPOINT_DIR}/personaplex_bnb4"
EMOTION_DIR = f"{PROJECT_ROOT}/emotion_finetune"
VENV_PYTHON = f"{VENV_DIR}/bin/python"
VENV_ACTIVATE = f"source {VENV_DIR}/bin/activate"
LOG_PATH = f"{PROJECT_ROOT}/live_server_emotion.log"
PID_PATH = f"{PROJECT_ROOT}/.run_live_emotion.pid"

os.makedirs("/workspace", exist_ok=True)
print("Parameters loaded.")
print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"HOST:PORT    = {HOST}:{PORT}")
print(f"AVATAR_EMOTION = {AVATAR_EMOTION}  (guidance={EMOTION_GUIDANCE})")
print(f"EMOTION_ADAPTER_PATH = {EMOTION_ADAPTER_PATH or EMOTION_ADAPTER_HF or '(none - stock neutral server)'}")
if PUBLIC_URL:
    print(f"PUBLIC_URL = {PUBLIC_URL}")
"""
INFER_STRUCTURE_CODE = r"""# Step 0 - clone (if needed) and validate the project structure.
if GIT_REPO_URL and not os.path.isdir(os.path.join(PROJECT_ROOT, "IMTalker")):
    run(f"git clone {GIT_REPO_URL} {PROJECT_ROOT}")
    if GIT_BRANCH:
        run(f"git checkout {GIT_BRANCH}", cwd=PROJECT_ROOT)

REQUIRED_STRUCTURE = [
    "IMTalker",
    "IMTalker/liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py",
    "IMTalker/liveTry.py",
    "IMTalker/ws_av_binary_codec.py",
    "IMTalker/static/index_v3_binary_fullscreen_aj_nodrop.html",
    "scripts/download_live_assets.sh",
    "emotion_finetune",
    "emotion_finetune/emotion_modules.py",
    "emotion_finetune/launch_emotion_live.py",
    "run_live_emotion.sh",
]
missing = [p for p in REQUIRED_STRUCTURE if not os.path.exists(os.path.join(PROJECT_ROOT, p))]
if missing:
    raise RuntimeError("project structure incomplete; missing:\n  " + "\n  ".join(missing))
run("chmod +x run_live_emotion.sh scripts/download_live_assets.sh", cwd=PROJECT_ROOT, check=False)
print("[ok] project structure validated")
"""

INFER_SYSDEPS_CODE = """# Step 1 - system dependencies (matches live.md Fresh Pod Setup).
run(
    "export DEBIAN_FRONTEND=noninteractive && apt-get update && "
    "apt-get install -y python3.11 python3.11-venv ffmpeg git",
    retries=2,
)
"""

INFER_GPU_CODE = """# Step 2 - GPU / driver check.
smi = subprocess.run("nvidia-smi", shell=True, executable="/bin/bash",
                     capture_output=True, text=True)
print(smi.stdout)
if smi.returncode != 0:
    raise RuntimeError("nvidia-smi failed - this notebook needs a CUDA GPU pod")
"""

INFER_VENV_CODE = f"""# Step 3 - Python 3.11 venv.
run(f"python3.11 -m venv {{VENV_DIR}}")
run(f"{{VENV_ACTIVATE}} && python -m pip install --upgrade pip wheel")
run(f"{{VENV_ACTIVATE}} && python -m pip install setuptools==80.9.0")
"""

INFER_TORCH_CODE = """# Step 4 - pinned PyTorch (cu128, required for RTX 5090 / sm_120).
run(
    f"{VENV_ACTIVATE} && pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 "
    f"--index-url https://download.pytorch.org/whl/cu128",
    retries=3,
)
torch_info = assert_torch_pin()
if not torch_info["cuda_available"]:
    raise RuntimeError("CUDA is not available inside the venv")
"""

INFER_PYDEPS_CODE = """# Step 5 - IMTalker Python dependencies.
run(f"{VENV_ACTIVATE} && pip install -r {IMTALKER_DIR}/requirement.txt", retries=2)
"""

INFER_HFDEPS_CODE = """# Step 6 - Hugging Face tooling + streaming/codec dependencies.
run(f'{VENV_ACTIVATE} && pip install "huggingface_hub[cli]" hf_transfer tensorboard', retries=2)
run(
    f'{VENV_ACTIVATE} && pip install "sphn>=0.2.0,<0.3.0" einops sentencepiece aiohttp av aiortc bitsandbytes',
    retries=2,
)
"""

INFER_HFAUTH_CODE = """# Step 7 - Hugging Face authentication.
import getpass

token = os.environ.get("HF_TOKEN", "").strip()
if not token:
    token = getpass.getpass("Hugging Face token (read access): ").strip()
os.environ["HF_TOKEN"] = token
run(f"{VENV_ACTIVATE} && hf auth login --token $HF_TOKEN", quiet=True)
print("[ok] Hugging Face authenticated")
"""

INFER_ASSETS_CODE = """# Step 8 - download the runtime assets (the project's own script).
run(
    f"{VENV_ACTIVATE} && export HF_HUB_ENABLE_HF_TRANSFER=1 && "
    f"export SPEECH2AVATAR_ROOT={PROJECT_ROOT} && "
    f"bash scripts/download_live_assets.sh",
    cwd=PROJECT_ROOT,
    retries=2,
)
"""

INFER_MOSHI_CODE = """# Step 9 - install the PersonaPlex/Moshi package (--no-deps, per live.md).
moshi_candidates = [
    os.path.join(PERSONAPLEX_BNB4_DIR, "moshi"),
    os.path.join(PERSONAPLEX_DIR, "moshi"),
]
moshi_dir = next((p for p in moshi_candidates if os.path.isdir(p)), None)
if moshi_dir is None:
    raise RuntimeError(f"moshi package not found in {moshi_candidates}")
run(f"{VENV_ACTIVATE} && pip install -e {moshi_dir} --no-deps", retries=2)
print(f"[ok] moshi installed from {moshi_dir}")
"""

INFER_ADAPTER_CODE = """# Step 10 - resolve the emotion adapter (local path or Hugging Face).
if EMOTION_ADAPTER_HF and not EMOTION_ADAPTER_PATH:
    repo_id, _, fname = EMOTION_ADAPTER_HF.partition(":")
    if not repo_id or not fname:
        raise RuntimeError("EMOTION_ADAPTER_HF must be 'repo_id:filename'")
    EMOTION_ADAPTER_PATH = f"{CHECKPOINT_DIR}/emotion/{fname}"
    if not os.path.exists(EMOTION_ADAPTER_PATH):
        hf_download(repo_id, fname, f"{CHECKPOINT_DIR}/emotion", repo_type="model")

if EMOTION_ADAPTER_PATH:
    if not os.path.exists(EMOTION_ADAPTER_PATH):
        raise RuntimeError(f"emotion adapter not found: {EMOTION_ADAPTER_PATH}")
    size_mb = os.path.getsize(EMOTION_ADAPTER_PATH) / 1e6
    print(f"[ok] emotion adapter: {EMOTION_ADAPTER_PATH} ({size_mb:.1f} MB)")
    print(f"     avatar emotion: {AVATAR_EMOTION} (guidance={EMOTION_GUIDANCE})")
else:
    if AVATAR_EMOTION.lower() not in ("neutral", "none", ""):
        print(f"[warn] AVATAR_EMOTION={AVATAR_EMOTION} but no adapter configured - "
              "the server will run neutral")
    print("[ok] no emotion adapter - stock neutral AH server")
"""
INFER_LAUNCH_MD = """## Step 11 - Launch the emotion-enabled live server

Launches `run_live_emotion.sh` in the background (it sources the venv, resolves
the checkpoints, exports the emotion configuration, and starts the stock AH
server through the emotion wrapper). The emotion is read from the environment:

* `EMOTION_ADAPTER_PATH` - trained adapter (empty = stock neutral server)
* `AVATAR_EMOTION` - constant expression for the whole conversation
* `EMOTION_GUIDANCE` - output-space strength (1.5-3 for stronger, natural expressions)
"""

INFER_LAUNCH_CODE = """env_overrides = {
    "HOST": HOST,
    "PORT": str(PORT),
    "VENV_DIR": VENV_DIR,
    "SPEECH2AVATAR_ROOT": PROJECT_ROOT,
    "PROJECT_ROOT": PROJECT_ROOT,
    "EMOTION_ADAPTER_PATH": EMOTION_ADAPTER_PATH,
    "AVATAR_EMOTION": AVATAR_EMOTION,
    "EMOTION_GUIDANCE": str(EMOTION_GUIDANCE),
    "EMOTION_LORA_RANK": str(EMOTION_LORA_RANK),
    "EMOTION_LORA_ALPHA": str(EMOTION_LORA_ALPHA),
}
if REF_PATH:
    env_overrides["REF_PATH"] = REF_PATH
if A_CFG_SCALE:
    env_overrides["A_CFG_SCALE"] = str(A_CFG_SCALE)
if NFE:
    env_overrides["NFE"] = str(NFE)
if VOICE_PROMPT:
    env_overrides["VOICE_PROMPT"] = VOICE_PROMPT
if DISABLE_LORA:
    env_overrides["DISABLE_LORA"] = str(DISABLE_LORA)
if ENABLE_SEARCH:
    env_overrides["ENABLE_SEARCH"] = "1"
    env_overrides["ROUTER_THRESHOLD"] = str(ROUTER_THRESHOLD)
    env_overrides["ROUTER_RULES"] = str(ROUTER_RULES)
    if WEB_SEARCH_ENABLED:
        env_overrides["WEB_SEARCH_ENABLED"] = "1"
        env_overrides["WEB_SEARCH_API_KEY"] = WEB_SEARCH_API_KEY
    else:
        env_overrides["WEB_SEARCH_ENABLED"] = "0"

print("Resolved environment overrides for run_live_emotion.sh:")
for k, v in env_overrides.items():
    print(f"  {k}={'***' if k == 'WEB_SEARCH_API_KEY' else v}")

import subprocess
launch_env = os.environ.copy()
launch_env.update(env_overrides)
log_file = open(LOG_PATH, "w")
live_proc = subprocess.Popen(
    ["bash", "run_live_emotion.sh"], cwd=PROJECT_ROOT, env=launch_env,
    stdout=log_file, stderr=subprocess.STDOUT,
)
with open(PID_PATH, "w") as f:
    f.write(str(live_proc.pid))
print(f"[ok] launched emotion live server: pid={live_proc.pid}")
print(f"     log file: {LOG_PATH}")
"""

INFER_HEALTH_CODE = """# Step 12 - wait for healthy startup.
import time

HEALTHY_MARKERS = {
    "lora loaded": "lora loaded",
    "using direct Moshi reply hidden": "using direct Moshi reply hidden",
    "installed PersonaPlex graphed hidden capture": "installed PersonaPlex graphed hidden capture",
    "serving": "serving",
}
EMOTION_MARKERS = ("[emotion] live emotion enabled", "[emotion] disabled")

start = time.time()
seen = set()
emotion_seen = False
while time.time() - start < STARTUP_TIMEOUT_SEC:
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", errors="ignore") as f:
            log = f.read()
        for key, marker in HEALTHY_MARKERS.items():
            if marker in log and key not in seen:
                seen.add(key)
                print(f"[ok] {key}")
        if not emotion_seen and any(m in log for m in EMOTION_MARKERS):
            emotion_seen = True
            for line in log.splitlines():
                if "[emotion]" in line:
                    print(f"[ok] {line.strip()}")
        if "Uvicorn running on" in log:
            print("[ok] Uvicorn running")
            break
        if "Traceback" in log or "Missing required path" in log:
            print("[error] startup failed - last log lines:")
            tail_file(LOG_PATH, 40)
            raise RuntimeError("live server startup failed (see log above)")
    time.sleep(POLL_INTERVAL_SEC)
else:
    print("[error] startup timed out - last log lines:")
    tail_file(LOG_PATH, 40)
    raise RuntimeError(f"live server did not become healthy within {STARTUP_TIMEOUT_SEC}s")

missing = set(HEALTHY_MARKERS) - seen
if missing:
    print(f"[warn] markers not seen yet: {missing}")
print("[ok] live server is healthy")
"""

INFER_HTTP_CODE = r"""# Step 13 - HTTP / Uvicorn confirmation.
import urllib.request

try:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=10) as resp:
        print(f"[ok] HTTP {resp.status} from http://127.0.0.1:{PORT}/")
except Exception as exc:
    print(f"[warn] HTTP check failed: {exc}")

if PUBLIC_URL:
    print(f"\\nOpen the avatar UI: {PUBLIC_URL}")
    print(f"Avatar emotion: {AVATAR_EMOTION}")
"""

INFER_OPS_MD = """## Operational cells (run any time)

To **change the emotion**, stop the server (cell below), set a new
`AVATAR_EMOTION` in the parameters cell, and re-run the launch + health cells.
"""

INFER_TAIL_CODE = """# Tail the live server log.
tail_file(LOG_PATH, 200)
"""

INFER_STOP_CODE = """# Stop switch - set STOP_SERVER = True and re-run to terminate the live server.
STOP_SERVER = False

if not STOP_SERVER:
    print("set STOP_SERVER = True and re-run this cell to stop the live server.")
else:
    import os, signal
    with open(PID_PATH) as f:
        pid = int(f.read().strip())
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        print(f"stopped live server process group of pid {pid}")
    except ProcessLookupError:
        print("process already gone")
"""


def build_inference_notebook() -> list[dict]:
    return [
        md(INFER_INTRO_MD),
        md("## Parameters\n\nEdit these before running."),
        code(INFER_PARAMS_CODE),
        md("## Utilities"),
        code(TRAIN_UTILS_CODE),
        md("## Step 0 - Clone / validate the project"),
        code(INFER_STRUCTURE_CODE),
        md("## Step 1 - System dependencies"),
        code(INFER_SYSDEPS_CODE),
        md("## Step 2 - GPU / driver check"),
        code(INFER_GPU_CODE),
        md("## Step 3 - Python 3.11 venv"),
        code(INFER_VENV_CODE),
        md("## Step 4 - Pinned PyTorch (cu128)"),
        code(INFER_TORCH_CODE),
        md("## Step 5 - IMTalker Python dependencies"),
        code(INFER_PYDEPS_CODE),
        md("## Step 6 - Hugging Face tooling + streaming dependencies"),
        code(INFER_HFDEPS_CODE),
        md("## Step 7 - Hugging Face authentication"),
        code(INFER_HFAUTH_CODE),
        md("## Step 8 - Download runtime assets"),
        code(INFER_ASSETS_CODE),
        md("## Step 9 - Install PersonaPlex / Moshi"),
        code(INFER_MOSHI_CODE),
        md("## Step 10 - Resolve the emotion adapter"),
        code(INFER_ADAPTER_CODE),
        md(INFER_LAUNCH_MD),
        code(INFER_LAUNCH_CODE),
        md("## Step 12 - Wait for healthy startup"),
        code(INFER_HEALTH_CODE),
        md("## Step 13 - HTTP / Uvicorn confirmation"),
        code(INFER_HTTP_CODE),
        md(INFER_OPS_MD),
        code(INFER_TAIL_CODE),
        code(INFER_STOP_CODE),
    ]


def main() -> None:
    write_nb(build_training_notebook(), ROOT / "RunPod_Emotion_Training.ipynb")
    write_nb(build_inference_notebook(), ROOT / "RunPod_Emotion_Live_Inference.ipynb")


if __name__ == "__main__":
    main()