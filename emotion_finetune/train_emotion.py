"""Emotion-conditioned fine-tuning of the IMTalker FM generator.

Trains ONLY the emotion modules (zero-init emotion projection + emotion LoRA)
on top of the frozen live stack (base generator + with-audio LoRA):

  loss = L1_flow + L1_velocity                      (original objective)
       + lambda_anchor * L1(pred(e=0), frozen_ref)  (neutral preservation)
       + lambda_margin * hinge(wrong vs right)      (conditioning is used)

Single GPU:   python train_emotion.py --dataset_path ...
Multi GPU:    torchrun --nproc_per_node=N train_emotion.py --dataset_path ...
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "IMTalker"), str(_REPO_ROOT / "IMTalker" / "generator")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from emotion_finetune.emotion_dataset import (
    EmotionAudioMotionDataset,
    collate_emotion_batch,
    stratified_split,
)
from emotion_finetune.emotion_modules import (
    EMOTIONS,
    clean_generator_state,
    clear_emotion,
    emotion_trainable_parameters,
    save_emotion_adapter,
    set_emotion,
    set_emotion_lora_enabled,
    upgrade_fm_with_emotion,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="IMTalker emotion fine-tuning (frozen base + emotion LoRA)")
    # paths
    p.add_argument("--repo_root", default=str(_REPO_ROOT / "IMTalker"))
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--generator_path", required=True)
    p.add_argument("--base_lora_path", default="", help="with-audio LoRA checkpoint (the live stack); empty = raw base")
    p.add_argument("--base_lora_rank", type=int, default=64)
    p.add_argument("--base_lora_alpha", type=float, default=128.0)
    p.add_argument("--exp_root", default="exps")
    p.add_argument("--exp_name", default="emotion_lora_v1")
    # model / emotion
    p.add_argument("--emotions", default=",".join(EMOTIONS))
    p.add_argument("--emotion_lora_rank", type=int, default=16)
    p.add_argument("--emotion_lora_alpha", type=float, default=32.0)
    p.add_argument("--emotion_lora_dropout", type=float, default=0.0)
    # data
    p.add_argument("--wav2vec_sec", type=float, default=0.96)
    p.add_argument("--num_prev_frames", type=int, default=10)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--conditioning_mode", choices=("live", "full"), default="live")
    p.add_argument("--require_smirk_gaze", type=int, default=1)
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=4)
    # optimization
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--max_epochs", type=int, default=40)
    p.add_argument("--max_steps", type=int, default=0, help="0 = no step cap")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", choices=("auto", "bf16", "fp16", "off"), default="auto")
    p.add_argument("--ema_decay", type=float, default=0.999)
    # losses
    p.add_argument("--lambda_anchor", type=float, default=0.5)
    p.add_argument("--lambda_margin", type=float, default=0.2)
    p.add_argument("--margin", type=float, default=0.05)
    # bookkeeping
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every_steps", type=int, default=25)
    p.add_argument("--save_every_steps", type=int, default=1000)
    p.add_argument("--resume", default="auto", help="auto | <path> | none")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# DDP / reproducibility helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    if "RANK" not in os.environ:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return 0, 1, 0, device, False
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return (
        dist.get_rank(),
        dist.get_world_size(),
        local_rank,
        torch.device(f"cuda:{local_rank}"),
        True,
    )


def cleanup_ddp():
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_main(rank: int) -> bool:
    return rank == 0


# ---------------------------------------------------------------------------
# Model construction (frozen live stack + emotion modules)
# ---------------------------------------------------------------------------

def build_model_opt(args, device) -> argparse.Namespace:
    return argparse.Namespace(
        fps=float(args.fps),
        rank=str(device),
        wav2vec_sec=float(args.wav2vec_sec),
        num_prev_frames=int(args.num_prev_frames),
        only_last_features=True,
        skip_fm_audio_encoder=True,          # dataset feeds precomputed [T,768]
        audio_adapter_mode="none",
        no_pose_cam=args.conditioning_mode == "live",  # exact-zero pose/cam/gaze == live path
        ode_atol=1e-5,
        ode_rtol=1e-5,
        torchdiffeq_ode_method="euler",
        dim_c=32,
        dim_w=32,
        dim_motion=32,
        dim_h=512,
        mlp_ratio=4.0,
        fmt_depth=8,
        num_heads=8,
        audio_dropout_prob=0.1,
        seed=int(args.seed),
        fix_noise_seed=False,
        static_pose_cam=False,
        debug_session=False,
    )


def build_frozen_live_stack(args, device):
    """FMGenerator + base checkpoint (+ optional frozen with-audio LoRA)."""
    from generator.FM import FMGenerator
    from generator.train_lora import apply_lora_to_model

    opt = build_model_opt(args, device)
    fm = FMGenerator(opt)

    base_ckpt = torch.load(args.generator_path, map_location="cpu")
    missing, unexpected = fm.load_state_dict(clean_generator_state(base_ckpt), strict=False)
    print(f"[model] base generator loaded: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    if args.base_lora_path:
        apply_lora_to_model(
            fm,
            rank=int(args.base_lora_rank),
            alpha=float(args.base_lora_alpha),
            dropout=0.0,  # dropout is inactive at eval; 0.0 keeps the frozen function exact
        )
        lora_ckpt = torch.load(args.base_lora_path, map_location="cpu")
        lora_state = clean_generator_state(lora_ckpt)
        missing_lora, unexpected_lora = fm.load_state_dict(lora_state, strict=False)
        lora_keys = sum(1 for key in lora_state if "lora_" in key)
        print(
            f"[model] with-audio LoRA loaded: {args.base_lora_path} "
            f"lora_keys={lora_keys} missing={len(missing_lora)} unexpected={len(unexpected_lora)}",
            flush=True,
        )
    else:
        print("[model] no base LoRA -- training on the raw base generator", flush=True)

    for param in fm.parameters():
        param.requires_grad_(False)
    fm.to(device)
    fm.eval()  # frozen backbone stays in eval mode for the whole run
    return fm, opt

# ---------------------------------------------------------------------------
# EMA (trainable emotion parameters only)
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, named_params, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {name: param.detach().clone() for name, param in named_params}

    @torch.no_grad()
    def update(self, named_params):
        for name, param in named_params:
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {name: value.cpu() for name, value in self.shadow.items()}

    def load_state_dict(self, state, named_params):
        devices = {name: param.device for name, param in named_params}
        for name, value in state.items():
            if name in self.shadow:
                self.shadow[name] = value.to(devices.get(name, "cpu"))


def named_trainable_params(fm):
    trainable = set(id(p) for p in emotion_trainable_parameters(fm))
    return [(name, param) for name, param in fm.named_parameters() if id(param) in trainable]


# ---------------------------------------------------------------------------
# LR schedule: linear warmup -> cosine decay to min_lr_ratio
# ---------------------------------------------------------------------------

def lr_at_step(step: int, total_steps: int, base_lr: float, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return base_lr * float(step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)


# ---------------------------------------------------------------------------
# Loss computation (the heart of the emotion fine-tune)
# ---------------------------------------------------------------------------

def append_dims(t: torch.Tensor, ndims: int) -> torch.Tensor:
    return t.reshape(*t.shape, *((1,) * ndims))


def compute_flow_target(m_now: torch.Tensor):
    noise = torch.randn_like(m_now)
    times = torch.rand(m_now.size(0), device=m_now.device)
    t = append_dims(times, m_now.ndim - 1)
    noised = t * m_now + (1 - t) * noise
    gt_flow = m_now - noise
    return noised, gt_flow, times


def velocity_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])


def emotion_training_losses(fm, batch, args, autocast_ctx):
    """Returns (total_loss, metrics_dict).

    Three forward passes of the small FMT:
      1. correct emotion   -> flow-matching L1 + velocity (the real objective)
      2. no emotion        -> anchor L1 against the frozen reference function
                              (fp32, keeps emotion=0 == original model)
      3. permuted emotion  -> hinge margin so the conditioning is causally used
    """
    m_now = batch["m_now"]
    noised, gt_flow, times = compute_flow_target(m_now)
    batch = dict(batch)
    batch["m_now"] = noised

    # --- 1) main flow-matching loss with the correct emotion ----------------
    with autocast_ctx:
        set_emotion(fm, batch["emotion"])
        pred = fm(batch, t=times)
        fm_loss = F.l1_loss(pred.float(), gt_flow.float())
        vel_loss = velocity_l1(pred.float(), gt_flow.float())

    # --- 2) neutral anchor (fp32 for a clean preservation signal) -----------
    anchor_loss = pred.new_zeros(())
    if float(args.lambda_anchor) > 0:
        clear_emotion(fm)
        set_emotion_lora_enabled(fm, False)
        with torch.no_grad():
            ref_pred = fm(batch, t=times)  # frozen reference function
        set_emotion_lora_enabled(fm, True)
        pred_neutral = fm(batch, t=times)  # trainable path, still no emotion
        anchor_loss = F.l1_loss(pred_neutral.float(), ref_pred.float())

    # --- 3) emotion margin (permuted labels must fit worse) ------------------
    margin_loss = pred.new_zeros(())
    if float(args.lambda_margin) > 0:
        batch_size = int(batch["emotion"].shape[0])
        perm = torch.randperm(batch_size, device=batch["emotion"].device)
        same = (batch["emotion"][perm] - batch["emotion"]).abs().sum(dim=-1) < 1e-6
        if int((~same).sum()) > 0:
            with autocast_ctx:
                set_emotion(fm, batch["emotion"][perm])
                pred_wrong = fm(batch, t=times)
            per_sample = lambda a, b: (a - b).abs().mean(dim=(1, 2))
            d_right = per_sample(pred.float(), gt_flow.float())
            d_wrong = per_sample(pred_wrong.float(), gt_flow.float())
            # Correct direction: penalise when the WRONG emotion fits nearly as
            # well as the correct one (d_wrong - d_right < margin).  The loss
            # decreases toward 0 as the model learns to use the conditioning.
            hinge = F.relu(float(args.margin) - (d_wrong - d_right))
            margin_loss = hinge[~same].mean()

    clear_emotion(fm)

    total = (
        fm_loss
        + vel_loss
        + float(args.lambda_anchor) * anchor_loss
        + float(args.lambda_margin) * margin_loss
    )
    metrics = {
        "loss": float(total.detach()),
        "fm": float(fm_loss.detach()),
        "vel": float(vel_loss.detach()),
        "anchor": float(anchor_loss.detach()),
        "margin": float(margin_loss.detach()),
    }
    return total, metrics

# ---------------------------------------------------------------------------
# Checkpoint save / resume
# ---------------------------------------------------------------------------

def save_training_checkpoint(fm, optimizer, scaler, ema, args, epoch, step, path):
    payload = {
        "format": "emotion_train_state_v1",
        "args": vars(args),
        "epoch": int(epoch),
        "step": int(step),
        "emotion_projection": {k: v.detach().cpu() for k, v in fm.emotion_projection.state_dict().items()},
        "emotion_lora": {
            k: v.detach().cpu()
            for k, v in fm.state_dict().items()
            if ".emotion_lora_A." in k or ".emotion_lora_B." in k
        },
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, str(tmp))
    os.replace(tmp, path)


def load_training_checkpoint(fm, optimizer, scaler, ema, path, device):
    payload = torch.load(str(path), map_location="cpu")
    model_sd = fm.state_dict()
    loaded = 0
    with torch.no_grad():
        for key, value in payload.get("emotion_projection", {}).items():
            full = f"emotion_projection.{key}"
            if full in model_sd:
                model_sd[full].copy_(value.to(device))
                loaded += 1
        for key, value in payload.get("emotion_lora", {}).items():
            if key in model_sd:
                model_sd[key].copy_(value.to(device))
                loaded += 1
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if ema is not None and payload.get("ema") is not None:
        ema.load_state_dict(payload["ema"], named_trainable_params(fm))
    with contextlib.suppress(Exception):
        torch.set_rng_state(payload["torch_rng"].cpu() if torch.is_tensor(payload["torch_rng"]) else payload["torch_rng"])
        np.random.set_state(payload["numpy_rng"])
        random.setstate(payload["python_rng"])
    print(f"[resume] loaded {loaded} emotion params from {path} (epoch={payload.get('epoch')} step={payload.get('step')})", flush=True)
    return int(payload.get("epoch", 0)), int(payload.get("step", 0))




# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(fm, val_loader, args, device, max_batches: int = 40):
    fm_loss_sum = vel_sum = anchor_sum = 0.0
    count = 0
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        noised, gt_flow, times = compute_flow_target(batch["m_now"])
        batch = dict(batch)
        batch["m_now"] = noised
        set_emotion(fm, batch["emotion"])
        pred = fm(batch, t=times)
        fm_loss_sum += float(F.l1_loss(pred.float(), gt_flow.float()))
        vel_sum += float(velocity_l1(pred.float(), gt_flow.float()))
        clear_emotion(fm)
        set_emotion_lora_enabled(fm, False)
        ref_pred = fm(batch, t=times)
        set_emotion_lora_enabled(fm, True)
        pred_neutral = fm(batch, t=times)
        anchor_sum += float(F.l1_loss(pred_neutral.float(), ref_pred.float()))
        count += 1
    clear_emotion(fm)
    count = max(1, count)
    return {
        "val_fm": fm_loss_sum / count,
        "val_vel": vel_sum / count,
        "val_anchor": anchor_sum / count,
    }


def move_batch_to_device(batch: dict, device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out

# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device, ddp = setup_ddp()
    seed_all(int(args.seed) + rank)
    emotions = [e.strip() for e in str(args.emotions).split(",") if e.strip()]

    exp_dir = Path(args.exp_root) / args.exp_name
    ckpt_dir = exp_dir / "checkpoints"
    if is_main(rank):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "train_config.json").write_text(json.dumps(vars(args), indent=2))

    # ---- model -------------------------------------------------------------
    fm, model_opt = build_frozen_live_stack(args, device)
    summary = upgrade_fm_with_emotion(
        fm,
        emotion_dim=len(emotions),
        emotions=emotions,
        lora_rank=int(args.emotion_lora_rank),
        lora_alpha=float(args.emotion_lora_alpha),
        lora_dropout=float(args.emotion_lora_dropout),
        verbose=is_main(rank),
    )
    fm.eval()

    # ---- data ----------------------------------------------------------------
    data_opt = argparse.Namespace(
        dataset_path=args.dataset_path,
        wav2vec_sec=float(args.wav2vec_sec),
        fps=float(args.fps),
        num_prev_frames=int(args.num_prev_frames),
    )
    probe = EmotionAudioMotionDataset(
        data_opt,
        conditioning_mode=args.conditioning_mode,
        emotions=emotions,
        require_smirk_gaze=bool(args.require_smirk_gaze),
    )
    train_stems, val_stems = stratified_split(probe.samples, args.val_fraction, seed=int(args.seed))
    if is_main(rank):
        (exp_dir / "split.json").write_text(json.dumps({"train": train_stems, "val": val_stems}, indent=2))
        print(f"[data] train={len(train_stems)} val={len(val_stems)}", flush=True)

    train_dataset = EmotionAudioMotionDataset(
        data_opt, stems=train_stems, conditioning_mode=args.conditioning_mode,
        emotions=emotions, require_smirk_gaze=bool(args.require_smirk_gaze),
    )
    val_dataset = EmotionAudioMotionDataset(
        data_opt, stems=val_stems, conditioning_mode=args.conditioning_mode,
        emotions=emotions, require_smirk_gaze=bool(args.require_smirk_gaze),
    )

    if ddp:
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(
            train_dataset, batch_size=int(args.batch_size), sampler=train_sampler,
            num_workers=int(args.num_workers), collate_fn=collate_emotion_batch,
            pin_memory=True, persistent_workers=int(args.num_workers) > 0, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=int(args.batch_size), shuffle=False,
            num_workers=0, collate_fn=collate_emotion_batch,
        )
    else:
        from torch.utils.data import DataLoader

        train_sampler = None
        train_loader = DataLoader(
            train_dataset, batch_size=int(args.batch_size), shuffle=True,
            num_workers=int(args.num_workers), collate_fn=collate_emotion_batch,
            pin_memory=True, persistent_workers=int(args.num_workers) > 0, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=int(args.batch_size), shuffle=False,
            num_workers=0, collate_fn=collate_emotion_batch,
        )

    # ---- optimization --------------------------------------------------------
    trainable = emotion_trainable_parameters(fm)
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * int(args.max_epochs)
    if int(args.max_steps) > 0:
        total_steps = min(total_steps, int(args.max_steps))

    amp_mode = str(args.amp)
    if amp_mode == "auto":
        amp_mode = "bf16" if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else "off"
    use_autocast = amp_mode in ("bf16", "fp16") and device.type == "cuda"
    autocast_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda") if amp_mode == "fp16" and device.type == "cuda" else None
    if is_main(rank):
        print(
            f"[train] steps/epoch={steps_per_epoch} total_steps={total_steps} "
            f"amp={amp_mode} batch={args.batch_size} lr={args.lr} "
            f"lambda_anchor={args.lambda_anchor} lambda_margin={args.lambda_margin}",
            flush=True,
        )

    ema = EMA(named_trainable_params(fm), decay=float(args.ema_decay))

    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP

        fm = DDP(fm, device_ids=[local_rank], find_unused_parameters=True)

    # ---- resume ----------------------------------------------------------------
    start_epoch, global_step = 0, 0
    resume_path = None
    if str(args.resume) == "auto":
        candidate = ckpt_dir / "last.ckpt"
        if candidate.exists():
            resume_path = candidate
    elif str(args.resume) not in ("", "none"):
        resume_path = Path(args.resume)
    if resume_path is not None and Path(resume_path).exists():
        raw_fm = fm.module if ddp else fm
        start_epoch, global_step = load_training_checkpoint(
            raw_fm, optimizer, scaler, ema, resume_path, device
        )
        start_epoch = min(start_epoch, int(args.max_epochs) - 1)

    log_path = exp_dir / "train_log.txt"

    def log_line(line: str) -> None:
        if is_main(rank):
            print(line, flush=True)
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    log_line(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] training start | world={world_size} "
        f"trainable={summary['trainable_params']:,} targets={summary['targets']}"
    )

    # ---- loop -------------------------------------------------------------------
    stop_training = False
    for epoch in range(start_epoch, int(args.max_epochs)):
        if stop_training:
            break
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        t_epoch = time.time()
        running = {"loss": 0.0, "fm": 0.0, "vel": 0.0, "anchor": 0.0, "margin": 0.0}
        seen = 0

        for batch in train_loader:
            if global_step >= total_steps:
                stop_training = True
                break
            lr = lr_at_step(global_step, total_steps, float(args.lr), int(args.warmup_steps), float(args.min_lr_ratio))
            for group in optimizer.param_groups:
                group["lr"] = lr

            batch = move_batch_to_device(batch, device)
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if use_autocast
                else contextlib.nullcontext()
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss, metrics = emotion_training_losses(fm, batch, args, autocast_ctx)

            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
                optimizer.step()

            raw_fm = fm.module if ddp else fm
            ema.update(named_trainable_params(raw_fm))
            global_step += 1
            seen += 1
            for key in running:
                running[key] += metrics[key]

            if is_main(rank) and (global_step % int(args.log_every_steps) == 0 or global_step == 1):
                log_line(
                    f"[epoch {epoch:03d} step {global_step:06d}/{total_steps}] lr={lr:.2e} "
                    f"loss={running['loss']/seen:.5f} fm={running['fm']/seen:.5f} "
                    f"vel={running['vel']/seen:.5f} anchor={running['anchor']/seen:.5f} "
                    f"margin={running['margin']/seen:.5f}"
                )
                running = {key: 0.0 for key in running}
                seen = 0

            if is_main(rank) and global_step % int(args.save_every_steps) == 0:
                save_training_checkpoint(
                    raw_fm, optimizer, scaler, ema, args, epoch, global_step,
                    ckpt_dir / f"step_{global_step:07d}.ckpt",
                )
                save_training_checkpoint(
                    raw_fm, optimizer, scaler, ema, args, epoch, global_step,
                    ckpt_dir / "last.ckpt",
                )
                save_emotion_adapter(
                    raw_fm, exp_dir / "emotion_adapter_latest.ckpt",
                    config_extra={"step": global_step, "epoch": epoch, "conditioning_mode": args.conditioning_mode},
                    override_params=ema.state_dict(),
                )

        # ---- end of epoch: validation + checkpoint ----------------------------
        raw_fm = fm.module if ddp else fm
        val_metrics = validate(raw_fm, val_loader, args, device)
        log_line(
            f"[epoch {epoch:03d} done] time={time.time()-t_epoch:.1f}s "
            f"val_fm={val_metrics['val_fm']:.5f} val_vel={val_metrics['val_vel']:.5f} "
            f"val_anchor={val_metrics['val_anchor']:.6f}"
        )
        if is_main(rank):
            save_training_checkpoint(
                raw_fm, optimizer, scaler, ema, args, epoch + 1, global_step,
                ckpt_dir / "last.ckpt",
            )
            save_emotion_adapter(
                raw_fm, exp_dir / "emotion_adapter_latest.ckpt",
                config_extra={"step": global_step, "epoch": epoch + 1, "conditioning_mode": args.conditioning_mode},
                override_params=ema.state_dict(),
            )

    if is_main(rank):
        raw_fm = fm.module if ddp else fm
        save_training_checkpoint(
            raw_fm, optimizer, scaler, ema, args, int(args.max_epochs), global_step,
            ckpt_dir / "last.ckpt",
        )
        save_emotion_adapter(
            raw_fm, exp_dir / "emotion_adapter_final.ckpt",
            config_extra={"step": global_step, "epoch": int(args.max_epochs), "conditioning_mode": args.conditioning_mode},
            override_params=ema.state_dict(),
        )
        log_line(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] training finished at step {global_step}")
        print(f"[done] adapter exported: {exp_dir / 'emotion_adapter_final.ckpt'}", flush=True)

    cleanup_ddp()


if __name__ == "__main__":
    main()