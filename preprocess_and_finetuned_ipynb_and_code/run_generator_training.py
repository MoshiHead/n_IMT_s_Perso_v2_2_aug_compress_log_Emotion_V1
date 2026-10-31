"""
IMTalker Generator Training - launcher script (new file; no repo file modified).

Thin wrapper around the repository's own training implementation
(generator/train.py). Reused unchanged:

  - System (LightningModule: FMGenerator + flow-matching L1/velocity losses + EMA)
  - DataModule / AudioMotionSmirkGazeDataset (train = all but last N files, val = last N)
  - TrainOptions / BaseOptions (all model hyper-parameters)
  - the exact Trainer configuration from generator/train.py:
        accelerator='gpu', devices=-1,
        strategy='ddp_find_unused_parameters_true' if torch.cuda.device_count() > 1 else 'auto',
        max_steps=opt.iter, val_check_interval=opt.display_freq,
        check_val_every_n_epoch=None, TensorBoardLogger,
        ModelCheckpoint(every_n_train_steps=save_freq, save_top_k=-1, save_last=True)

Added on top (required by the training notebook):
  - CSV logger + plain-text progress log (epoch / step / loss / lr)
  - automatic resume from checkpoints/last.ckpt (full training state via Lightning)
  - a test video rendered with the repo's InferenceAgent after every N epochs
  - a small-dataset-safe train/val split (identical to the repo when >110 files)
  - a key-rename adapter fixing the repo's dataset/model key mismatch:
    generator/dataset.py returns 'gaze_now'/'pose_now'/'cam_now' while
    generator/FM.py reads batch['gaze']/batch['pose']/batch['cam']
    (the repo's own train.py would raise KeyError on the first batch).
"""
import argparse
import math
import os
import sys
from pathlib import Path

import torch

# torch>=2.6 flips torch.load to weights_only=True, which breaks pytorch-lightning
# 2.2.1 checkpoint (de)serialization - restore the old default (repo pins PL 2.2.1).
_TORCH_LOAD = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _TORCH_LOAD(*args, **kwargs)


torch.load = _torch_load_compat


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="IMTalker generator training (repo-faithful wrapper)")
    p.add_argument("--repo_root", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--exp_root", required=True)      # e.g. <repo>/exps
    p.add_argument("--exp_name", required=True)      # e.g. generator_exp
    p.add_argument("--wav2vec_model_path", required=True)
    p.add_argument("--renderer_ckpt", required=True)
    # training hyper-parameters (defaults = repository defaults)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_steps", type=int, default=5000000)
    p.add_argument("--max_epochs", type=int, default=-1)
    p.add_argument("--save_freq", type=int, default=100000)
    p.add_argument("--display_freq", type=int, default=10000)
    p.add_argument("--resume_ckpt", default=None, help="explicit checkpoint to resume from")
    p.add_argument("--no_auto_resume", action="store_true", help="disable auto-resume from last.ckpt")
    # per-epoch test video
    p.add_argument("--test_ref", default=None)
    p.add_argument("--test_aud", default=None)
    p.add_argument("--test_out_dir", default=None)
    p.add_argument("--test_every_n_epochs", type=int, default=1)
    p.add_argument("--test_crop", type=int, default=1,
                   help="1 = auto-crop face in the test image (repo --crop), 0 = use image as-is")
    p.add_argument("--test_cfg", type=float, default=3.0)
    p.add_argument("--test_nfe", type=int, default=10)
    p.add_argument("--test_seed", type=int, default=42)
    p.add_argument("--log_every_n_steps", type=int, default=50)
    return p.parse_args(argv)


class _KeyRenameDataset(torch.utils.data.Dataset):
    """Adapter fixing the repo's dataset/model key mismatch (see module docstring).

    generator/dataset.py returns 'gaze_now'/'pose_now'/'cam_now' but
    generator/FM.py reads batch['gaze']/batch['pose']/batch['cam'].
    """

    RENAME = (("gaze_now", "gaze"), ("pose_now", "pose"), ("cam_now", "cam"))

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        b = self.ds[i]
        for k_from, k_to in self.RENAME:
            if k_from in b:
                b[k_to] = b[k_from]
        return b


def main(args):
    repo_root = Path(args.repo_root).resolve()
    gen_dir = repo_root / "generator"
    for p in (str(repo_root), str(gen_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
    from generator.train import System, DataModule, TrainOptions  # repo classes
    from generator.dataset import AudioMotionSmirkGazeDataset     # repo dataset
    from imtalker_train_utils import TextLogCallback, EpochTestVideoCallback

    # ---- build opt exactly like the repo's TrainOptions, then override ----
    parser = argparse.ArgumentParser()
    parser = TrainOptions().initialize(parser)
    opt = parser.parse_args([])
    opt.dataset_pat = args.dataset_path          # repo CLI name (train.py)
    opt.dataset_path = args.dataset_path         # name actually read by AudioMotionSmirkGazeDataset
    opt.batch_size = args.batch_size
    opt.lr = args.lr
    opt.iter = args.max_steps
    opt.save_freq = args.save_freq
    opt.display_freq = args.display_freq
    opt.resume_ckpt = args.resume_ckpt
    opt.rank = "cuda"
    opt.wav2vec_model_path = args.wav2vec_model_path
    opt.fix_noise_seed = True                    # deterministic per-epoch test videos
    opt.seed = args.test_seed

    exp_root = Path(args.exp_root).resolve()
    exp_root.mkdir(parents=True, exist_ok=True)
    exp_dir = exp_root / args.exp_name
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    opt.exp_path = str(exp_root)
    opt.exp_name = args.exp_name

    # ---- datasets (repo split: train = all but last N files, val = last N) ----
    n_total = len(sorted((Path(opt.dataset_path) / "motion").glob("*.pt")))
    if n_total < 2:
        raise RuntimeError(
            f"Need at least 2 preprocessed clips, found {n_total} in "
            f"{opt.dataset_path}/motion - run the preprocessing notebooks first."
        )
    val_count = 100 if n_total > 110 else max(1, n_total // 10)  # repo split when >110 files
    print(f"[Data] {n_total} clips -> train={n_total - val_count}, val={val_count}", flush=True)

    class PreloadedDataModule(DataModule):
        """Repo DataModule, but datasets are built once (avoids scanning files twice)."""

        def __init__(self, opt, val_count):
            super().__init__(opt)
            self.train_dataset = _KeyRenameDataset(
                AudioMotionSmirkGazeDataset(opt=opt, start=0, end=-val_count))
            self.val_dataset = _KeyRenameDataset(
                AudioMotionSmirkGazeDataset(opt=opt, start=-val_count, end=-1))

        def setup(self, stage=None):
            pass  # datasets already built in __init__

    dm = PreloadedDataModule(opt, val_count)
    system = System(opt)

    # ---- resume target ----
    ckpt_path = None
    if args.resume_ckpt and Path(args.resume_ckpt).exists():
        ckpt_path = str(args.resume_ckpt)
    elif not args.no_auto_resume:
        last = exp_dir / "checkpoints" / "last.ckpt"
        if last.exists():
            ckpt_path = str(last)
    if ckpt_path:
        print(f"[Resume] {ckpt_path}", flush=True)

    # ---- callbacks / loggers ----
    tb_logger = TensorBoardLogger(save_dir=str(exp_root), name=args.exp_name)   # repo logger
    csv_logger = CSVLogger(save_dir=str(exp_root), name=args.exp_name)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(exp_dir / "checkpoints"),
        filename="{step:06d}",
        every_n_train_steps=opt.save_freq,
        save_top_k=-1,
        save_last=True,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    text_log = TextLogCallback(exp_dir / "train_log.txt", every_n_steps=args.log_every_n_steps)
    callbacks = [checkpoint_callback, lr_monitor, text_log]

    if args.test_ref and args.test_aud:
        callbacks.append(EpochTestVideoCallback(
            ref_path=args.test_ref, aud_path=args.test_aud,
            out_dir=args.test_out_dir or str(exp_dir / "test_videos"),
            renderer_ckpt=args.renderer_ckpt, wav2vec_model_path=args.wav2vec_model_path,
            every_n_epochs=args.test_every_n_epochs, crop=bool(args.test_crop),
            a_cfg_scale=args.test_cfg, nfe=args.test_nfe, seed=args.test_seed,
        ))

    # val_check_interval must not exceed the number of train batches per epoch
    n_gpu = max(1, torch.cuda.device_count())
    steps_per_epoch = max(1, math.ceil(len(dm.train_dataset) / (opt.batch_size * n_gpu)))
    display_freq = min(opt.display_freq, steps_per_epoch)

    # ---- trainer: exactly the repo's configuration ----
    strategy = "ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto"
    print(f"[Trainer] gpus={n_gpu} strategy={strategy} max_steps={opt.iter} "
          f"max_epochs={args.max_epochs} val_check_interval={display_freq}", flush=True)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        strategy=strategy,
        max_steps=opt.iter,
        max_epochs=args.max_epochs,
        val_check_interval=display_freq,
        check_val_every_n_epoch=None,
        logger=[tb_logger, csv_logger],
        callbacks=callbacks,
        enable_progress_bar=True,
    )

    trainer.fit(system, dm, ckpt_path=ckpt_path)
    print("[Done] training finished", flush=True)


if __name__ == "__main__":
    main(parse_args())
