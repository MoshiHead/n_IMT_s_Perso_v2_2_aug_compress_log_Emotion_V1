"""
Training utilities for IMTalker Generator training (new files for the training
notebook - no repository file is modified).

Contains the two Lightning callbacks used by run_generator_training.py and
04_train_generator.ipynb:

  - TextLogCallback:        appends human-readable progress (epoch / step /
                            loss / lr) to a plain-text log file on rank 0.
  - EpochTestVideoCallback:  after every N training epochs, renders a test video
                            with the current generator weights using the
                            repository's own inference pipeline
                            (generator/generate.py: InferenceAgent).
"""
import time
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only


class TextLogCallback(pl.Callback):
    """Writes epoch / step / loss / lr lines to a plain-text log file (rank 0)."""

    def __init__(self, log_file, every_n_steps=50):
        super().__init__()
        self.log_file = Path(log_file)
        self.every_n_steps = max(1, int(every_n_steps))
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, line):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        self._write(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] training started | "
            f"gpus={trainer.num_devices} | max_steps={trainer.max_steps} | "
            f"max_epochs={trainer.max_epochs}"
        )

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.every_n_steps != 0:
            return
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        loss = float(trainer.callback_metrics.get("train_loss", float("nan")))
        fm = float(trainer.callback_metrics.get("fm_loss", float("nan")))
        self._write(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] epoch={trainer.current_epoch:04d} "
            f"step={trainer.global_step:07d} lr={lr:.3e} "
            f"train_loss={loss:.6f} fm_loss={fm:.6f}"
        )

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        self._write(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] VALIDATION epoch={trainer.current_epoch:04d} "
            f"step={trainer.global_step:07d} "
            f"val_loss={float(m.get('val_loss', float('nan'))):.6f} "
            f"val_fm_loss={float(m.get('val_fm_loss', float('nan'))):.6f}"
        )

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        self._write(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] epoch={trainer.current_epoch:04d} "
            f"completed at step={trainer.global_step:07d}"
        )


class EpochTestVideoCallback(pl.Callback):
    """
    Renders a test video after every N training epochs with the current
    generator weights, using the repository's own inference pipeline
    (generator/generate.py: InferenceAgent). Rank 0 only.

    The current weights are saved to a temporary checkpoint containing the
    'model.*' entries of the LightningModule's state_dict - exactly the format
    InferenceAgent._load_generator_weights() expects - and loaded into the
    agent's FMGenerator through the repo's own loading code path. The frozen
    wav2vec2 weights are skipped (identical to the pretrained copy the agent
    already holds), which keeps the temporary file small.
    """

    def __init__(self, ref_path, aud_path, out_dir, renderer_ckpt, wav2vec_model_path,
                 every_n_epochs=1, crop=True, a_cfg_scale=3.0, nfe=10, seed=42):
        super().__init__()
        self.ref_path = str(ref_path)
        self.aud_path = str(aud_path)
        self.out_dir = Path(out_dir)
        self.renderer_ckpt = str(renderer_ckpt)
        self.wav2vec_model_path = str(wav2vec_model_path)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.crop = bool(crop)
        self.a_cfg_scale = float(a_cfg_scale)
        self.nfe = int(nfe)
        self.seed = int(seed)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._agent = None

    def _build_agent_opt(self, generator_ckpt):
        import argparse
        from generator.generate import InferenceOptions  # repo option class
        parser = argparse.ArgumentParser()
        parser = InferenceOptions().initialize(parser)
        opt = parser.parse_args([])
        opt.rank = 0
        opt.renderer_path = self.renderer_ckpt
        opt.generator_path = str(generator_ckpt)
        opt.wav2vec_model_path = self.wav2vec_model_path
        opt.res_dir = str(self.out_dir)
        opt.a_cfg_scale = self.a_cfg_scale
        opt.nfe = self.nfe
        opt.seed = self.seed
        opt.fix_noise_seed = True  # deterministic noise -> comparable videos across epochs
        return opt

    def _get_agent(self, generator_ckpt):
        if self._agent is None:
            from generator.generate import InferenceAgent  # repo inference pipeline
            self._agent = InferenceAgent(self._build_agent_opt(generator_ckpt))
        else:
            self._agent._load_generator_weights(str(generator_ckpt), self._agent.rank)
        return self._agent

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        tmp_ckpt = self.out_dir / f"_tmp_generator_epoch{trainer.current_epoch:04d}.ckpt"
        try:
            sd = pl_module.state_dict()
            sd = {k: v.cpu() for k, v in sd.items()
                  if k.startswith("model.") and not k.startswith("model.audio_encoder.wav2vec2")}
            torch.save({"state_dict": sd}, tmp_ckpt)

            agent = self._get_agent(tmp_ckpt)
            out_mp4 = self.out_dir / f"epoch_{trainer.current_epoch:04d}.mp4"
            agent.run_inference(
                str(out_mp4), self.ref_path, self.aud_path, None, None,
                a_cfg_scale=self.a_cfg_scale, nfe=self.nfe, crop=self.crop, seed=self.seed,
            )
            print(f"[EpochTest] epoch {trainer.current_epoch}: saved {out_mp4}", flush=True)
        except Exception as e:  # never let the test video kill the training
            print(f"[EpochTest] epoch {trainer.current_epoch}: FAILED - {e}", flush=True)
        finally:
            if tmp_ckpt.exists():
                tmp_ckpt.unlink()
