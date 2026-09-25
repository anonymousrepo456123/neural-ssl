"""I-JEPA model (NDT_JEPA) — latent-space pretraining for BIT neural spike data.

Architecture:

  ┌────────────────────────────────────────────────────────────────┐
  │  spikes (B, T, C)                                              │
  │     │                                                          │
  │     ├── online encoder (NeuralEncoder) ── with masking ──┐     │
  │     │       output: context_repr (B, T_p, D)             │     │
  │     │                                                    │     │
  │     │                                                    ▼     │
  │     │                                              JEPA Predictor
  │     │                                              (small transformer)
  │     │                                                    │     │
  │     │                                                    ▼     │
  │     │                                            pred_repr (B,T_p,D)
  │     │                                                    │     │
  │     └── target encoder (EMA of online) ── no masking ────┤     │
  │             output: target_repr (B, T_p, D)              │     │
  │                                                          ▼     │
  │                                              SmoothL1(masked positions)
  └────────────────────────────────────────────────────────────────┘

Loss is computed in latent space (no spike reconstruction). The target
encoder is updated by EMA of the online encoder after every optimiser
step via the Trainer's ``post_optimizer_step`` hook.

Checkpoint convention: ``save_checkpoint`` writes the *target* (EMA)
encoder to ``encoder.bin`` / ``encoder_config.pth``, matching the
I-JEPA recommendation (EMA encoder for downstream tasks) and the
format expected by ``finetune.py``. The online (student) encoder is
saved separately as ``student_encoder.bin``.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import DictConfig, update_config
from utils.datasets import DATASET_CHANNELS
from models.model_output import ModelOutput
from models.ndt import NeuralEncoder, NeuralEncoderLayer, DEFAULT_CONFIG


@dataclass
class JEPAOutput(ModelOutput):
    loss: torch.FloatTensor = None
    n_examples: torch.LongTensor = None
    preds: torch.FloatTensor = None
    targets: torch.FloatTensor = None
    mask: Optional[torch.LongTensor] = None


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class JEPAPredictor(nn.Module):
    """Small transformer that maps online-encoder outputs to predicted
    target-encoder outputs.

    Following V-JEPA 2.1, the predictor is shallower and (optionally)
    narrower than the main encoder. Operates on the full patched sequence
    (the encoder already replaces masked patches with the learned
    ``mask_token``), so the predictor can attend over context patches to
    fill in the masked ones.

    Uses the same NeuralEncoderLayer as the encoder (RoPE, causal/acausal
    context mask) so positional information is properly propagated.
    """

    def __init__(
        self,
        encoder_hidden: int,
        max_F: int,
        config: DictConfig,
    ):
        super().__init__()
        self.encoder_hidden = encoder_hidden
        self.pred_hidden = config.hidden_size

        if self.pred_hidden != encoder_hidden:
            self.in_proj = nn.Linear(encoder_hidden, self.pred_hidden)
            self.out_proj = nn.Linear(self.pred_hidden, encoder_hidden)
        else:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()

        # Fresh predictor-side mask token so the encoder's mask_token does not
        # overfit to the predictor's input distribution.
        self.pred_mask_token = nn.Parameter(torch.randn(1, 1, self.pred_hidden))

        self.layers = nn.ModuleList([
            NeuralEncoderLayer(layer_idx=i, max_F=max_F, config=config)
            for i in range(config.n_layers)
        ])
        self.out_norm = nn.LayerNorm(self.pred_hidden)

    def forward(
        self,
        context_repr: torch.FloatTensor,        # (B, T_p, encoder_hidden)
        attn_mask: torch.LongTensor,            # (B, T_p, T_p)
        timestamp: torch.LongTensor,            # (B, T_p)
        targets_mask: Optional[torch.LongTensor] = None,   # (B, T_p)
    ) -> torch.FloatTensor:
        x = self.in_proj(context_repr)

        if targets_mask is not None:
            mask_tok = self.pred_mask_token.expand_as(x)
            x = torch.where(targets_mask.unsqueeze(-1).bool(), mask_tok, x)

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, timestamp=timestamp)

        x = self.out_norm(x)
        x = self.out_proj(x)
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class NDT_JEPA(nn.Module):
    """I-JEPA pretraining model built on the BIT NDT encoder.

    ``forward`` signature matches the SSL path of the original NDT so the
    BIT Trainer drives it unchanged.  Returns a ``JEPAOutput`` with the
    SmoothL1 distance per masked position averaged into a scalar ``loss``.

    Hook: after every optimiser step the trainer calls
    ``post_optimizer_step()`` to EMA-update the target encoder
    (triggered automatically by Trainer when the model exposes that method).
    """

    def __init__(
        self,
        config: DictConfig,
        **kwargs,
    ):
        super().__init__()
        config = update_config(DEFAULT_CONFIG, config)
        self.config = config
        self.method = kwargs["method_name"]
        assert self.method == "jepa", (
            f"NDT_JEPA only supports method_name='jepa', got {self.method!r}"
        )

        # Accept either a string key into DATASET_CHANNELS or a direct dict.
        features = kwargs["features"]
        if isinstance(features, str):
            channel_dict = DATASET_CHANNELS[features]
        else:
            channel_dict = features

        # Optional warm-start from a pretrained NDT encoder.
        encoder_pt_path = config["encoder"].pop("from_pt", None)
        if encoder_pt_path is not None:
            enc_cfg = torch.load(os.path.join(encoder_pt_path, "encoder_config.pth"))
            config["encoder"] = update_config(config.encoder, enc_cfg)
            print(f"Warm-starting NDT_JEPA encoder from {encoder_pt_path}")

        # Online (student) encoder.
        self.encoder = NeuralEncoder(channel_dict, config.encoder)
        if encoder_pt_path is not None:
            self.encoder.load_state_dict(
                torch.load(os.path.join(encoder_pt_path, "encoder.bin")),
                strict=False,
            )

        # Target encoder — frozen copy, EMA-updated after every optimiser step.
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.target_encoder.eval()

        # Predictor.
        self.predictor = JEPAPredictor(
            encoder_hidden=self.encoder.hidden_size,
            max_F=config.encoder.embedder.max_F,
            config=config.predictor,
        )

        # EMA schedule: cosine or linear ramp from momentum_start → momentum_end.
        # Set up by the trainer via set_total_steps().
        self.momentum_start = float(kwargs.get("momentum_start", 0.996))
        self.momentum_end   = float(kwargs.get("momentum_end",   1.0))
        self.momentum_schedule = kwargs.get("momentum_schedule", "cosine")
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long))
        self._total_steps: Optional[int] = None

        # SmoothL1 loss in latent space.
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        # LayerNorm the target reps to remove magnitude DOF (standard in
        # DINO/JEPA to prevent trivial collapse).
        self.norm_target = bool(kwargs.get("norm_target", True))

    def train(self, mode: bool = True):
        """Keep target encoder always in eval mode for stable representations."""
        super().train(mode)
        self.target_encoder.eval()
        return self

    # ----- EMA bookkeeping -----

    def set_total_steps(self, n: int) -> None:
        self._total_steps = int(n)
        print(f"[NDT_JEPA] total optimiser steps for EMA schedule: {self._total_steps}")

    def _current_momentum(self) -> float:
        if self._total_steps is None or self._total_steps <= 0:
            return self.momentum_start
        s = float(self._step.item())
        T = float(self._total_steps)
        if self.momentum_schedule == "cosine":
            frac = 0.5 * (1.0 - math.cos(math.pi * min(s / T, 1.0)))
        else:
            frac = min(s / T, 1.0)
        return self.momentum_start + frac * (self.momentum_end - self.momentum_start)

    @torch.no_grad()
    def post_optimizer_step(self) -> None:
        """EMA update of target encoder. Called by the Trainer after every optimizer.step()."""
        m = self._current_momentum()
        for p_t, p_o in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            p_t.data.mul_(m).add_(p_o.data, alpha=1.0 - m)
        for b_t, b_o in zip(self.target_encoder.buffers(), self.encoder.buffers()):
            b_t.data.copy_(b_o.data)
        self._step += 1

    # ----- Forward pass -----

    def forward(
        self,
        spikes: torch.FloatTensor,
        spikes_mask: torch.LongTensor,
        spikes_timestamp: torch.LongTensor,
        targets: Optional[torch.FloatTensor] = None,
        targets_lengths: Optional[torch.LongTensor] = None,
        dataset_name: Optional[torch.Tensor] = None,
    ) -> JEPAOutput:

        # Online encoder: masking active — returns targets_mask marking which
        # patches were replaced with mask_token.
        x_online, spm_online, ts_online, targets_mask, _ = self.encoder(
            spikes, spikes_mask, spikes_timestamp,
            dataset_name=dataset_name, targets=None, force_mask=True,
        )

        # Target encoder: no masking, no grad — "true" representations.
        with torch.no_grad():
            x_target, _, _, _, _ = self.target_encoder(
                spikes, spikes_mask, spikes_timestamp,
                dataset_name=dataset_name, targets=None, force_mask=False,
            )
            if self.norm_target:
                x_target = F.layer_norm(x_target, x_target.shape[-1:])

        # Attention mask for the predictor (same convention as NeuralEncoder).
        B, T, _ = x_online.size()
        ctx_mask = self.encoder.context_mask[:T, :T].expand(B, T, T).to(x_online.device)
        self_mask = torch.eye(T, device=x_online.device, dtype=torch.int64).expand(B, T, T)
        attn_mask = self_mask | (ctx_mask & spm_online.unsqueeze(1))

        x_pred = self.predictor(
            x_online, attn_mask=attn_mask, timestamp=ts_online,
            targets_mask=targets_mask,
        )

        # Loss on positions that are both masked and non-padded.
        valid = targets_mask.bool() & spm_online.bool() if targets_mask is not None else spm_online.bool()

        per_elem = self.loss_fn(x_pred, x_target.detach())  # (B, T_p, D)
        per_pos  = per_elem.mean(dim=-1)                    # (B, T_p)

        n_valid  = valid.sum().clamp(min=1)
        loss_sum = (per_pos * valid.to(per_pos.dtype)).sum()

        return JEPAOutput(
            loss=loss_sum,
            n_examples=n_valid,
            preds=x_pred.detach(),
            targets=x_target.detach(),
            mask=valid.to(torch.long),
        )

    # ----- Checkpoint helpers -----

    def save_checkpoint(self, save_dir: str) -> None:
        """Save checkpoints compatible with ``finetune.py``.

        ``encoder.bin`` contains the *target* (EMA) encoder weights —
        following the I-JEPA recommendation that the EMA teacher
        produces better downstream representations. The online (student)
        encoder is saved as ``student_encoder.bin`` for diagnostics.
        """
        os.makedirs(save_dir, exist_ok=True)
        # Target encoder → encoder.bin (used by finetune.py)
        torch.save(self.target_encoder.state_dict(),
                   os.path.join(save_dir, "encoder.bin"))
        # Online encoder — diagnostic only
        torch.save(self.encoder.state_dict(),
                   os.path.join(save_dir, "student_encoder.bin"))
        torch.save(dict(self.config.encoder),
                   os.path.join(save_dir, "encoder_config.pth"))

        # JEPA-only artefacts (not consumed by finetune.py but needed for resume).
        torch.save(self.predictor.state_dict(),
                   os.path.join(save_dir, "predictor.bin"))
        torch.save({
            "step": int(self._step.item()),
            "total_steps": self._total_steps,
            "momentum_start": self.momentum_start,
            "momentum_end": self.momentum_end,
            "momentum_schedule": self.momentum_schedule,
        }, os.path.join(save_dir, "jepa_state.pth"))

    def load_checkpoint(self, load_dir: str, strict: bool = False) -> None:
        """Load from a JEPA checkpoint directory.

        Reads ``encoder.bin`` into the *target* encoder (that is what
        ``save_checkpoint`` now writes there). For backward-compat with
        old checkpoints that saved the online encoder, also copies into
        the student encoder.
        """
        enc_path = os.path.join(load_dir, "encoder.bin")
        enc_state = torch.load(enc_path, map_location="cpu")
        self.target_encoder.load_state_dict(enc_state, strict=strict)

        student_path = os.path.join(load_dir, "student_encoder.bin")
        if os.path.exists(student_path):
            self.encoder.load_state_dict(
                torch.load(student_path, map_location="cpu"), strict=strict
            )
        else:
            # Old-style checkpoint: encoder.bin was the student — load into both.
            self.encoder.load_state_dict(enc_state, strict=strict)

        pred_path = os.path.join(load_dir, "predictor.bin")
        if os.path.exists(pred_path):
            self.predictor.load_state_dict(
                torch.load(pred_path, map_location="cpu"), strict=strict
            )
        state_path = os.path.join(load_dir, "jepa_state.pth")
        if os.path.exists(state_path):
            st = torch.load(state_path, map_location="cpu")
            self._step = torch.tensor([st["step"]], dtype=torch.long)
            self._total_steps = st.get("total_steps", self._total_steps)
