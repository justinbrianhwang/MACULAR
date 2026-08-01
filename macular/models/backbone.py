"""Backbone adapters (proposal 11.2 / C1).

Real backbones (Qwen3-VL, Ministral-3, Llama-3.2-Vision) implement ``encode`` to
turn a page image + region proposals into per-region features of dim ``d_in``.
That step needs the actual VLM weights and a GPU, so it is the integration point
left for the co-author machine.

``MockBackboneAdapter`` produces features on CPU so the whole core trains and is
tested without any VLM. It also powers a *learnable* synthetic task: it plants
the ground-truth PII/clinical signal into a few feature dims (plus noise) so a
correctly-wired model can actually drive the loss down.
"""

from __future__ import annotations

from typing import Protocol

import torch

from .core import MacularConfig


class BackboneAdapter(Protocol):
    def encode(self, images, boxes) -> torch.Tensor:
        """Return per-region features, shape (B, N, d_in)."""
        ...


class MockBackboneAdapter:
    """CPU stand-in for a VLM backbone. No real weights; deterministic."""

    def __init__(self, cfg: MacularConfig):
        self.cfg = cfg

    def encode(self, images, boxes) -> torch.Tensor:  # pragma: no cover - trivial
        B, N = boxes.shape[:2]
        return torch.randn(B, N, self.cfg.d_in)


def make_synthetic_batch(cfg: MacularConfig, batch=4, n_regions=24, seed=0):
    """A learnable synthetic batch: region features encode the labels in a few
    dims (plus noise), so training should reduce the loss — proving the wiring.

    Returns (region_feats, boxes, pii_labels, clinical_labels, valid_mask).
    """
    g = torch.Generator().manual_seed(seed)
    B, N = batch, n_regions

    pii_labels = torch.randint(0, cfg.n_pii_classes, (B, N), generator=g)
    clinical_labels = torch.randint(0, cfg.n_clinical, (B, N), generator=g)

    feats = 0.3 * torch.randn(B, N, cfg.d_in, generator=g)
    # plant one-hot-ish signal for pii label in the first block of dims...
    for c in range(cfg.n_pii_classes):
        feats[..., c] += (pii_labels == c).float()
    # ...and for clinical label in the next block.
    off = cfg.n_pii_classes
    for c in range(cfg.n_clinical):
        if off + c < cfg.d_in:
            feats[..., off + c] += (clinical_labels == c).float()

    boxes = torch.rand(B, N, 4, generator=g)
    valid_mask = torch.ones(B, N, dtype=torch.bool)
    return feats, boxes, pii_labels, clinical_labels, valid_mask
