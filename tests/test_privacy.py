"""Privacy adversary / leakage-measurement tests (proposal 11.9).

The leakage metric is the part of this project most able to produce a
confident-looking but meaningless number, so it is pinned down here: gradient
reversal must actually reverse, and the leakage score must be measured RELATIVE
to what the PII type alone already predicts (a plain reconstruction score is
almost entirely explained by type, which the gate keeps on purpose).
"""

import pytest

torch = pytest.importorskip("torch")

from macular.models.core import grad_reverse  # noqa: E402
from macular.models import (  # noqa: E402
    MacularConfig, MacularModel, macular_loss, make_synthetic_batch,
)
from macular.models.train import warmup  # noqa: E402


def test_gradient_reversal_flips_sign():
    x = torch.randn(4, 3, requires_grad=True)
    grad_reverse(x, 1.0).sum().backward()
    assert torch.allclose(x.grad, -torch.ones_like(x))

    x2 = torch.randn(4, 3, requires_grad=True)
    grad_reverse(x2, 0.5).sum().backward()
    assert torch.allclose(x2.grad, -0.5 * torch.ones_like(x2))


def test_adversary_is_wired_into_forward_and_loss():
    cfg = MacularConfig(d_in=32, d=48, n_pii_classes=6, n_clinical=8,
                        n_graph_layers=1)
    model = MacularModel(cfg)
    feats, boxes, pii, clin, mask = make_synthetic_batch(cfg, batch=2, n_regions=10)
    out = model(feats, boxes, key_padding_mask=~mask)
    assert out["adv_recon"].shape == (2, 10, cfg.d_recon)

    target = torch.randn(2, 10, cfg.d_recon)
    _, parts = macular_loss(out, pii, clin, valid_mask=mask, recon_target=target)
    assert parts["adversary"] > 0.0


def test_adversary_can_be_disabled():
    cfg = MacularConfig(d_in=32, d=48, n_graph_layers=1, use_adversary=False)
    model = MacularModel(cfg)
    feats, boxes, pii, clin, mask = make_synthetic_batch(cfg, batch=2, n_regions=10)
    out = model(feats, boxes, key_padding_mask=~mask)
    assert "adv_recon" not in out
    _, parts = macular_loss(out, pii, clin, valid_mask=mask)
    assert parts["adversary"] == 0.0


def test_warmup_schedule():
    assert warmup(0, 100, 1.0) == 0.0        # nothing at step 0
    assert 0.0 < warmup(15, 100, 1.0) < 1.0  # ramping
    assert warmup(100, 100, 1.0) == 1.0      # saturated


def test_hard_mask_blocks_clinical_gradient_to_pii_head():
    """The single mechanism that distinguishes MACULAR from a sequential
    pipeline: with the soft gate the clinical loss reaches the PII head; with a
    hard mask it must not."""
    def grad_norm(hard):
        cfg = MacularConfig(d_in=32, d=48, n_pii_classes=6, n_clinical=8,
                            n_graph_layers=1, hard_mask=hard)
        model = MacularModel(cfg)
        feats, boxes, pii, clin, mask = make_synthetic_batch(cfg, batch=2,
                                                             n_regions=10)
        out = model(feats, boxes, key_padding_mask=~mask)
        loss, _ = macular_loss(out, pii, clin, valid_mask=mask,
                              w_pii=0.0, w_clinical=1.0, w_cons=0.0, w_adv=0.0)
        loss.backward()
        g = model.gate.pii_head.weight.grad
        return float(g.abs().sum()) if g is not None else 0.0

    assert grad_norm(hard=False) > 0.0
    assert grad_norm(hard=True) == 0.0
