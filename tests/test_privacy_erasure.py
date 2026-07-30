"""Tests for the erasure mechanism and the discrete attack metrics.

These exist because the gate experiment failed in two ways that a test could
have caught earlier: a mechanism claimed to protect without evidence, and a
metric too noisy to detect the effect it was measuring.
"""

import torch

from macular.privacy import fit_leace, probe_leakage, inversion_attack
from macular.privacy.probes import linear_probe, mlp_probe


def _linear_concept(n=600, d=32, k=4, seed=0):
    torch.manual_seed(seed)
    z = torch.randint(0, k, (n,))
    dirs = torch.randn(k, d)
    return torch.randn(n, d) + dirs[z] * 2.0, z


def test_leace_zeroes_the_cross_covariance():
    """The actual guarantee: Cov(erased features, concept) == 0."""
    x, z = _linear_concept()
    er = fit_leace(x, z)
    assert er.residual_covariance(x, z) < 1e-4


def test_leace_defeats_a_linear_probe():
    x, z = _linear_concept()
    before = linear_probe(x, z)
    after = linear_probe(fit_leace(x, z)(x), z)
    assert before["accuracy"] > 0.9
    assert after["accuracy_above_majority"] < 0.05


def test_leace_does_not_defeat_a_nonlinear_probe():
    """The caveat that must never be dropped from a paper claim.

    LEACE guards LINEAR readouts. Build a concept that lives in nonlinear
    structure (the sign-symmetric magnitude of a direction, which no linear map
    separates) and confirm a nonlinear probe still reads it after erasure. If
    this test ever starts failing, someone has widened the guarantee by accident
    and the reported claim must be re-checked.
    """
    torch.manual_seed(0)
    n, d = 800, 16
    z = torch.randint(0, 2, (n,))
    direction = torch.randn(d)
    x = torch.randn(n, d) * 0.5
    # class 1 sits at BOTH +2 and -2 along `direction`: linearly inseparable,
    # trivially separable by a nonlinear probe.
    sign = torch.where(torch.rand(n) < 0.5, -1.0, 1.0)
    x = x + (z.float() * sign * 2.0).unsqueeze(1) * direction

    er = fit_leace(x, z)
    xe = er(x)
    assert er.residual_covariance(x, z) < 1e-4          # guarantee holds
    after = mlp_probe(xe, z, epochs=400)
    assert after["accuracy_above_majority"] > 0.10, (
        "nonlinear probe should still recover a nonlinearly-encoded concept "
        f"after LEACE, got {after}")


def test_probe_leakage_reports_both_probes():
    x, z = _linear_concept()
    out = probe_leakage(x, z)
    assert "linear" in out and "mlp" in out
    assert out["nonlinear_leakage"] == out["mlp"]["accuracy_above_majority"]


def test_inversion_metrics_are_discrete_and_stable():
    """The reason we moved off cosine: identical inputs must give identical
    numbers, and repeated runs must not swing the way the cosine metric did."""
    torch.manual_seed(0)
    names = [f"patient-{i:03d}" for i in range(120)]
    # A representation that literally encodes the text -> attack should succeed.
    x = torch.zeros(len(names), 24)
    for i, t in enumerate(names):
        for j, c in enumerate(t[:24]):
            x[i, j] = ord(c) / 128.0

    a = inversion_attack(x, names, seed=0)
    b = inversion_attack(x, names, seed=0)
    assert a["exact_match"] == b["exact_match"]      # deterministic
    assert 0.0 <= a["cer"] <= 1.0
    assert a["cer"] < 0.5, f"attack should read a transparent encoding: {a}"


def test_inversion_fails_on_noise():
    """A representation carrying nothing must not yield recoverable text."""
    torch.manual_seed(0)
    names = [f"patient-{i:03d}" for i in range(120)]
    x = torch.randn(len(names), 24)
    out = inversion_attack(x, names, seed=0)
    assert out["exact_match"] == 0.0


def test_model_redaction_modes_run():
    from macular.models import MacularConfig, MacularModel

    torch.manual_seed(0)
    b, n, d_in = 2, 10, 24
    feats = torch.randn(b, n, d_in)
    boxes = torch.rand(b, n, 4)
    x_fit = feats.reshape(-1, d_in)
    z_fit = torch.randint(0, 4, (b * n,))
    eraser = fit_leace(x_fit, z_fit)

    for mode in ("none", "hard", "gate", "leace"):
        cfg = MacularConfig(d_in=d_in, redaction=mode)
        model = MacularModel(cfg)
        if mode == "leace":
            model.set_eraser(eraser)
        out = model(feats, boxes)
        assert out["z_safe"].shape == (b, n, cfg.d)
        assert out["pii_logits"].shape == (b, n, cfg.n_pii_classes)


def test_linear_guardedness_survives_a_linear_projector():
    """Why erasing BEFORE the projector is sound.

    LEACE zeroes Cov(r(x), z). Any linear map W keeps it zero, since
    Cov(W r(x), z) = W Cov(r(x), z) = 0. So the guarantee still holds at the
    projector output. It does NOT survive a nonlinearity — which is why the
    relation-graph output is attacked separately (ctx_* in erasure_comparison).
    """
    x, z = _linear_concept(n=800, d=24)
    xe = fit_leace(x, z)(x)
    proj = torch.nn.Linear(24, 64)
    with torch.no_grad():
        projected = proj(xe)
    assert linear_probe(projected, z)["accuracy_above_majority"] < 0.05


def test_relation_graph_output_is_exposed_for_attack():
    """The deployed representation must be attackable, not just the gate output."""
    from macular.models import MacularConfig, MacularModel

    torch.manual_seed(0)
    model = MacularModel(MacularConfig(d_in=16))
    out = model(torch.randn(2, 6, 16), torch.rand(2, 6, 4))
    assert "z_ctx_safe" in out
    assert out["z_ctx_safe"].shape == out["z_safe"].shape


def test_leace_path_is_differentiable():
    """The erasure must not silently cut gradients to the backbone."""
    from macular.models import MacularConfig, MacularModel

    torch.manual_seed(0)
    d_in = 24
    feats = torch.randn(1, 8, d_in, requires_grad=True)
    eraser = fit_leace(torch.randn(200, d_in), torch.randint(0, 3, (200,)))
    model = MacularModel(MacularConfig(d_in=d_in, redaction="leace"))
    model.set_eraser(eraser)
    model(feats, torch.rand(1, 8, 4))["z_safe"].sum().backward()
    assert feats.grad is not None and feats.grad.abs().sum() > 0
