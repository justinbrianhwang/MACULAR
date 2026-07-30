"""MACULAR model core tests (require torch: pip install -e '.[model]')."""

import pytest

torch = pytest.importorskip("torch")

from macular.models import (  # noqa: E402
    MacularConfig, MacularModel, macular_loss, make_synthetic_batch, fit_synthetic,
)


def test_forward_shapes():
    cfg = MacularConfig(d_in=32, d=48, n_pii_classes=6, n_clinical=8,
                        n_heads=4, n_graph_layers=1)
    model = MacularModel(cfg)
    feats, boxes, pii, clin, mask = make_synthetic_batch(cfg, batch=2, n_regions=10)
    out = model(feats, boxes, key_padding_mask=~mask)
    assert out["pii_logits"].shape == (2, 10, 6)
    assert out["clinical_student"].shape == (2, 10, 8)
    assert out["m"].shape == (2, 10, 1)


def test_clinical_loss_backprops_through_gate_into_pii_head():
    """THE core MACULAR property (proposal 11.5): a clinical-only loss must
    reach the PII head, because z_safe depends on m = f(pii_head)."""
    cfg = MacularConfig(d_in=32, d=48, n_pii_classes=6, n_clinical=8,
                        n_graph_layers=1)
    model = MacularModel(cfg)
    feats, boxes, pii, clin, mask = make_synthetic_batch(cfg, batch=2, n_regions=10)
    out = model(feats, boxes, key_padding_mask=~mask)

    # clinical loss ONLY (no PII term)
    _, _ = macular_loss(out, pii, clin, valid_mask=mask,
                        w_pii=0.0, w_clinical=1.0, w_cons=0.0)
    loss, _ = macular_loss(out, pii, clin, valid_mask=mask,
                           w_pii=0.0, w_clinical=1.0, w_cons=0.0)
    loss.backward()

    g = model.gate.pii_head.weight.grad
    assert g is not None, "PII head received no gradient from the clinical loss"
    assert g.abs().sum().item() > 0.0, "gate did not route clinical signal to PII head"


def test_teacher_is_stop_grad_and_ema():
    cfg = MacularConfig(d_in=16, d=24, n_graph_layers=1)
    model = MacularModel(cfg)
    # teacher params never require grad
    assert all(not p.requires_grad for p in model.clinical_teacher.parameters())
    before = [p.clone() for p in model.clinical_teacher.parameters()]
    # nudge the student, then EMA-update the teacher
    for p in model.clinical_student.parameters():
        p.data.add_(1.0)
    model.update_teacher()
    moved = any(not torch.equal(b, a) for b, a in
                zip(before, model.clinical_teacher.parameters()))
    assert moved, "EMA teacher did not track the student"


def test_training_reduces_loss():
    """Wiring sanity: on a learnable synthetic signal, loss should drop."""
    cfg = MacularConfig(d_in=64, d=96, n_pii_classes=8, n_clinical=12,
                        n_graph_layers=2)
    _, history = fit_synthetic(steps=60, cfg=cfg, lr=2e-3, seed=0)
    assert history[-1] < history[0] * 0.7, (
        f"loss did not decrease enough: {history[0]:.3f} -> {history[-1]:.3f}")


def test_documents_to_batch_from_real_data():
    """OCR->features bridge: our Documents become model-core tensors, with PII
    labels correctly mapped and Hangul vs Latin producing different features."""
    from macular.models import documents_to_batch, FEATURE_DIM, config_for_features
    from macular.models.features import char_features
    from macular.data.generate import build_document
    from macular.data.pii_generators import Family
    import numpy as np

    rng = np.random.RandomState(0)
    docs = [build_document(rng, Family("A"), "en", f"d{i}", "train")
            for i in range(6)]
    feats, boxes, pii, clin, mask = documents_to_batch(docs, max_regions=40)
    assert feats.shape == (6, 40, FEATURE_DIM)
    assert (pii[mask] > 0).any() and (pii[mask] == 0).any()  # both classes present
    # script sensitivity: Korean vs English text give different char features
    assert not torch.allclose(char_features("홍길동"), char_features("John Doe"))


def test_train_core_on_real_documents_learns():
    """End-to-end: train the model core on real synthetic docs via OCR features;
    PII F1 should be well above chance."""
    from macular.models import fit_on_documents
    from macular.data.generate import build_document
    from macular.data.pii_generators import Family
    import numpy as np

    rng = np.random.RandomState(0)
    train = [build_document(rng, Family("A"), "en", f"tr{i}", "train")
             for i in range(60)]
    val = [build_document(rng, Family("C"), "en", f"va{i}", "test")
           for i in range(30)]
    _, res = fit_on_documents(train, val, epochs=40, lr=3e-3, source="gt")
    assert res["loss_history"][-1] < res["loss_history"][0] * 0.7
    assert res["val_pii_f1"]["f1"] > 0.6   # learns PII detection on real layout


def test_corrupt_text_hits_target_error_rate():
    """The OCR-error simulator must degrade text at roughly the requested rate.

    Averaged over seeds: a single draw has high variance (a 32-char string at
    p=0.3 can land anywhere in ~0.09-0.44), so assert on the mean and on
    monotonicity, not on one sample.
    """
    import numpy as np
    from macular.models import corrupt_text
    from macular.evaluation.metrics import cer as cer_metric

    text = "Patient Kim Minjun 010-2345-6789"
    assert corrupt_text(text, 0.0, np.random.RandomState(0)) == text  # no-op

    def mean_cer(p, n=20):
        return float(np.mean([
            cer_metric(text, corrupt_text(text, p, np.random.RandomState(s)))
            for s in range(n)]))

    low, mid, high = mean_cer(0.1), mean_cer(0.3), mean_cer(0.5)
    assert low < mid < high                    # monotonic in the target rate
    assert abs(low - 0.1) < 0.1                # roughly tracks the target
    assert 0.15 < mid < 0.45


def _docs(cf, n_train=100, n_val=50):
    from macular.data.generate import build_document
    from macular.data.pii_generators import Family
    import numpy as np
    rng = np.random.RandomState(0)
    train = [build_document(rng, Family("A"), "en", f"tr{i}", "train",
                            counterfactual_layout=cf) for i in range(n_train)]
    val = [build_document(rng, Family("C"), "en", f"va{i}", "test",
                          counterfactual_layout=cf) for i in range(n_val)]
    return train, val


def test_ocr_errors_degrade_pii_only_without_positional_shortcut():
    """The cascade the proposal is built on (section 2) — but it is only
    OBSERVABLE once the positional shortcut is removed.

    On the default layout PII sits in a fixed column, so the model answers from
    geometry and OCR corruption changes nothing (flat curve). On counterfactual
    layout the model must read text, and OCR errors then degrade PII F1.
    """
    from macular.models import ocr_error_propagation

    tr_d, va_d = _docs(cf=False)
    flat = ocr_error_propagation(tr_d, va_d, cer_levels=(0.0, 0.5),
                                 epochs=60, max_docs=100)
    # shortcut present -> corrupting text does essentially nothing
    assert abs(flat[0]["val_pii_f1"] - flat[-1]["val_pii_f1"]) < 0.05

    tr_c, va_c = _docs(cf=True)
    real = ocr_error_propagation(tr_c, va_c, cer_levels=(0.0, 0.5),
                                 epochs=60, max_docs=100)
    # shortcut removed -> OCR quality measurably matters
    assert real[-1]["val_pii_f1"] < real[0]["val_pii_f1"] - 0.03


def test_no_predict_all_collapse_on_counterfactual():
    """Guard the class-weighting bug: with raw inverse-frequency weights the
    model predicted PII everywhere (recall 1.0, precision = base rate) and
    ignored the text. Sqrt weighting must keep predictions discriminative."""
    from macular.models import fit_on_documents
    tr, va = _docs(cf=True)
    _, res = fit_on_documents(tr, va, epochs=80, lr=3e-3, source="gt",
                              max_docs=100)
    f1 = res["val_pii_f1"]
    assert f1["recall"] < 0.98, "model collapsed to predicting PII everywhere"
    assert f1["precision"] > 0.45, "precision at base rate => not learning text"
