"""Smoke tests for the parts that run today. Run: pytest -q"""

import numpy as np

from macular.evaluation.metrics import (
    binary_metrics, document_zero_leakage_rate, cer, wer,
)
from macular.data.generate import build_document, PAGE_W, PAGE_H
from macular.data.pii_generators import Family
from macular.baselines import coordinate_only


def test_binary_metrics_basic():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 0]
    m = binary_metrics(y_true, y_pred)
    assert m["tp"] == 1 and m["fn"] == 1 and m["fp"] == 0
    assert abs(m["precision"] - 1.0) < 1e-9
    assert abs(m["recall"] - 0.5) < 1e-9
    # F2 weights recall, so F2 < F1 here (precision high, recall low)
    assert m["f2"] < m["f1"]


def test_dzlr():
    ids = ["d1", "d1", "d2", "d2"]
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 0]  # d2 leaks (missed a PII), d1 clean
    dz = document_zero_leakage_rate(ids, y_true, y_pred)
    assert dz["n_documents"] == 2
    assert dz["zero_leakage_documents"] == 1
    assert abs(dz["dzlr"] - 0.5) < 1e-9


def test_cer_wer():
    assert cer("abc", "abc") == 0.0
    assert abs(cer("abc", "abd") - 1 / 3) < 1e-9
    assert wer("a b c", "a b c") == 0.0
    assert abs(wer("a b c", "a x c") - 1 / 3) < 1e-9


def test_build_document_labels_exact():
    rng = np.random.RandomState(0)
    doc = build_document(rng, Family("A"), "en", "t-A-00000", "train")
    assert doc.width == PAGE_W and doc.height == PAGE_H
    assert len(doc.candidates) > 0
    # every bbox normalized within [0,1]
    for c in doc.candidates:
        for v in c.bbox.as_list():
            assert 0.0 <= v <= 1.0
    # there is at least one PII and one non-PII candidate
    assert any(c.is_pii for c in doc.candidates)
    assert any(not c.is_pii for c in doc.candidates)


def test_coordinate_only_runs():
    rng = np.random.RandomState(1)
    train = [build_document(rng, Family("A"), "en", f"train-A-{i:05d}", "train")
             for i in range(40)]
    test = [build_document(rng, Family("C"), "en", f"test-C-{i:05d}", "test")
            for i in range(20)]
    result = coordinate_only.run(train, test)
    assert "metrics" in result
    assert 0.0 <= result["metrics"]["f1"] <= 1.0
    assert "shortcut_detected" in result


def test_counterfactual_removes_positional_shortcut():
    """The audit must be meaningful: the hardened layout should drop the
    coordinate-only shortcut well below the detection threshold (proposal 14.4)."""
    def audit(cf):
        rng = np.random.RandomState(0)
        tr = [build_document(rng, Family("A"), "en", f"tr{i}", "train",
                             counterfactual_layout=cf) for i in range(150)]
        te = [build_document(rng, Family("C"), "en", f"te{i}", "test",
                             counterfactual_layout=cf) for i in range(80)]
        return coordinate_only.run(tr, te)
    plain = audit(False)
    cf = audit(True)
    # default layout leaks the shortcut; counterfactual should not be detected
    assert plain["shortcut_detected"] is True
    assert cf["shortcut_detected"] is False
    assert plain["metrics"]["f1"] > cf["metrics"]["f1"] + 0.2


def test_ocr_aggregate_per_language():
    from macular.baselines.ocr import aggregate
    records = [
        ("en", "abc", "abc"),      # perfect
        ("en", "abcd", "abxd"),    # 1/4 char error
        ("ko", "가나다", "가나다"),  # perfect
    ]
    out = aggregate(records)
    assert set(out["per_language"]) == {"en", "ko"}
    assert out["per_language"]["ko"]["cer_mean"] == 0.0
    assert out["per_language"]["en"]["n_regions"] == 2
    # WER is meaningless for CJK (no word spaces) -> null; defined for en
    assert out["per_language"]["ko"]["wer_mean"] is None
    assert out["per_language"]["en"]["wer_mean"] is not None
    # macro is the mean of per-language means, not blended by region count
    assert 0.0 <= out["macro"]["cer_macro_over_languages"] <= 1.0


def test_ocr_run_wiring(tmp_path):
    """Region-level plumbing: one record per non-empty candidate, grouped by
    language. Uses an injected OCR fn so no Tesseract is required."""
    from macular.data.generate import generate_dataset
    from macular.schema import read_jsonl
    from macular.baselines import ocr

    generate_dataset(str(tmp_path), n_per_split=2, languages=["en"],
                     seed=0, render_images=True)
    docs = read_jsonl(str(tmp_path / "test.jsonl"))
    calls = {"n": 0}

    def fake_ocr(crop, tess_lang):
        calls["n"] += 1
        assert tess_lang == "eng"
        return "x"

    res = ocr.run(docs, str(tmp_path), ocr_fn=fake_ocr)
    assert res.get("level") == "region"
    assert "en" in res["per_language"]
    assert res["n_regions"] == calls["n"] > 0


def test_ocr_engine_dispatch_skips_when_absent():
    """Unknown/absent engines are reported cleanly, never crash."""
    from macular.baselines import ocr
    unknown = ocr.run([], "nowhere", engine="does_not_exist")
    assert "error" in unknown and "does_not_exist" in unknown["error"]
    # easyocr is not installed in this env -> skipped, not a crash
    if not ocr.EasyOCREngine().available():
        res = ocr.run([], "nowhere", engine="easyocr")
        assert res.get("skipped") is True and res["engine"] == "easyocr"


def test_prescription_uses_real_meds():
    """Realism upgrade: prescription docs carry real RxNorm-style meds."""
    from macular.data.clinical_content import MED_LIST
    rng = np.random.RandomState(0)
    doc = build_document(rng, Family("A"), "en", "rx-A-0", "train",
                         doc_type="prescription")
    med_names = {m[0] for m in MED_LIST}
    texts = " ".join(c.text for c in doc.candidates)
    assert any(name in texts for name in med_names)
    assert any(c.clinical_type == "MED_NAME" for c in doc.candidates)
    # FHIR path references RxNorm
    assert any("rxnorm" in (c.fhir_path or "") for c in doc.candidates)


def test_lab_uses_real_loinc():
    from macular.data.clinical_content import LOINC_PANEL
    rng = np.random.RandomState(0)
    doc = build_document(rng, Family("A"), "en", "lab-A-0", "train",
                         doc_type="laboratory_report")
    assert any("loinc" in (c.fhir_path or "") for c in doc.candidates)
    names = {p[0] for p in LOINC_PANEL}
    assert any(c.text in names for c in doc.candidates)


def test_funsd_format_converter():
    """FUNSD/XFUND -> Document conversion (no network; in-memory fixture)."""
    from macular.realdata.funsd_format import form_items_to_document
    items = [
        {"box": [10, 20, 110, 60], "text": "Name", "label": "question"},
        {"box": [120, 20, 320, 60], "text": "John Doe", "label": "answer"},
        {"box": [10, 5, 300, 18], "text": "FORM TITLE", "label": "header"},
        {"box": [0, 0, 0, 0], "text": "", "label": "other"},   # skipped (empty)
    ]
    doc = form_items_to_document(items, 400, 600, "funsd-x", "en", "test",
                                 image_path="images/x.png")
    assert len(doc.candidates) == 3           # empty one dropped
    assert doc.doc_type == "form" and doc.language == "en"
    for c in doc.candidates:
        for v in c.bbox.as_list():
            assert 0.0 <= v <= 1.0
    # header -> title, question -> label, answer -> value
    blocks = {c.text: c.block_type for c in doc.candidates}
    assert blocks["FORM TITLE"] == "title"
    assert blocks["Name"] == "label"
    assert blocks["John Doe"] == "value"


def test_every_family_is_complete_and_disjoint():
    """Adding a PII family means adding it to EVERY pool.

    Families D and E were added so a concept eraser could be fitted across
    several PII value distributions (the family-A-only fit does not transfer to
    family B). A family missing from one pool dict raises KeyError only at
    document-generation time, deep inside a run, so check it here instead.
    """
    import itertools

    import numpy as np

    from macular.data.pii_generators import (
        Family, all_families, _GIVEN, _SURNAME, _ORG, _STREET,
        _PHONE_PREFIX, _ID_CENTURY,
    )

    fams = all_families()
    assert set(fams) >= {"A", "B", "C", "D", "E"}

    rng = np.random.RandomState(0)
    for name in fams:
        fam = Family(name)
        for lang in ("ko", "en", "ja"):
            # Every accessor must work for every (family, language) pair.
            for fn in (fam.full_name, fam.address, fam.organization,
                       fam.phone, fam.national_id, fam.patient_id, fam.email):
                assert fn(rng, lang)

    for pool in (_GIVEN, _SURNAME, _ORG):
        for lang in ("ko", "en", "ja"):
            sets = {f: set(pool[(f, lang)]) for f in fams}
            for a, b in itertools.combinations(fams, 2):
                assert not sets[a] & sets[b], (lang, a, b, sets[a] & sets[b])

    for a, b in itertools.combinations(fams, 2):
        assert not set(_STREET[a]) & set(_STREET[b])
    # Numeric PII must not collide either.
    assert len({_PHONE_PREFIX[f] for f in fams}) == len(fams)
    assert len({_ID_CENTURY[f] for f in fams}) == len(fams)
