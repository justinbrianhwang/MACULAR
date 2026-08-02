"""Checks for the OCR domain-adaptation experiment.

The expensive parts need a GPU and a 0.9B model, so these cover the logic that
can silently produce a wrong number: prompt masking and CER aggregation.
"""

import torch
from transformers import BatchFeature

from macular.models import ocr_adapt


class _FakeProc:
    """Tokenizes text as one id per character; images are ignored."""

    def apply_chat_template(self, msg, add_generation_prompt=True, tokenize=False):
        return ocr_adapt.PROMPT

    def __call__(self, images=None, text="", return_tensors=None):
        ids = torch.tensor([[ord(c) % 100 for c in text]])
        return BatchFeature({"input_ids": ids,
                             "pixel_values": torch.zeros(1, 3, 4, 4)})

    def batch_decode(self, out, skip_special_tokens=True):
        return ["".join(chr(int(i)) for i in out[0])]


class _FakeModel:
    device = "cpu"

    def generate(self, input_ids=None, pixel_values=None, **kw):
        # echoes the prompt then the answer; eval must slice off the prompt
        n = input_ids.shape[1] if input_ids is not None else 0
        return torch.tensor([[0] * n + [ord(c) for c in "hello"]])

    def eval(self):
        return self


def test_prompt_tokens_are_masked_from_the_loss():
    """Training on the prompt would teach the model to emit "OCR:", not to read."""
    batch = ocr_adapt._batch(_FakeProc(), _FakeModel(), None, "abc")
    prompt_len = len(ocr_adapt.PROMPT)
    assert (batch["labels"][:, :prompt_len] == -100).all()
    assert (batch["labels"][:, prompt_len:] != -100).all()


def test_cer_is_aggregated_per_language_and_wer_is_null():
    """Whitespace WER is meaningless for ko/ja/zh, so it must stay null."""
    items = [(None, "hello", "en"), (None, "hello", "ko")]
    rep = ocr_adapt.evaluate_cer(_FakeProc(), _FakeModel(), items)
    assert rep["en"]["cer"] == 0.0 and rep["ko"]["cer"] == 0.0
    assert rep["en"]["wer"] is None
    assert rep["macro"]["n_regions"] == 2


def test_halving_keeps_every_language_on_both_sides():
    """A flat halving of the language-grouped jsonl trained on ja, evaluated on
    es, and silently reported no ja row — the XFUND run that measured nothing."""
    class _Doc:
        def __init__(self, lang, i):
            self.language, self.i = lang, i

    docs = [_Doc("ja", i) for i in range(50)] + [_Doc("es", i) for i in range(50)]
    train, evald = ocr_adapt.halve_by_language(docs)
    for half in (train, evald):
        assert {d.language for d in half} == {"ja", "es"}
    assert len(train) + len(evald) == 100
    assert not ({(d.language, d.i) for d in train}
                & {(d.language, d.i) for d in evald})


def test_each_seed_gets_fresh_base_weights_and_one_baseline(monkeypatch):
    """get_peft_model injects adapters in place, so reusing the model across
    seeds would silently make seed N train on top of seed N-1 — a 6-epoch run
    wearing a 3-seed costume. The baseline must run once: no adapter is
    attached and decoding is greedy, so it cannot vary by seed."""
    loaded, trained, baselines = [], [], []

    def fake_load(device, dtype, model_id):
        m = object()
        loaded.append(m)
        return "proc", m

    def fake_eval(proc, model, items, **kw):
        baselines.append(model)
        return {"en": {"cer": 0.5, "exact_match": 0.5}, "macro": {"cer": 0.5}}

    def fake_train(proc, model, tr, ev, seed, *a, **kw):
        trained.append((seed, model))
        return fake_eval(proc, model, ev), [0.1], 7

    monkeypatch.setattr(ocr_adapt, "_load", fake_load)
    monkeypatch.setattr(ocr_adapt, "_regions",
                        lambda *a: [(None, "x", "en")])
    monkeypatch.setattr(ocr_adapt, "evaluate_cer", fake_eval)
    monkeypatch.setattr(ocr_adapt, "_train_one_seed", fake_train)

    res = ocr_adapt.finetune_and_measure([], [], "d", seeds=[0, 1, 2])

    assert [s for s, _ in trained] == [0, 1, 2]
    assert len({id(m) for _, m in trained}) == 3      # never reused
    assert len(baselines) == 1 + 3                    # one baseline + one per seed
    assert res["cer_delta_agg"]["en"]["n"] == 3


def test_cer_length_weighting_uses_gold_length():
    """A long region must not count the same as a two-character one."""
    class _Wrong(_FakeModel):
        def generate(self, input_ids=None, **kw):
            n = input_ids.shape[1] if input_ids is not None else 0
            return torch.tensor([[0] * n + [ord("x")]])

    items = [(None, "ab", "en"), (None, "a" * 20, "en")]
    rep = ocr_adapt.evaluate_cer(_FakeProc(), _Wrong(), items)
    # dominated by the long region, so well above the short one's error alone
    assert rep["en"]["cer"] > 0.8
