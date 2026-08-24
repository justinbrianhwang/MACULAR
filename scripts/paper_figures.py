"""Generate the paper's figures (all PDF) into paper/figures/.

Reads only results/*.json and the datasets; every number matches FINDINGS.md.
Run: python scripts/paper_figures.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = os.path.join("paper", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "figure.dpi": 150})


def _load(name):
    with open(os.path.join("results", name), encoding="utf-8") as f:
        return json.load(f)


def _after(d, lang, key="cer"):
    return [r["cer_after"][lang][key] for r in d["per_seed"]]


# --- Fig 1: run-level divergence -------------------------------------------

def fig_divergence():
    a = _load("ocr_adapt_xfund.json")        # seeds 0,1,2
    b = _load("ocr_adapt_xfund_repro.json")  # seeds 0,3,4,5 (0 repeats)
    ja = [(s, v) for d in (a, b) for s, v in
          zip(d["seeds"], _after(d, "ja"))]

    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    xs = range(len(ja))
    vals = [v for _, v in ja]
    ax.scatter(xs, vals, s=28, color=["tab:red" if s == 0 else "tab:blue"
                                      for s, _ in ja], zorder=3)
    for x, (s, v) in zip(xs, ja):
        ax.annotate(f"seed {s}", (x, v), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7)
    # connect the two seed-0 runs
    s0 = [(x, v) for x, (s, v) in zip(xs, ja) if s == 0]
    ax.plot([s0[0][0], s0[1][0]], [s0[0][1], s0[1][1]], "--",
            color="tab:red", lw=1, zorder=2)
    ax.axhline(0.846, color="gray", lw=1, ls=":")
    ax.text(len(ja) - 0.5, 0.846, "baseline 0.846", va="bottom", ha="right",
            fontsize=7, color="gray")
    ax.set_ylabel("ja CER after adaptation")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(["run 1"] * 3 + ["run 2"] * 4)
    ax.set_title("Identical config; seed 0 twice (red): 0.099 vs 0.630",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_divergence.pdf"))
    plt.close(fig)


# --- Fig 2: gate detection rate vs n ---------------------------------------

def fig_gate():
    from scripts.eval_gate_analysis import detection_rate, _lang_pairs

    q = _load("ocr_adapt_xfund_qwen25.json")
    r = _load("ocr_adapt_xfund_r32.json")
    ns = [10, 20, 30, 50, 75, 100, 150, 200]
    subtle = [detection_rate(_lang_pairs(q["per_seed"][1]["eval_pairs"], "ja"),
                             _lang_pairs(q["eval_pairs_before"], "ja"), n)
              for n in ns]
    gross = [detection_rate(_lang_pairs(r["per_seed"][1]["eval_pairs"], "es"),
                            _lang_pairs(r["per_seed"][0]["eval_pairs"], "es"), n)
             for n in ns]

    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.plot(ns, subtle, "o-", label=r"subtle ($\Delta$CER 0.11)")
    ax.plot(ns, gross, "s-", label=r"gross ($\Delta$CER 0.39)")
    ax.axhline(0.997, color="gray", lw=0.8, ls=":")
    ax.axvline(100, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("gate size $n$ (regions)")
    ax.set_ylabel("detection rate")
    ax.set_ylim(0.8, 1.005)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_gate.pdf"))
    plt.close(fig)


# --- Fig 3: cascade under shortcut control ---------------------------------

def fig_cascade():
    # results/ocr_propagation_{default,cf}_ep80.json: 3 seeds each, per-seed
    # curves drawn faintly, seed mean bold (FINDINGS 1.2 / 4c.6).
    import json
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for name, label, mk, col in (("default_ep80", "default layouts", "s", "C0"),
                                 ("cf_ep80", "counterfactual layouts", "o", "C1")):
        d = json.load(open(f"results/ocr_propagation_{name}.json", encoding="utf-8"))
        xs = [r["cer"] for r in d["per_seed"][0]["curve"]]
        curves = [[r["val_pii_f1"] for r in s["curve"]] for s in d["per_seed"]]
        for c in curves:
            ax.plot(xs, c, "-", color=col, alpha=0.25, lw=1)
        mean = [sum(v) / len(v) for v in zip(*curves)]
        ax.plot(xs, mean, mk + "-", color=col, label=label)
    ax.set_xlabel("injected OCR CER")
    ax.set_ylabel("downstream layout F1")
    ax.set_ylim(0.45, 1.0)
    ax.legend(fontsize=8)
    ax.set_title("Flat line = model answers from geometry", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_cascade.pdf"))
    plt.close(fig)


# --- Fig 4: redaction utility-vs-leakage trade-off -------------------------

def fig_tradeoff():
    # FINDINGS 4.1b post-graph numbers
    data = {  # backbone -> mechanism -> (ctx nonlinear probe, clinical F1)
        "PaddleOCR-VL": {"none": (0.975, 0.976), "hard_mask": (0.843, 0.947),
                         "gate": (0.937, 0.917), "leace": (0.917, 0.942)},
        "Qwen2-VL-2B": {"none": (0.968, 0.899), "hard_mask": (0.859, 0.778),
                        "gate": (0.929, 0.826), "leace": (0.860, 0.890)},
        "Ministral-3": {"none": (0.975, 0.944), "hard_mask": (0.879, 0.731),
                        "gate": (0.959, 0.779), "leace": (0.896, 0.897)},
    }
    colors = {"none": "gray", "hard_mask": "tab:blue", "gate": "tab:red",
              "leace": "tab:green"}
    markers = {"PaddleOCR-VL": "o", "Qwen2-VL-2B": "s", "Ministral-3": "^"}

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for bb, mechs in data.items():
        for m, (leak, util) in mechs.items():
            ax.scatter(leak, util, c=colors[m], marker=markers[bb], s=45,
                       edgecolors="black", linewidths=0.4, zorder=3)
    ax.axvline(0.847, color="gray", lw=0.8, ls=":")
    ax.text(0.848, 0.735, "majority\nbaseline", fontsize=7, color="gray")
    from matplotlib.lines import Line2D
    leg1 = [Line2D([], [], marker="o", ls="", color=c, label=m,
                   markeredgecolor="black", markeredgewidth=0.4)
            for m, c in colors.items()]
    leg2 = [Line2D([], [], marker=mk, ls="", color="black", label=bb)
            for bb, mk in markers.items()]
    first = ax.legend(handles=leg1, loc="lower right", fontsize=7,
                      title="mechanism", title_fontsize=7)
    ax.add_artist(first)
    ax.legend(handles=leg2, loc="upper left", fontsize=7, title="backbone",
              title_fontsize=7)
    ax.set_xlabel("post-graph nonlinear probe accuracy (leakage) $\\to$ worse")
    ax.set_ylabel("clinical macro-F1 (utility)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_tradeoff.pdf"))
    plt.close(fig)


# --- Fig 5: epoch/rank ablation spread -------------------------------------

def fig_ablation():
    files = [("1 ep, r16", "ocr_adapt_xfund_ep1.json"),
             ("2 ep, r8", "ocr_adapt_xfund_r8.json"),
             ("2 ep, r16", None),  # pooled from the two main runs
             ("2 ep, r32", "ocr_adapt_xfund_r32.json")]
    pooled_r16 = (_after(_load("ocr_adapt_xfund.json"), "ja")
                  + _after(_load("ocr_adapt_xfund_repro.json"), "ja"))
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for i, (label, fn) in enumerate(files):
        vals = pooled_r16 if fn is None else _after(_load(fn), "ja")
        ax.scatter([i] * len(vals), vals, s=24, color="tab:blue", zorder=3)
    ax.axhline(0.846, color="gray", lw=0.8, ls=":")
    ax.text(3.4, 0.846, "baseline", fontsize=7, color="gray", va="bottom",
            ha="right")
    ax.set_xticks(range(len(files)))
    ax.set_xticklabels([l for l, _ in files], fontsize=8)
    ax.set_ylabel("ja CER after adaptation")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_ablation.pdf"))
    plt.close(fig)


# --- Fig 6: data samples, one crop panel per corpus/language ---------------

def fig_samples():
    from PIL import Image
    from macular.runner import _load_split
    from macular.baselines.ocr import _crop

    picks = [  # (panel title, data_dir, language, n crops)
        ("Synthetic ko", "data/meddoc_cf_engines", "ko", 3),
        ("Synthetic ja", "data/meddoc_cf_engines", "ja", 3),
        ("Synthetic en", "data/meddoc_cf_engines", "en", 3),
        ("XFUND ja (real)", "data/xfund", "ja", 3),
        ("XFUND es (real)", "data/xfund", "es", 3),
        ("XFUND zh (real)", "data/xfund_cjk", "zh", 3),
        ("FUNSD en (real)", "data/funsd", "en", 3),
    ]

    rows = []
    for title, ddir, lang, k in picks:
        docs = [d for d in _load_split({"data_dir": ddir}, "test")
                if d.language == lang]
        crops = []
        for doc in docs:
            img = Image.open(os.path.join(ddir, doc.image_path)).convert("RGB")
            for c in doc.candidates:
                t = (c.text or "").strip()
                # readable value-like fields: long enough, and containing a
                # digit, space or non-ASCII char (skips bare labels like UNIT)
                if not (6 <= len(t) <= 28):
                    continue
                if lang in ("ko", "ja", "zh"):
                    # a CJK row must actually show its script
                    if not any(ord(ch) > 127 for ch in t):
                        continue
                elif not any(ch.isdigit() or ch == " " for ch in t):
                    continue
                crop = _crop(img, c.bbox, doc.width, doc.height)
                if crop is None:
                    continue
                w, h = crop.size
                if w < 80 or not (1.5 <= w / h <= 9):
                    continue
                crops.append(crop.convert("RGB"))
                if len(crops) >= k:
                    break
            if len(crops) >= k:
                break
        rows.append((title, crops))

    ncols = max(len(c) for _, c in rows)
    fig, axes = plt.subplots(len(rows), ncols,
                             figsize=(2.1 * ncols, 0.95 * len(rows)),
                             gridspec_kw={"hspace": 0.9, "wspace": 0.15})
    for r, (title, crops) in enumerate(rows):
        for c in range(ncols):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(crops):
                ax.imshow(crops[c], aspect="equal")
            if c == 0:
                ax.set_title(title, fontsize=8, loc="left", pad=3)
    fig.savefig(os.path.join(OUT, "fig_samples.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_samples.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_divergence, fig_gate, fig_cascade, fig_tradeoff,
              fig_ablation, fig_samples):
        f()
        print("wrote", f.__name__)
