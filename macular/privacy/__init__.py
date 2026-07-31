"""Privacy mechanisms and evaluation for MACULAR.

Why this package exists, in one paragraph:

The original design put a *differentiable redaction gate* and a *GRL adversary*
in the model and measured leakage as a held-out attacker's cosine similarity to
the true region content. Three things went wrong, all measured:

1. The gate showed no privacy advantage over a hard mask (proposal gate #4).
2. The GRL adversary contributed nothing measurable.
3. Re-running an identical config flipped the SIGN of the leakage number, and a
   3-seed sweep produced a std larger than the mean (2 of 9 runs collapsed
   outright). The metric could not resolve the effect at any seed count.

(1) and (2) are a canonical failure mode: adversarial attribute removal hides an
attribute from the training-time adversary without removing it (Elazar &
Goldberg, EMNLP 2018; Gonen & Goldberg, NAACL 2019). The successor with an
actual guarantee is closed-form linear concept erasure (INLP, ACL 2020 -> RLACE,
ICML 2022 -> LEACE, NeurIPS 2023): see ``leace.py``.

(3) is a measurement problem, and it is fixed by a *discrete* attack metric.
Continuous cosine similarity drifts with training noise; exact-match recovery
rate and CER do not. See ``inversion.py`` and ``probes.py``.

IMPORTANT: linear erasure guards against LINEAR readouts only. Nonlinear probes
recover much of an "erased" concept. Any claim made with this package must be
reported alongside the nonlinear probe in ``probes.py`` — otherwise it repeats
the gate's mistake at a more sophisticated level.
"""

from .leace import LeaceEraser, fit_leace
from .probes import probe_leakage, linear_probe, mlp_probe
from .inversion import inversion_attack, prior_floor, attack_with_floor

__all__ = ["LeaceEraser", "fit_leace", "probe_leakage", "linear_probe",
           "mlp_probe", "inversion_attack", "prior_floor", "attack_with_floor"]
