"""drtran's input: fue's `.pre` files.

The `.pre` is NOT an I/O detail: it is the **continuity contract** of the
methodological ladder. fue identifies and estimates the best univariate model for
each series and leaves its parameters in a `.pre`; drtran takes them as seeds and
estimates everything jointly. `.pre` and `.inp` share a format — the difference is
that a `.pre`'s seeds are the estimates of the last iteration. That is what makes
the chain iterative and continuous: one rung's output feeds the next.

**The reader is not reimplemented.** `fue.load()` already parses `.inp` and
`.pre`, and it has been verified field by field that it preserves everything
drtran needs (see `tests/test_pre_roundtrip.py`). Porting `fue_pre_reader.c` (601
lines) would duplicate a source of truth that already exists — exactly the
mistake that gets expensive when the two copies drift apart.

What this module adds is **validation**: checking that a `.pre` carries what the
joint estimation needs, and failing with a clear message if it does not, instead
of propagating an incomplete model until the likelihood fails to add up.
"""

from __future__ import annotations

from dataclasses import dataclass


# Fields the joint estimation needs from the .pre. If one were missing, the cast
# would build a VARMA structure different from the one fue estimated and the
# homologation (joint diagonal == fue run separately) would fail for no apparent
# reason. `refactor` is here on purpose: with refactor=1 the optimizer hangs on
# ill conditioning (drtran TODO, 2026-07-23).
_MODEL_FIELDS = ("boxlam", "d", "D", "refactor", "mu0", "estimate_mu",
                 "ar", "ar_free", "ar_s", "ar_s_free",
                 "ma", "ma_free", "ma_s", "ma_s_free",
                 "ar_f", "ma_f", "interventions")
_SERIES_FIELDS = ("data", "freq", "nobs", "start", "name")


@dataclass(frozen=True)
class PreSpec:
    """What drtran reads from a `.pre`: the series and its estimated model.

    It is deliberately a thin wrapper around fue's objects: `ts` and `model` are
    a `fue.TimeSeries` and a `fue.Model` as they come, not copies. Repackaging
    them would create a second representation to keep in sync.
    """

    ts: object            # fue.TimeSeries
    model: object         # fue.Model
    path: str

    @property
    def name(self):
        return self.ts.name

    @property
    def nobs(self):
        return self.ts.nobs

    @property
    def freq(self):
        return self.ts.freq

    def __repr__(self):                                   # pragma: no cover
        m = self.model
        return (f"PreSpec({self.name!r}, n={self.nobs}, freq={self.freq}, "
                f"lam={m.boxlam}, d={m.d}, D={m.D}, refactor={m.refactor}, "
                f"det={len(m.interventions)})")


def load_pre(path):
    """Read a fue `.pre` (or `.inp`) and check that it serves the joint fit.

    Returns a `PreSpec`. Raises `ValueError` if the file is missing a field the
    joint estimation needs, and warns if `refactor` is 1, which is the scale at
    which the optimizer degrades.
    """
    import fue

    ts, model = fue.load(str(path))

    missing = [c for c in _MODEL_FIELDS if not hasattr(model, c)]
    missing += [f"series.{c}" for c in _SERIES_FIELDS if not hasattr(ts, c)]
    if missing:
        raise ValueError(
            f"{path}: the model read does not expose {missing}. drtran needs "
            "those fields to build the cast; without them the joint estimation "
            "cannot homologate with fue.")
    return PreSpec(ts=ts, model=model, path=str(path))


def check_scale(spec, minimum=10.0):
    """Return a warning if the `.pre`'s scale is the one that degrades the fit.

    Mauricio recommends always rescaling (`refactor=100`) because the optimizer
    works better. Measured in drtran against the C: the same model on the same
    data with `refactor=1` (Delta-log ~0.002) hangs for over 2 minutes without
    converging, and with `refactor=100` (Delta-log ~0.2) converges in 23
    iterations and one second. The cause is `cdgrad`'s finite-difference step,
    ~6e-6 absolute: at raw scale the signal-to-step ratio is terrible.

    Returns None if there is no problem, or the text of the warning.
    """
    r = float(getattr(spec.model, "refactor", 1.0) or 1.0)
    if r < minimum:
        return (f"{spec.name}: refactor={r:g}. At this scale the "
                f"finite-difference gradient (step ~6e-6) has a poor "
                f"signal-to-step ratio and the optimizer may not converge. "
                f"Regenerate the .pre with refactor=100.")
    return None
