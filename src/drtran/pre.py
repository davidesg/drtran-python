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


def write_pre(fit, series=0, path=None, std_errors=None):
    """Write back a `.pre` with the JOINTLY re-estimated univariate block.

    fue leaves a `.pre`, drtran reads it as a seed and re-estimates everything
    together — and the univariate blocks **move** while it does: on the canonical
    case the two `omega_d1` go from their univariate seeds to -0.040867 and
    -0.094588 once the transfer is fitted beside them. This writes that back.

    **What it is not.** It is not a better starting point. Measured on the
    canonical case, the written `.pre` evaluates at -772.840628 on the diagonal
    where the original evaluates at -767.424341 — WORSE, and necessarily so: the
    blocks written here are optimal *with the transfer in the model*, and the
    diagonal's optimum is by definition fue's separate estimates. That is the
    gate this whole port is built on. Both starting points reach the same joint
    optimum (-718.287406) in the same number of iterations.

    **What it is for**, then:

    * carrying the estimates into a MODIFIED specification — add a lag, free a
      covariance — so the next model starts from the best current description of
      each series rather than from the pre-transfer one;
    * handing the joint estimates back to fue and fuf, since the result is a
      valid `.pre` and those programs can read, plot and forecast from it;
    * the record: a `.pre` is the human-readable statement of what was estimated.

    What is NOT written is the transfer. A `.pre` is a UNIVARIATE file: it has
    room for the series' ARMA, deterministics and mean, and nowhere to put
    omega(B)/delta(B). That is the design, not a gap — the network is declared
    separately in the `.dag` and the `.cns`, and `x0_from_pre` deliberately
    starts the transfers at zero.

    `fue.report.write_pre` wants a fitted `fue.Model`, i.e. one carrying a
    `_result` with `.params` and `.std_errors` walked in `count_npar_build_par`
    order. drtran's model was not fitted by fue, so that is built here from the
    joint fit. The order needs no translation: `build_slots` is already an exact
    mirror of `fue.cast_us._build_initial_x`, which is the same enumeration.

    `std_errors` defaults to computing them, which is the expensive part; pass a
    `StdErrors` to reuse one. They are required — `_extract_fitted` reads them —
    so this cannot be done on a fit whose Hessian was refused.
    """
    import copy
    from types import SimpleNamespace

    import numpy as np

    from .estimate import standard_errors, unpack

    cs = fit.cast_spec
    sc = cs.series[series]
    if path is None:
        raise ValueError("write_pre needs a path")

    if std_errors is None:
        std_errors = standard_errors(fit)
    if std_errors.ifault:
        raise RuntimeError(
            "the standard errors are not available (the Hessian at this point "
            f"is not usable: ifault={std_errors.ifault}), and a .pre carries "
            "them; re-estimate before writing one")

    params = np.asarray(unpack(fit)["series"][series], float)
    off = cs.npar_links + sum(s.npar for s in cs.series[:series])
    se = np.asarray(std_errors.se_of_slot[off:off + sc.npar], float)
    # a slot the `.cns` fixed has no standard error; the `.pre` still needs a
    # number in the column, and zero is the honest one for a fixed parameter
    se = np.nan_to_num(se, nan=0.0)

    model = copy.deepcopy(sc.spec.model)
    model._result = SimpleNamespace(params=params, std_errors=se)
    model.write_pre(str(path))
    return str(path)


def next_pre_path(source, suffix=1):
    """`ES_CPI_m10.pre` -> `ES_CPI_m10.1.pre`, fue's iteration convention.

    Deliberately never the source path: overwriting the input of the run that
    produced it destroys the only record of where the estimates came from, and
    makes the run unrepeatable.
    """
    import os

    base = os.path.basename(str(source))
    stem = base[:-4] if base.lower().endswith(".pre") else base
    return os.path.join(os.path.dirname(str(source)) or ".",
                        f"{stem}.{suffix}.pre")
