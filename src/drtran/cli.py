"""Command line interface — the same one the C binary offers.

The option letters are `drtran.c`'s own, verbatim, including the `getopt` string
and the single-dash long alias `-estwin`. That is deliberate: a command line
written for the C must run here unchanged, or the port is not a port. Anything
else — argparse's `--long-options`, a subcommand tree, a different default cast —
would make the two programs superficially similar and practically incompatible.

What is NOT here yet is refused loudly rather than ignored (`-a`, `-estwin`/`-R`,
`-C`, `-L`): a silently dropped option is how a script starts reporting numbers
that answer a different question from the one it asked.

Run `drtran -h` for the options; the text below is the entry point's plumbing.
"""

from __future__ import annotations

import getopt
import os
import sys

import numpy as np

from . import __version__

USAGE = """\
DRTRAN {version} (Python): Box-Jenkins transfer function models by exact ML

A bridge between fue (univariate models) and drvarma (exact VARMA likelihood).
Reads two models already specified in fue (.pre) and estimates them JOINTLY:

    Y_t = SUM_j [omega_j(B)/delta_j(B)] B^b_j X_j,t + N_t

and, with -n, a NETWORK of such equations (a DAG), in which a series may be at
once an output and an input.

Usage: drtran output.pre input1.pre [input2.pre ...] [options]
       The FIRST file is the output (Y); the rest are the exogenous inputs.

MODEL AND OUTPUT
  -m NAME  model name; results go to NAME.out
           (default: <output>_<input>, from the two .pre file names)
  -o FILE  write the results to FILE instead of NAME.out  ('-' for stdout)
  -Q       skip the standard errors. They cost (k^2+3k)/2 likelihood
           evaluations for k free parameters, which is the slow part on a large
           network; the estimates and the log-likelihood do not change.
  -v       verbose

TRANSFER FUNCTION  (one per input)
           nu(B) = omega(B)/delta(B) * B^b, in the Box-Jenkins convention:
             omega(B) = omega_0 - omega_1 B - ... - omega_s B^s
             delta(B) = 1     - delta_1 B - ... - delta_r B^r
  -b N     pure delay B^b                    (default: identified)
  -r N     order of the denominator delta(B) (default: identified)
  -s N     order of the numerator omega(B)   (default: identified)
           With several inputs, give a comma-separated list, one value per
           input:  -b 1,0  -s 0,1.  A single value applies to ALL inputs.
  -0       NO transfer: fit the models jointly and DIAGONALLY. This is the
           homologation mode against fue (it must reproduce fue run separately).

THE CAST
  -V       EMBED the transfer in the VARMA.  THIS IS THE DEFAULT.
  -S       SUBTRACT the transfer instead (the old cast, which truncates).

SHARED AND FIXED PARAMETERS
  -c FILE  constraints file, in the names the program prints:
             delta1[1] = phi_2[B^1]           # share
             omega1[1] = omega1[0] * theta_2[B^1]   # product
             omega2[1] = 0.0                  # fix at a value
             q[2,1]    = free                 # free an innovation covariance
           The covariances q[i,j] start out FIXED at zero: a diagonal covariance
           is the default and freeing one is a modelling decision.

THE TRANSFER NETWORK
  -n FILE  declare a NETWORK of transfers (a DAG) instead of the default star.
           One link per line, output first:  OUTPUT <- INPUT   b r s
           A cycle is rejected: the system would be simultaneous.

IDENTIFICATION
  -p       PREWHITEN ONLY: filter the input with its own ARMA, apply the same
           filter to the output, read the CCF and suggest (b, r, s).
  -i       IDENTIFY THE NETWORK from the diagonal model.
  -g NAME  GUIDED: like -i, and WRITE NAME.dag and NAME.cns ready for -n/-c.

WHAT IS ESTIMATED
  By default THE .pre RULES: every coefficient is free or fixed according to its
  flag in the file ("0.0000  0" is a FIXED coefficient, not a starting value).
  To override:
  -N       fix the noise ARMA parameters (those of Y)
  -X       fix the input ARMA parameters
  -D       fix ALL deterministic coefficients of Y
  -E       fix ALL deterministic coefficients of the inputs
  -M       fix all means at their .pre values

FORECASTING
  -f L     forecast L periods ahead, with 95% bands, in the ORIGINAL units
  -O g     forecast origin (default: the window end with -estwin, otherwise the
           end of the data). -O -1 is the CURRENT end of the data: append a new
           datum to the input .pre and it becomes the new origin, with NO
           drifting parameters.

FIXED WINDOW AND OUT-OF-SAMPLE EVALUATION
  -estwin E   estimate ONCE on observations 1..E and hold the parameters FIXED.
           (-R E is a hidden alias.) Needs -f H. The origin then rolls forward
           one datum at a time over E..n-H, comparing each forecast with what
           actually happened: MAE, RMSE and MAPE by horizon.
           The variances the model reports are THEORETICAL. This is the only way
           to decide EMPIRICALLY whether one model forecasts better than
           another -- run it on two specifications and compare.
  -C FILE  also write the per-origin errors to FILE (CSV).

NOT PORTED YET (refused rather than ignored)
  -a FILE  aggregates        -L  LaTeX report
"""

_OPTSTRING = "r:s:b:f:m:c:n:a:R:C:g:O:Lp0iXNDEMVSvho:Q"

_NOT_PORTED = {
    "-a": "aggregates (-a)",
    "-L": "the LaTeX report (-L)",
}


class CliError(Exception):
    """A bad command line, reported without a traceback."""


def _orders(text, n_inputs, what):
    """Parse `-b`/`-r`/`-s`: one value per input, or a single value for all."""
    try:
        vals = [int(t) for t in text.split(",")]
    except ValueError:
        raise CliError(f"-{what} expects integers, got {text!r}")
    if len(vals) == 1:
        vals = vals * n_inputs
    if len(vals) != n_inputs:
        raise CliError(f"-{what} has {len(vals)} values for {n_inputs} inputs")
    if any(v < 0 for v in vals):
        raise CliError(f"-{what} cannot be negative")
    return vals


def _fix_matching(table, seeds, predicate):
    """Fix every slot whose name matches, AT ITS SEED VALUE.

    By index, not by name: names repeat (two AR factors both print `phi_1[B^1]`,
    because the C numbers the power inside the factor) and `set_fixed` only
    reaches the first. A bulk switch that fixed one of two identical names would
    be worse than one that failed.
    """
    from .slots import FIXED, Slot, SlotTable

    nuevos = []
    for i, s in enumerate(table.slots):
        if predicate(s.name):
            nuevos.append(Slot(s.name, FIXED, value=float(seeds[i])))
        else:
            nuevos.append(s)
    return SlotTable(nuevos)


def _series_of(name):
    """The 1-based series index a univariate slot name belongs to, or 0.

    Names are `phi_2[B^1]`, `omega_d1[1,0]`, `mu[3]` — the index is what follows
    the underscore, or what is inside the brackets for `mu`.
    """
    if name.startswith("mu["):
        return int(name[3:name.index("]")])
    for pref in ("phi_", "theta_", "omega_d", "delta_d"):
        if name.startswith(pref):
            rest = name[len(pref):]
            digits = ""
            for ch in rest:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            return int(digits) if digits else 0
    return 0


def _apply_switches(table, seeds, fix_out_arma, fix_inp_arma,
                    fix_out_det, fix_inp_det, fix_mu):
    if fix_out_arma:
        table = _fix_matching(table, seeds, lambda n: (
            n.startswith(("phi_", "theta_")) and _series_of(n) == 1))
    if fix_inp_arma:
        table = _fix_matching(table, seeds, lambda n: (
            n.startswith(("phi_", "theta_")) and _series_of(n) > 1))
    if fix_out_det:
        table = _fix_matching(table, seeds, lambda n: (
            n.startswith(("omega_d", "delta_d")) and _series_of(n) == 1))
    if fix_inp_det:
        table = _fix_matching(table, seeds, lambda n: (
            n.startswith(("omega_d", "delta_d")) and _series_of(n) > 1))
    if fix_mu:
        table = _fix_matching(table, seeds, lambda n: n.startswith("mu["))
    return table


# ── reporting ────────────────────────────────────────────────────────────────
def _signif(p):
    """The C's significance codes, so the two reports read the same."""
    if p != p:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""


def report_fit(fit, table, names, se=None):
    """The estimated model, slot by slot.

    `se` is a `StdErrors` (or None to leave the inference columns out). They are
    computed from the Hessian recomputed AT the optimum, not from the optimiser's
    accumulated matrix — see `estimate.standard_errors` for why that distinction
    is the difference between a standard error and a number.
    """
    from .slots import ALIAS, FIXED, FREE, LINCOMB, PRODUCT

    out = ["=" * 70,
           "  JOINT ESTIMATION — exact maximum likelihood",
           "=" * 70,
           f"  series        : {', '.join(names)}",
           f"  log-likelihood: {fit.loglik:.6f}",
           f"  status        : {fit.status}  (termcode={fit.termcode}, "
           f"iterations={fit.nit})",
           ""]
    if se is None or se.ifault:
        out += ["  parameter                 estimate", "  " + "-" * 42]
    else:
        out += ["  parameter                 estimate    std.error   t-stat  p-val",
                "  " + "-" * 66]

    for i, sl in enumerate(table.slots):
        v = fit.x[i]
        if sl.kind == FIXED:
            out.append(f"  {sl.name:<20s} {v:12.6f}       (fixed)")
            continue
        if sl.kind == ALIAS:
            tail = f"       (= {table.slots[sl.pa].name})"
        elif sl.kind == PRODUCT:
            sg = "-" if sl.value < 0 else ""
            tail = (f"       (= {sg}{table.slots[sl.pa].name}"
                    f" * {table.slots[sl.pb].name})")
        elif sl.kind == LINCOMB:
            terms = []
            for sg, a, b in sl.terms:
                t = table.slots[a].name + (f" * {table.slots[b].name}"
                                           if b >= 0 else "")
                terms.append(("- " if sg < 0 else "+ ") + t)
            tail = "       (= " + " ".join(terms).lstrip("+ ") + ")"
        else:
            tail = ""

        if sl.kind != FREE:
            out.append(f"  {sl.name:<20s} {v:12.6f}{tail}")
            continue
        if se is None or se.ifault or se.se_of_slot[i] != se.se_of_slot[i]:
            out.append(f"  {sl.name:<20s} {v:12.6f}")
        else:
            out.append(f"  {sl.name:<20s} {v:12.6f} {se.se_of_slot[i]:12.6f}"
                       f" {se.t[i]:8.3f} {se.p[i]:6.4f} {_signif(se.p[i])}")

    if se is not None and not se.ifault:
        out.append("")
        out.append("  Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        out.append("  Standard errors from the Hessian recomputed at the optimum")
        out.append("  (finite differences), not from the optimiser's BFGS matrix.")
    out.append("=" * 70)
    return "\n".join(out)


def _dates(ts, origin, horizon):
    """Calendar labels for the forecast, the C's `period/year`.

    A horizon without a date is a number the reader has to place by counting,
    which is how an out-of-sample forecast gets read as an in-sample one.
    """
    start = getattr(ts, "start", None)
    freq = int(getattr(ts, "freq", 0) or 0)
    n = origin if origin is not None else ts.nobs
    if not start or freq < 1:
        return [f"{n + l}" for l in range(1, horizon + 1)]
    y0, p0 = int(start[0]), int(start[1])
    out = []
    for l in range(1, horizon + 1):
        k = (p0 - 1) + n - 1 + l                      # 0-based period counter
        out.append(f"{k % freq + 1:>2d}/{y0 + k // freq}" if freq > 1
                   else f"{y0 + k}")
    return out


def _forecast_block(fit, cast_spec, horizon, origin):
    """The forecast report, one table per series, in the C's own layout.

    LEVEL in original units, then the PERIOD and ANNUAL variations with their
    standard errors, then the residual (ERR) on the observed rows. The variations
    are differences of the TRANSFORMED level (`ystar` in the C), not of the level
    itself: with a log transformation and refactor=100 a difference of 100*log IS
    the percentage change, exactly rather than approximately.
    """
    from .forecast import forecast, to_level, variance_decomposition

    fc = forecast(fit, L=horizon, origin=origin)
    out = []

    for i, sc in enumerate(cast_spec.series):
        model = sc.spec.model
        ts = model.series
        freq = int(getattr(ts, "freq", 1) or 1)
        refc = float(getattr(model, "refactor", 1.0) or 1.0)
        lam = float(getattr(model, "boxlam", 0.0) or 0.0)
        # the C's `vscale`: in log models the variation goes in % (x100/refactor,
        # as fuf does); in levels, as a plain difference (/refactor, as forsil)
        vscale = (100.0 / refc) if abs(lam) < 1e-8 else (1.0 / refc)
        unit = "%" if abs(lam) < 1e-8 else "dif"

        level, ystar_f, ystar_h = to_level(fc, cast_spec, series=i,
                                           origin=origin, transformed=True)
        star = np.concatenate([ystar_h, ystar_f])
        nb = len(ystar_h)
        dates = _dates(ts, origin, horizon)

        se_l = fc.se("level", i)
        se_p = fc.se("diff", i)
        se_a = fc.se("annual", i) if freq > 1 and fc.var_annual is not None \
            else None

        out += ["", "=" * 78,
                f"  FORECAST REPORT — {sc.name}",
                f"  origin: observation {nb}    lead time: {horizon}"
                f"    variation in {unit}",
                "=" * 78,
                "     DATE  |     LEVEL     STD  |   PERIOD    STD  "
                "|   ANNUAL    STD",
                "  " + "-" * 64]

        # Observed rows: the last `horizon`+1, the origin included. The C also
        # carries an ERR column with the one-step residual; it is left out here
        # on purpose. `elf` returns the residuals in its own internal scale, and
        # this port has not established the factor that puts them back in the
        # series' units -- the diagnostics never needed it, because the CCF is
        # scale-invariant. Printing them unscaled would be a plausible number
        # that is wrong, which is exactly what the ERR column already was in the
        # C (see docs/PORTE.md 5.5).
        for t in range(max(1, nb - horizon), nb + 1):
            d = _dates(ts, t - 1, 1)[0]
            per = f"{vscale * (star[t - 1] - star[t - 2]):8.2f}" if t >= 2 else "       -"
            ann = (f"{vscale * (star[t - 1] - star[t - 1 - freq]):8.2f}"
                   if freq > 1 and t - 1 - freq >= 0 else "       -")
            out.append(f"  {d:>8s}  | {ts.data[t - 1]:9.2f}       - "
                       f"| {per}      - | {ann}      -")
        out.append("  " + "-" * 64)

        for l in range(horizon):
            t = nb + l                          # index into `star`
            per = vscale * (star[t] - star[t - 1])
            ann = (vscale * (star[t] - star[t - freq])
                   if freq > 1 and t - freq >= 0 else None)
            sa = f"{vscale * se_a[l]:6.2f}" if se_a is not None else "     -"
            av = f"{ann:8.2f}" if ann is not None else "       -"
            out.append(f"  {dates[l]:>8s}  | {level[l]:9.2f}  {se_l[l]:6.2f} "
                       f"| {per:8.2f} {vscale * se_p[l]:6.2f} "
                       f"| {av} {sa}")
        out.append("=" * 68)

    # ── the forecast error variance decomposition ────────────────────────────
    receivers = sorted({l.out for l in cast_spec.links})
    if receivers:
        out += ["", "=" * 78,
                "  FORECAST ERROR VARIANCE DECOMPOSITION",
                "  How much of the error of forecasting the output comes from",
                "  EACH source of innovation.",
                "=" * 78]
        for i in receivers:
            shares, why = variance_decomposition(fc, series=i)
            if shares is None:
                out += ["", f"  {cast_spec.series[i].name}: NOT REPORTED — {why}"]
                continue
            head = "    l  " + "".join(
                f"{('own noise' if j == i else cast_spec.series[j].name):>12.12s}"
                for j in range(cast_spec.m))
            out += ["", f"  {cast_spec.series[i].name}"
                        "  (% of the forecast error variance of the LEVEL)",
                    "", head, "  " + "-" * (5 + 12 * cast_spec.m)]
            for l in range(fc.L):
                out.append(f"  {l + 1:3d}  " + "".join(
                    f"{100.0 * shares[l, j]:11.1f}%" for j in range(cast_spec.m)))
        out.append("=" * 78)

    out += ["",
            "  Note: the level's s.e. is in original units; the variations are",
            "  differences of the transformed level, so with a log model they",
            "  are percentages. With a Box-Cox the level band is not symmetric."]
    return "\n".join(out)


# ── the driver ───────────────────────────────────────────────────────────────
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = ["-R" if a == "-estwin" else a for a in argv]

    try:
        # gnu_getopt, not getopt: the .pre files come FIRST on the command line,
        # and plain getopt stops at the first non-option. The C uses GNU getopt,
        # which permutes; anything else would reject every real invocation.
        opts, rest = getopt.gnu_getopt(argv, _OPTSTRING)
    except getopt.GetoptError as e:
        sys.stderr.write(f"drtran: {e}\n\n{USAGE.format(version=__version__)}")
        return 1

    o = dict(opt_b=None, opt_r=None, opt_s=None, horizon=0, origin=None,
             model_name=None, outfile=None, cons=None, net=None, guide=None,
             prewhiten_only=False, net_ident=False, no_transfer=False,
             no_stderr=False, estwin=0, rolling_csv=None,
             embed=True, verbose=False,
             fix_out_arma=False, fix_inp_arma=False, fix_out_det=False,
             fix_inp_det=False, fix_mu=False)

    for flag, arg in opts:
        if flag == "-h":
            sys.stdout.write(USAGE.format(version=__version__))
            return 0
        if flag in _NOT_PORTED:
            sys.stderr.write(
                f"drtran: {_NOT_PORTED[flag]} is not ported yet — the C binary "
                f"has it.\n        Refusing rather than ignoring the option.\n")
            return 2
        if flag == "-b":
            o["opt_b"] = arg
        elif flag == "-r":
            o["opt_r"] = arg
        elif flag == "-s":
            o["opt_s"] = arg
        elif flag == "-f":
            o["horizon"] = int(arg)
        elif flag == "-O":
            o["origin"] = int(arg)
        elif flag == "-m":
            o["model_name"] = arg
        elif flag == "-o":
            o["outfile"] = arg
        elif flag == "-c":
            o["cons"] = arg
        elif flag == "-n":
            o["net"] = arg
        elif flag == "-p":
            o["prewhiten_only"] = True
        elif flag == "-i":
            o["net_ident"] = True
        elif flag == "-g":
            o["guide"] = arg
            o["net_ident"] = True
            o["no_transfer"] = True
        elif flag == "-0":
            o["no_transfer"] = True
        elif flag == "-V":
            o["embed"] = True
        elif flag == "-S":
            o["embed"] = False
        elif flag == "-v":
            o["verbose"] = True
        elif flag == "-Q":
            o["no_stderr"] = True
        elif flag == "-R":
            o["estwin"] = int(arg)
        elif flag == "-C":
            o["rolling_csv"] = arg
        elif flag == "-N":
            o["fix_out_arma"] = True
        elif flag == "-X":
            o["fix_inp_arma"] = True
        elif flag == "-D":
            o["fix_out_det"] = True
        elif flag == "-E":
            o["fix_inp_det"] = True
        elif flag == "-M":
            o["fix_mu"] = True

    try:
        return _run(o, rest)
    except CliError as e:
        sys.stderr.write(f"drtran: {e}\n")
        return 1


def _run(o, files):
    from .cast import Link, build_cast_spec
    from .estimate import fit as estimar
    from .estimate import x0_full
    from .network import check_acyclic, read_dag
    from .pre import load_pre
    from .slots import build_slots, read_cns

    if len(files) < 2:
        raise CliError("at least two .pre files are needed (output and input); "
                       "run with -h for the options")
    for p in files:
        if not os.path.exists(p):
            raise CliError(f"cannot open {p}")

    specs = [load_pre(p) for p in files]
    names = [s.model.series.name or os.path.basename(f).rsplit(".", 1)[0]
             for s, f in zip(specs, files)]
    n_in = len(specs) - 1

    # --- the FIXED WINDOW. Only `nobs` is trimmed: the rest of the data stays
    # in the specs, which is what makes the evaluation honestly out of sample.
    full = list(specs)
    if o["estwin"]:
        nobs = specs[0].ts.nobs
        if o["estwin"] >= nobs:
            raise CliError(f"-estwin {o['estwin']} leaves no data out of sample "
                           f"({nobs} observations)")
        if o["horizon"] <= 0:
            raise CliError("-estwin needs a horizon; give -f H")
        from .evaluate import truncate
        specs = truncate(full, o["estwin"])
        sys.stderr.write(f"drtran: estimation window 1..{o['estwin']} "
                         f"(recursive evaluation to {nobs})\n")

    # ── the links ────────────────────────────────────────────────────────────
    if o["no_transfer"]:
        links = []
    elif o["net"]:
        if not os.path.exists(o["net"]):
            raise CliError(f"cannot open the network file {o['net']}")
        # read_dag already refuses a cycle; catching it here is what turns the
        # traceback into a command-line error message.
        try:
            links = read_dag(o["net"], names)
            check_acyclic(links, len(specs), names)
        except ValueError as e:
            raise CliError(str(e))
    else:
        # the star: every input feeds the first file
        if o["opt_b"] is o["opt_r"] is o["opt_s"] is None:
            links = _identify_star(specs, names, o["verbose"])
        else:
            b = _orders(o["opt_b"] or "0", n_in, "b")
            r = _orders(o["opt_r"] or "0", n_in, "r")
            s = _orders(o["opt_s"] or "0", n_in, "s")
            links = [Link(0, j + 1, b=b[j], r=r[j], s=s[j]) for j in range(n_in)]

    if o["prewhiten_only"]:
        sys.stdout.write(_prewhiten_report(specs, names) + "\n")
        return 0

    cs = build_cast_spec(specs, links=links)

    # ── the slot table: constraints and the bulk switches ────────────────────
    table = build_slots(cs)
    seeds = x0_full(cs, table)
    table = _apply_switches(table, seeds, o["fix_out_arma"], o["fix_inp_arma"],
                            o["fix_out_det"], o["fix_inp_det"], o["fix_mu"])
    if o["cons"]:
        if not os.path.exists(o["cons"]):
            raise CliError(f"cannot open the constraints file {o['cons']}")
        read_cns(o["cons"], table)
        seeds = x0_full(cs, table)

    if o["verbose"]:
        sys.stderr.write(table.report() + "\n")

    # ── estimate ─────────────────────────────────────────────────────────────
    f = estimar(cs, x0=seeds, embed=o["embed"], slots=table)
    if f.ifault:
        raise CliError(f"the likelihood could not be evaluated: ifault={f.ifault}")

    se = None
    if not o["no_stderr"]:
        from .estimate import standard_errors
        se = standard_errors(f, xitol=-1e-3)
        if se.ifault and o["verbose"]:
            sys.stderr.write("drtran: the Hessian at the optimum is not usable; "
                             "reporting without standard errors\n")
    parts = [report_fit(f, table, names, se)]

    # ── network identification (-i / -g) ─────────────────────────────────────
    if o["net_ident"]:
        from .netid import identify_network, report_network, write_guided

        red = identify_network(cs, x=f.x, embed=o["embed"])
        parts.append("")
        parts.append(report_network(red))
        if o["guide"]:
            dag, cns = write_guided(red, o["guide"])
            parts += ["", f"  wrote {dag} and {cns}",
                       f"  next:  drtran {' '.join(files)} -n {dag} -c {cns}"]

    # ── diagnostics ──────────────────────────────────────────────────────────
    if links and not o["net_ident"]:
        from .diagnose import report_adequacy, transfer_adequacy

        for k in range(len(links)):
            ad = transfer_adequacy(f, link_index=k, embed=o["embed"])
            parts.append("")
            parts.append(report_adequacy(ad))

    # ── forecast ─────────────────────────────────────────────────────────────
    if o["horizon"] > 0:
        # the origin: -O wins, then the window end, then the data end. -O -1 is
        # the CURRENT end, which is the real-time mode.
        nfull = full[0].ts.nobs
        if o["origin"] is not None and o["origin"] > 0:
            forigin = min(o["origin"], nfull)
        elif o["origin"] is not None and o["origin"] < 0:
            forigin = nfull
        elif o["estwin"]:
            forigin = o["estwin"]
        else:
            forigin = nfull

        if forigin != specs[0].ts.nobs:
            from .evaluate import truncate
            fc_cs = build_cast_spec(truncate(full, forigin), links=links)
        else:
            fc_cs = cs
        parts.append(_forecast_block(f, fc_cs, o["horizon"], None))

    # ── the rolling origin, out of sample ────────────────────────────────────
    if o["estwin"]:
        from .evaluate import (report_rolling, rolling_evaluation,
                               write_rolling_csv)
        try:
            ev = rolling_evaluation(f.x, full, links, o["estwin"],
                                    o["horizon"], embed=o["embed"])
        except ValueError as e:
            raise CliError(str(e))
        parts.append("")
        parts.append(report_rolling(ev))
        if o["rolling_csv"]:
            write_rolling_csv(ev, o["rolling_csv"])
            parts.append(f"  Per-origin errors written to {o['rolling_csv']}")

    text = "\n".join(parts) + "\n"

    target = o["outfile"]
    if target is None:
        name = o["model_name"] or f"{names[0]}_{names[1]}"
        target = f"{name}.out"
    if target == "-":
        sys.stdout.write(text)
    else:
        with open(target, "w") as fh:
            fh.write(text)
        sys.stdout.write(text)
        sys.stderr.write(f"drtran: results written to {target}\n")
    return 0


def _identify_star(specs, names, verbose):
    """Identify (b, r, s) for every input by prewhitening + CCF."""
    from .cast import Link, build_cast_spec
    from .identify import identify

    provisional = [Link(0, j, b=0, r=0, s=0) for j in range(1, len(specs))]
    cs = build_cast_spec(specs, links=provisional)
    out = []
    for lk in provisional:
        idt = identify(cs, lk)
        out.append(Link(0, lk.inp, b=idt.b, r=idt.r, s=idt.s))
        if verbose:
            sys.stderr.write(f"drtran: {names[0]} <- {names[lk.inp]}  "
                             f"identified b={idt.b} r={idt.r} s={idt.s}\n")
    return out


def _prewhiten_report(specs, names):
    from .cast import Link, build_cast_spec
    from .identify import identify, report

    provisional = [Link(0, j, b=0, r=0, s=0) for j in range(1, len(specs))]
    cs = build_cast_spec(specs, links=provisional)
    parts = []
    for lk in provisional:
        idt = identify(cs, lk)
        parts.append(report(idt, names=(names[lk.inp], names[lk.out])))
    return "\n\n".join(parts)


if __name__ == "__main__":                                 # pragma: no cover
    sys.exit(main())
