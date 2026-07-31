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
  -O g     forecast origin (default: the end of the data)

NOT PORTED YET (refused rather than ignored)
  -a FILE  aggregates        -estwin E / -R E  fixed-window estimation
  -C FILE  rolling errors    -L               LaTeX report
"""

_OPTSTRING = "r:s:b:f:m:c:n:a:R:C:g:O:Lp0iXNDEMVSvho:"

_NOT_PORTED = {
    "-a": "aggregates (-a)",
    "-R": "fixed-window estimation (-estwin/-R)",
    "-C": "rolling out-of-sample errors (-C)",
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
def report_fit(fit, table, names):
    """The estimated model, slot by slot.

    No standard errors: this port does not compute the Hessian yet, and printing
    a column of blanks is more honest than printing one the reader would take for
    an inference. It is on the TODO.
    """
    out = ["=" * 64,
           "  JOINT ESTIMATION — exact maximum likelihood",
           "=" * 64,
           f"  series      : {', '.join(names)}",
           f"  log-likelihood: {fit.loglik:.6f}",
           f"  status      : {fit.estado}  (termcode={fit.termcode}, "
           f"iterations={fit.nit})",
           "",
           "  parameter                     estimate    ",
           "  " + "-" * 46]
    from .slots import FIXED, FREE

    for i, s in enumerate(table.slots):
        if s.kind == FREE:
            det = ""
        elif s.kind == FIXED:
            det = "  (fixed)"
        else:
            det = "  (constrained)"
        out.append(f"  {s.name:<24s} {fit.x[i]:14.6f}{det}")
    out.append("=" * 64)
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
    from .forecast import forecast, to_level

    fc = forecast(fit, L=horizon, origin=origin)
    out = []
    for i, sc in enumerate(cast_spec.series):
        nivel = to_level(fc, cast_spec, serie=i, origin=origin)
        se = fc.se("level", i)
        fechas = _dates(sc.spec.model.series, origin, horizon)
        out += ["", "=" * 64,
                f"  FORECAST — {sc.name}  (original units)",
                "=" * 64,
                "   h   date       forecast       s.e.       95% interval",
                "  " + "-" * 60]
        for l in range(horizon):
            v, e = nivel[l], se[l]
            out.append(f"  {l + 1:3d}  {fechas[l]:>8s}  {v:12.4f}  {e:9.4f}   "
                       f"[{v - 1.96 * e:10.4f}, {v + 1.96 * e:10.4f}]")
        out.append("=" * 64)
    out.append("")
    out.append("  Note: the s.e. is that of the MODELLED scale; with a Box-Cox")
    out.append("  transformation the band is not symmetric around the level.")
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

    partes = [report_fit(f, table, names)]

    # ── network identification (-i / -g) ─────────────────────────────────────
    if o["net_ident"]:
        from .netid import identify_network, report_network, write_guided

        red = identify_network(cs, x=f.x, embed=o["embed"])
        partes.append("")
        partes.append(report_network(red))
        if o["guide"]:
            dag, cns = write_guided(red, o["guide"])
            partes += ["", f"  wrote {dag} and {cns}",
                       f"  next:  drtran {' '.join(files)} -n {dag} -c {cns}"]

    # ── diagnostics ──────────────────────────────────────────────────────────
    if links and not o["net_ident"]:
        from .diagnose import report_adequacy, transfer_adequacy

        for k in range(len(links)):
            ad = transfer_adequacy(f, link_index=k, embed=o["embed"])
            partes.append("")
            partes.append(report_adequacy(ad))

    # ── forecast ─────────────────────────────────────────────────────────────
    if o["horizon"] > 0:
        partes.append(_forecast_block(f, cs, o["horizon"], o["origin"]))

    texto = "\n".join(partes) + "\n"

    destino = o["outfile"]
    if destino is None:
        nombre = o["model_name"] or f"{names[0]}_{names[1]}"
        destino = f"{nombre}.out"
    if destino == "-":
        sys.stdout.write(texto)
    else:
        with open(destino, "w") as fh:
            fh.write(texto)
        sys.stdout.write(texto)
        sys.stderr.write(f"drtran: results written to {destino}\n")
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
    partes = []
    for lk in provisional:
        idt = identify(cs, lk)
        partes.append(report(idt, nombres=(names[lk.inp], names[lk.out])))
    return "\n\n".join(partes)


if __name__ == "__main__":                                 # pragma: no cover
    sys.exit(main())
