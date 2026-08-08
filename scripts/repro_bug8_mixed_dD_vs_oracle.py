#!/usr/bin/env python3
"""BUG-8 — drtran and the TASTE oracle agree to 1e-6 when output and input carry
the SAME differencing, and disagree by ~2x whenever the output carries an extra
seasonal difference.

TASTE shares no code with this family (`fue`, `drvarma`, `drtran` and their
Python ports all descend from `elfvarma`/`qnewtopt`/`nlatools`) and estimates by
unconditional sum-of-squares with backforecasting rather than exact ML. It is the
only second opinion available for the transfer itself, which is precisely the
part `fue` cannot validate.

Seven cases, one input (WTI, d=1 D=0) and seven CPI outputs, all monthly
2002-01..2019-12, n=216, refactor=100, lambda=0:

  * four outputs at d=1 D=0 with deterministic harmonics   -> SAME differencing
  * three outputs at d=1 D=1 (nabla_12)                    -> DIFFERENT

The split is what this script measures. It is NOT a tolerance check: it computes
the RELATIVE disagreement, because `battery.py` compares with an ABSOLUTE
tolerance of 5e-3 and these coefficients are of order 5e-3 — so two of the three
divergent cases are reported OK there at 112 % and 69 % relative error.

Run:
    python3 scripts/repro_bug8_mixed_dD_vs_oracle.py

Requires the TASTE oracle (see `~/Dropbox/SRC/atws/Taste`), its `tbatch` tool
built, and the staged `.pre` files under `oracle/data/passthrough8/`.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

TASTE = os.path.expanduser("~/Dropbox/SRC/atws/Taste/oracle")
DATOS = os.path.join(TASTE, "data")

MATCHED = ["pt8_es", "pt8_usa", "pt8_ca", "pt8_uk"]     # output d=1 D=0
MIXED   = ["pt8_fr", "pt8_de", "pt8_emu"]               # output d=1 D=1

REL_OK   = 0.05      # matched cases must agree within 5 % relative
REL_BAD  = 0.30      # mixed cases are considered divergent above 30 %

# A case that PASSES prints one aligned line per parameter; a case that FAILS
# prints the offending parameter in a different shape. Both must be read, and
# the failing shape matters most: it is the only one battery.py flags.
LINE = re.compile(r"^\s+(\S+)\s+([-\d.]+)\s+ref\s+([-\d.]+)\s+dif")
FALLO = re.compile(r"^\s+(\S+):\s*([-\d.]+),\s*se esperaba\s*([-\d.]+)")


def run(caso: str):
    """Return {param: (taste, drtran)} for one oracle case."""
    p = subprocess.run([os.path.join(TASTE, "battery.py"),
                        "--datos", DATOS, "--caso", caso, "-v"],
                       capture_output=True, text=True, cwd=TASTE)
    out = {}
    for ln in (p.stdout + p.stderr).splitlines():
        m = LINE.match(ln) or FALLO.match(ln)
        if m:
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


def main() -> int:
    if not os.path.isdir(DATOS):
        print(f"  no encuentro {DATOS} — el oraculo no esta montado aqui")
        return 77

    print(__doc__.split("Run:")[0].strip())
    print()
    print("   case        output spec       param        TASTE     drtran"
          "     abs dif    REL dif")
    print("  " + "-" * 78)

    worst_matched, worst_mixed = 0.0, 0.0
    for grupo, casos, spec in (("MATCHED", MATCHED, "d=1 D=0 det"),
                               ("MIXED",   MIXED,   "d=1 D=1    ")):
        for caso in casos:
            vals = run(caso)
            if not vals:
                print(f"   {caso:11} {spec}   (sin salida — "
                      f"¿tbatch compilado?)")
                continue
            for k, (taste, ref) in vals.items():
                if not k.startswith("omega["):
                    continue
                rel = abs(taste - ref) / abs(ref) if ref else float("inf")
                if grupo == "MATCHED":
                    worst_matched = max(worst_matched, rel)
                else:
                    worst_mixed = max(worst_mixed, rel)
                flag = "  <--" if rel > REL_BAD else ""
                print(f"   {caso:11} {spec}   {k:11} {taste:9.6f}  "
                      f"{ref:9.6f}  {abs(taste-ref):9.1e}  "
                      f"{100*rel:7.1f} %{flag}")
        print()

    print(f"  worst relative disagreement, SAME differencing : "
          f"{100*worst_matched:6.2f} %")
    print(f"  worst relative disagreement, MIXED differencing: "
          f"{100*worst_mixed:6.2f} %")
    print()

    if worst_matched <= REL_OK and worst_mixed >= REL_BAD:
        print("  BUG-8 REPRODUCED.")
        print()
        print("  The two estimators are documented to agree between 1e-5 and")
        print("  4e-3 depending on the parameter. The matched cases sit inside")
        print("  that band; the mixed ones are three orders of magnitude")
        print("  outside it, all in the same direction (TASTE larger).")
        print()
        print("  Note what battery.py alone would have said: its tolerance is")
        print("  ABSOLUTE (5e-3), and these coefficients are of order 5e-3, so")
        print("  pt8_fr and pt8_de report OK at the relative errors above.")
        return 1

    if worst_matched > REL_OK:
        print("  Matched-differencing cases no longer agree — that is a")
        print("  DIFFERENT regression from BUG-8. Investigate before assuming")
        print("  this bug is fixed.")
        return 2

    print("  Mixed-differencing cases now agree — BUG-8 looks FIXED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
