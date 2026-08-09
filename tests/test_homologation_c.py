"""Homologation with the C `drtran`: the port's external reference.

The values in `REF_C` come from running the compiled binary on the same `.pre`
files, and they are the strongest source of truth the port has — above any
internal invariant. They were generated with:

    ./bin/drtran tests/cases/ES_CPI_m10.pre tests/cases/WTI_ar1.pre \\
        -b B -r R -s S -S -o /tmp/out

(`-S` = the SUBTRACTING cast; `-V`, the embedded one, is the C's DEFAULT and is
also checked here.)

The C's battery (`test_battery.sh`, 296 PASS) covers the rest: synthetic truth,
pass-through with Y=X, the m6 network and the guided driver's round trip.
"""

import os
import subprocess

import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import Link, build_cast_spec  # noqa: E402
from drtran.estimate import fit, unpack  # noqa: E402

C_REPO = "/home/david/Dropbox/SRC/drtran"
CASES = os.path.join(C_REPO, "tests", "cases")
BIN = os.path.join(C_REPO, "bin", "drtran")
Y_PRE = os.path.join(CASES, "ES_CPI_m10.pre")
X_PRE = os.path.join(CASES, "WTI_ar1.pre")

# logL of the C binary, subtracting cast (-S). Measured difference with the
# port: ~1e-7.
#
# Re-homologated 2026-08-09 after the pre-sample stopped being zeroed and
# started being backcast (BUG-8 step 7, `drtran 655e255`+). Three of the four
# moved by ~1e-2. **(0,0,0) did not move at all**, and that is not luck: a
# contemporaneous transfer has a single nu weight, so the convolution is
# complete from t=1 and there is no pre-sample to fill. The EMBEDDED values
# below did not move either -- that cast never truncated -- which is the
# control that says this changed what it was meant to change.
REF_C = {
    (0, 0, 0): -736.774158,      # unchanged: no memory, no pre-sample
    (0, 1, 0): -721.727915,      # was -721.720197
    (0, 0, 1): -718.200295,      # was -718.183933
    (1, 1, 1): -756.527386,      # was -756.528944
}

# logL of the C binary with the EMBEDDED cast (-V), which is the C's DEFAULT: it
# puts the transfer inside the VARMA without subtracting, so it does not truncate
# at the start of the sample. This was blocked in Python until drvarma's modified
# Cholesky was fixed (the port used np.linalg.cholesky, the strict one, where the
# C uses choldcp).
REF_C_EMBED = {
    (0, 0, 0): -736.774158,      # no memory: agrees with the subtracting cast
    (0, 1, 0): -721.801539,
    (0, 0, 1): -718.287406,
    (1, 1, 1): -756.602851,
}
REF_DIAGONAL = -767.424341          # -0 (no transfer); also the sum of fue's
REF_OMEGA0 = 0.016002               # +/- 0.001935, t = 8.27 (the C's s.e.)

pytestmark = pytest.mark.skipif(
    not os.path.exists(Y_PRE), reason="the C drtran repo is missing")


@pytest.fixture(scope="module")
def two():
    return drtran.load_pre(Y_PRE), drtran.load_pre(X_PRE)


@pytest.mark.parametrize("brs", sorted(REF_C))
def test_the_port_reproduces_the_C_binary_by_subtraction(two, brs):
    b, r, s = brs
    Y, X = two
    f = fit(build_cast_spec([Y, X], links=[Link(0, 1, b=b, r=r, s=s)]),
            embed=False)
    assert f.ifault == 0
    assert f.loglik == pytest.approx(REF_C[brs], abs=1e-5)


@pytest.mark.parametrize("brs", sorted(REF_C_EMBED))
def test_the_port_reproduces_the_C_binary_embedded(two, brs):
    """The EMBEDDED cast, which is the C's default (-V).

    It produces a singular Phi_p by construction (the row orders differ), which
    is what uncovered the Cholesky porting bug in drvarma. With that fixed, it
    homologates to ~1e-7.
    """
    b, r, s = brs
    Y, X = two
    f = fit(build_cast_spec([Y, X], links=[Link(0, 1, b=b, r=r, s=s)]),
            embed=True)
    assert f.ifault == 0
    assert f.loglik == pytest.approx(REF_C_EMBED[brs], abs=1e-5)


def test_embedded_and_subtracting_agree_without_memory_and_differ_little_with_it(two):
    """Without memory (b=r=s=0) they agree exactly: there is no convolution to
    truncate.

    With memory they differ, but NOT in the sense that one is "better": the two
    likelihoods do not measure the same thing. The subtracting cast models the
    NOISE N = w_Y - transfer (and truncates the convolution at the start of the
    sample); the embedded one models the OBSERVED series w_Y with the transfer
    inside the VARMA. The C shows exactly the same pattern (-V below -S), so the
    difference is a property of the method, not a defect of the port.
    """
    Y, X = two
    cs0 = build_cast_spec([Y, X], links=[Link(0, 1, 0, 0, 0)])
    assert fit(cs0, embed=True).loglik == pytest.approx(
        fit(cs0, embed=False).loglik, abs=1e-6)

    for brs in ((0, 1, 0), (0, 0, 1), (1, 1, 1)):
        cs = build_cast_spec([Y, X], links=[Link(0, 1, *brs)])
        e = fit(cs, embed=True).loglik
        r = fit(cs, embed=False).loglik
        assert e != pytest.approx(r, abs=1e-6), "with memory they must differ"
        assert abs(e - r) < 1.0, "but by very little: only the pre-sample handling"
        # and each reproduces its own reference from the C
        assert e == pytest.approx(REF_C_EMBED[brs], abs=1e-5)
        assert r == pytest.approx(REF_C[brs], abs=1e-5)


def test_the_diagonal_agrees_with_the_C_and_with_fue(two):
    """A double anchor: the C's `-0` and the sum of fue's univariate likelihoods
    give the same number."""
    Y, X = two
    f = fit(build_cast_spec([Y, X]))
    assert f.loglik == pytest.approx(REF_DIAGONAL, abs=1e-4)


def test_omega_agrees_with_the_one_the_C_estimates(two):
    Y, X = two
    f = fit(build_cast_spec([Y, X], links=[Link(0, 1, b=0, r=0, s=0)]))
    omega0 = unpack(f)["links"][0][0][0]
    assert omega0 == pytest.approx(REF_OMEGA0, abs=1e-5)
    # the C reports s.e. 0.001935 -> t = 8.27; omega is very far from zero
    assert omega0 > 5 * 0.001935


@pytest.mark.skipif(not os.path.exists(BIN), reason="no compiled binary")
def test_against_the_live_binary(two, tmp_path):
    """It does not trust the tabulated values: it runs the C again and compares.

    If the C changes, this test catches it instead of carrying a stale reference.
    """
    Y, X = two
    b, r, s = 0, 1, 0
    for flag, embed in (("-S", False), ("-V", True)):
        out = tmp_path / f"c{flag}.out"
        res = subprocess.run(
            [BIN, Y_PRE, X_PRE, "-b", str(b), "-r", str(r), "-s", str(s),
             flag, "-o", str(out)],
            capture_output=True, text=True, timeout=300, cwd=C_REPO)
        line = [l for l in res.stdout.splitlines() if "Log-likelihood" in l]
        if not line:
            pytest.skip(f"the binary gave no log-likelihood: {res.stdout[-300:]}")
        ll_c = float(line[0].split(":")[1])

        f = fit(build_cast_spec([Y, X], links=[Link(0, 1, b=b, r=r, s=s)]),
                embed=embed)
        assert f.loglik == pytest.approx(ll_c, abs=1e-5), f"disagrees with {flag}"
