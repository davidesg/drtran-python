"""The common window, inferred from what the `.pre` already carries.

`nobs`, `start` and `freq` are enough, so the window comes out as a DATE range.
This is NOT the problem `check_alignment` refuses: two series can end on the
same date — the alignment right, the fit legitimate — while one has more
history. The cast then trims, and the longer one's `.pre` stops being an optimum
of the sample being fitted. See `docs/PASSTHROUGH_MEG_BANK.md`.
"""

import os
import warnings

import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import build_cast_spec, common_window  # noqa: E402

SF = "/home/david/Dropbox/SF_MEG/empirical"
PT8 = "/home/david/Dropbox/SRC/atws/Taste/oracle/data/passthrough8"

pytestmark = pytest.mark.skipif(
    not os.path.exists(SF), reason="the SF_MEG bank is missing")


@pytest.fixture(scope="module")
def largo():
    return drtran.load_pre(os.path.join(SF, "cases/ES_CPI/work/ES_CPI_m00.pre"))


@pytest.fixture(scope="module")
def corto():
    return drtran.load_pre(os.path.join(SF, "passthrough/WTI_2005.pre"))


def test_the_window_comes_out_as_dates_and_names_who_has_spare(largo, corto):
    """216 from 2002 against 180 from 2005, both ending 12/2019."""
    ini, fin, n, spare = common_window([largo, corto])
    assert ini == (2005, 1)
    assert fin == (2019, 12)
    assert n == 180
    assert spare == {0: 36}                 # the output carries 36 extra months


def test_series_of_the_same_window_have_no_spare(corto):
    otro = drtran.load_pre(os.path.join(SF, "cases/DE_CORE/work/DE_CORE_m00.pre"))
    ini, fin, n, spare = common_window([otro, corto])
    assert (ini, fin, n) == ((2005, 1), (2019, 12), 180)
    assert spare == {}


def test_the_mismatch_warns_and_the_match_does_not(largo, corto):
    """It must fire — this is the case that passed every check and was wrong.

    And it must stay silent on the matched pair, or it becomes noise and gets
    turned off.
    """
    otro = drtran.load_pre(os.path.join(SF, "cases/DE_CORE/work/DE_CORE_m00.pre"))
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        build_cast_spec([largo, corto])
    msgs = [str(w.message) for w in ws if issubclass(w.category, RuntimeWarning)]
    assert any("more history" in m and "01/2005-12/2019" in m for m in msgs)

    with warnings.catch_warnings(record=True) as ws2:
        warnings.simplefilter("always")
        build_cast_spec([otro, corto])
    assert not [w for w in ws2 if issubclass(w.category, RuntimeWarning)]
