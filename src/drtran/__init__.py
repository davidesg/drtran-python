"""drtran — Box-Jenkins transfer function models by exact maximum likelihood.

The bridge between two programs that already work:

- **fue** identifies and estimates **univariate** models and leaves one `.pre`
  per series.
- **drvarma** evaluates Mauricio's **exact VARMA likelihood** (`elf`) and
  maximises it with factored BFGS.

drtran reads the `.pre` files, builds the parametric *cast* (parameters -> VARMA
structure) and lets drvarma estimate **every parameter at once**: the transfer,
both ARMA, the deterministics, the means and the variances.

    Y_t = SUM_j omega_j(B)/delta_j(B) * B^b_j * X_j,t + N_t

Design principle, non-negotiable
--------------------------------
drvarma's `elf` is used **as it is**: not modified, not patched, not
special-cased. It is the reference implementation of the exact likelihood. Any
discrepancy with fue is a bug of drtran's cast, **never** of `elf`.

Validation criterion (the gate to everything else)
--------------------------------------------------
**Diagonal joint estimation == fue run separately.** With a diagonal structure
(diagonal AR/MA and covariance, no transfer) the exact likelihood factorises, so
the joint fit must reproduce the **sum** of the univariate ones. If it does not
match, the cast is wrong.
"""

from .cast import (Link, build_cast_spec, build_sigma, cast_diagonal,
                   compute_irf, x0_from_pre)
from .diagnose import (Adequacy, chi_test, report_adequacy,
                       transfer_adequacy)
from .embed import cast_embedded, loglik_embedded, normalize_phi0
from .estimate import Fit, fit, loglik, unpack, x0_full
from .forecast import (Forecast, error_variance, forecast,
                       forecast_mean, integrated_weights, psi_weights,
                       report_forecast, to_level, variance_decomposition)
from .identify import Identification, identify, prewhiten, report
from .netid import (Candidate, IdentifiedNetwork, identify_network,
                    report_network, residuals, write_guided)
from .network import check_acyclic, find_cycle, read_dag, write_dag
from .pre import PreSpec, check_scale, load_pre
from .slots import Slot, SlotTable, build_slots, read_cns

__version__ = "0.0.1"

__all__ = [
    "PreSpec", "load_pre", "check_scale",
    "Link", "build_cast_spec", "build_sigma", "cast_diagonal", "compute_irf",
    "x0_from_pre",
    "cast_embedded", "loglik_embedded", "normalize_phi0",
    "Adequacy", "transfer_adequacy", "report_adequacy", "chi_test",
    "Fit", "fit", "loglik", "unpack", "x0_full",
    "Identification", "identify", "prewhiten", "report",
    "Forecast", "forecast", "report_forecast", "psi_weights",
    "error_variance", "integrated_weights", "forecast_mean", "to_level",
    "variance_decomposition",
    "Candidate", "IdentifiedNetwork", "identify_network", "report_network",
    "residuals", "write_guided",
    "read_dag", "write_dag", "find_cycle", "check_acyclic",
    "Slot", "SlotTable", "build_slots", "read_cns",
    "__version__",
]
