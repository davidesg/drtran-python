"""drtran — modelos de transferencia de Box-Jenkins por máxima verosimilitud exacta.

El puente entre dos programas que ya funcionan:

- **fue** identifica y estima modelos **univariantes** y deja un `.pre` por serie.
- **drvarma** evalúa la **verosimilitud exacta VARMA** de Mauricio (`elf`) y la
  maximiza con BFGS factorizado.

drtran lee los `.pre`, construye el *cast* paramétrico (parámetros → estructura
VARMA) y deja que drvarma estime **todos los parámetros a la vez**: la
transferencia, los dos ARMA, los deterministas, las medias y las varianzas.

    Y_t = Σⱼ ωⱼ(B)/δⱼ(B) · B^bⱼ · Xⱼ,t + N_t

Principio de diseño, no negociable
----------------------------------
El `elf` de drvarma se usa **tal cual**: no se modifica, no se parchea, no se
caso-especializa. Es la implementación de referencia de la verosimilitud exacta.
Cualquier discrepancia con fue es un bug del cast de drtran, **nunca** de `elf`.

Criterio de validación (puerta de entrada a todo lo demás)
----------------------------------------------------------
**Estimación conjunta diagonal ≡ fue por separado.** Con estructura diagonal
(AR/MA y covarianza diagonales, sin transferencia) la verosimilitud exacta se
factoriza, así que la conjunta debe reproducir la **suma** de las univariantes.
Si no coincide, el cast está mal.
"""

from .cast import (Link, build_cast_spec, build_sigma, cast_diagonal,
                   compute_irf, x0_from_pre)
from .diagnose import (Adequacy, chi_test, report_adequacy,
                       transfer_adequacy)
from .embed import cast_embedded, loglik_embedded, normalize_phi0
from .estimate import Fit, fit, loglik, unpack, x0_full
from .identify import Identificacion, identify, prewhiten, report
from .netid import (Candidato, RedIdentificada, identify_network,
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
    "Identificacion", "identify", "prewhiten", "report",
    "Candidato", "RedIdentificada", "identify_network", "report_network",
    "residuals", "write_guided",
    "read_dag", "write_dag", "find_cycle", "check_acyclic",
    "Slot", "SlotTable", "build_slots", "read_cns",
    "__version__",
]
