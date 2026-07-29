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

from .cast import Link, build_cast_spec, cast_diagonal, compute_irf, x0_from_pre
from .estimate import Fit, fit, loglik, unpack
from .pre import PreSpec, check_scale, load_pre

__version__ = "0.0.1"

__all__ = [
    "PreSpec", "load_pre", "check_scale",
    "Link", "build_cast_spec", "cast_diagonal", "compute_irf", "x0_from_pre",
    "Fit", "fit", "loglik", "unpack",
    "__version__",
]
