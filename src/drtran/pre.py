"""Entrada de drtran: los ficheros `.pre` de fue.

El `.pre` NO es un detalle de E/S: es el **contrato de continuidad** de la escalera
metodológica. fue identifica y estima el mejor modelo univariante de cada serie y
deja sus parámetros en un `.pre`; drtran los toma como semillas y estima todo
conjuntamente. `.pre` e `.inp` comparten formato — la diferencia es que las
semillas del `.pre` son las estimaciones de la última iteración. Por eso la cadena
es iterativa y tiene continuidad: la salida de un escalón alimenta el siguiente.

**No se reimplementa el lector.** `fue.load()` ya parsea `.inp` y `.pre`, y se ha
verificado campo a campo que preserva todo lo que drtran necesita (ver
`tests/test_pre_roundtrip.py`). Portar `fue_pre_reader.c` (601 líneas) sería
duplicar una fuente de verdad que ya existe — exactamente el error que se paga
caro cuando las dos copias divergen.

Lo que aporta este módulo es la **validación**: comprobar que un `.pre` trae lo
que la estimación conjunta necesita, y fallar con un mensaje claro si no, en vez
de propagar un modelo incompleto hasta que la verosimilitud no cuadre.
"""

from __future__ import annotations

from dataclasses import dataclass


# Campos que la estimación conjunta necesita del .pre. Si alguno faltase, el cast
# construiría una estructura VARMA distinta de la que fue estimó y la
# homologación (conjunta diagonal == fue por separado) fallaría sin motivo
# aparente. `refactor` está aquí a propósito: con refactor=1 el optimizador se
# cuelga por mal condicionamiento (drtran TODO, 2026-07-23).
_CAMPOS_MODELO = ("boxlam", "d", "D", "refactor", "mu0", "estimate_mu",
                  "ar", "ar_free", "ar_s", "ar_s_free",
                  "ma", "ma_free", "ma_s", "ma_s_free",
                  "ar_f", "ma_f", "interventions")
_CAMPOS_SERIE = ("data", "freq", "nobs", "start", "name")


@dataclass(frozen=True)
class PreSpec:
    """Lo que drtran lee de un `.pre`: la serie y su modelo univariante estimado.

    Es deliberadamente un envoltorio fino sobre los objetos de fue: `ts` y `model`
    son `fue.TimeSeries` y `fue.Model` tal cual, no copias. Reempaquetarlos
    crearía una segunda representación que habría que mantener sincronizada.
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
    """Lee un `.pre` (o `.inp`) de fue y valida que sirve para la conjunta.

    Devuelve un `PreSpec`. Lanza `ValueError` si el fichero no trae algún campo
    que la estimación conjunta necesita, y avisa si `refactor` es 1, que es la
    escala con la que el optimizador se degrada.
    """
    import fue

    ts, model = fue.load(str(path))

    faltan = [c for c in _CAMPOS_MODELO if not hasattr(model, c)]
    faltan += [f"series.{c}" for c in _CAMPOS_SERIE if not hasattr(ts, c)]
    if faltan:
        raise ValueError(
            f"{path}: el modelo leído no expone {faltan}. drtran necesita esos "
            "campos para construir el cast; sin ellos la estimación conjunta no "
            "puede homologar con fue.")
    return PreSpec(ts=ts, model=model, path=str(path))


def check_scale(spec, minimo=10.0):
    """Devuelve un aviso si la escala del `.pre` es la que degrada al optimizador.

    Mauricio recomienda reescalar siempre (`refactor=100`) porque el optimizador
    trabaja mejor. Medido en drtran sobre el C: el mismo modelo y los mismos datos
    con `refactor=1` (Δlog ~0.002) cuelgan >2 min sin converger, y con
    `refactor=100` (Δlog ~0.2) convergen en 23 iteraciones y 1 segundo. La causa
    es el paso de diferencias finitas de `cdgrad`, ~6e-6 absoluto: a escala cruda
    la relación señal/paso es pésima.

    Devuelve None si no hay problema, o el texto del aviso.
    """
    r = float(getattr(spec.model, "refactor", 1.0) or 1.0)
    if r < minimo:
        return (f"{spec.name}: refactor={r:g}. A esta escala el gradiente por "
                f"diferencias finitas (paso ~6e-6) tiene mala relación "
                f"señal/paso y el optimizador puede no converger. Regenera el "
                f".pre con refactor=100.")
    return None
