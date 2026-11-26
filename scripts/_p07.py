"""Lo comun de los repro_p07_*: los datos de los casos de clase, el binario en
C de drtran y el codigo de salida.

LOS DATOS. Los casos de clase son `ES_CPI_m10.1.pre` y `WTI_ar1.1.pre`, que en
la raiz del repo estan ignorados por git (`*.1.pre`): en un clon limpio no
existen. Sus copias van en `tests/data/p07/` (es_cpi_m10_clase.pre,
wti_ar1_clase.pre) y los repro las abren por ruta absoluta (ES, WTI), desde
cualquier directorio. `pre(nombre)` da la ruta de un `.pre` por su nombre de
clase, para los repro que lo copian a una carpeta temporal con ese nombre.

EL BINARIO. Se busca junto a este repositorio, o bajo ATSW_SRC, igual que
tests/_rutas.py; DRTRAN_C manda si esta puesto.

EL CODIGO DE SALIDA. Cada repro sale con 1 mientras el defecto exista y con 0
cuando esta arreglado. Si no puede ni correr (un fichero que falta, el binario
que no esta, una excepcion al preparar el caso), sale con 2: eso no dice nada
del defecto, y un 1 por esa razon pareceria un defecto abierto.
"""
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get("ATSW_SRC") or os.path.dirname(ROOT)
DATOS = os.path.join(ROOT, "tests", "data", "p07")

ES = os.path.join(DATOS, "es_cpi_m10_clase.pre")
WTI = os.path.join(DATOS, "wti_ar1_clase.pre")
_CLASE = {"ES_CPI_m10.1.pre": ES, "WTI_ar1.1.pre": WTI}

NO_CORRE = 2


def _no_corre(tipo, valor, tb):
    traceback.print_exception(tipo, valor, tb)
    sys.stderr.write("\nEL REPRO NO HA PODIDO CORRER (sale con 2): esto no dice "
                     "nada del defecto.\n")
    sys.stderr.flush()
    sys.stdout.flush()
    os._exit(NO_CORRE)


sys.excepthook = _no_corre


def binario():
    cands = [os.environ.get("DRTRAN_C", ""),
             os.path.join(SRC, "atsw-gui", "engines", "drtran", "bin", "drtran"),
             os.path.join(SRC, "drtran", "bin", "drtran")]
    for c in cands:
        if c and os.path.exists(c):
            return c
    sys.stderr.write("No encuentro el binario de drtran: pon DRTRAN_C=<ruta>.\n")
    sys.exit(NO_CORRE)


def pre(nombre):
    return _CLASE.get(nombre, os.path.join(ROOT, nombre))
