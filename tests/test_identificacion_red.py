"""Identificación de la RED: leer las CCF de los residuos del diagonal.

Es el paso 3 de la escalera. Lo que se comprueba aquí es que el puerto lee las
mismas CCF que el C y propone lo mismo — no que la propuesta sea buena, que es
juicio del analista.

**Ojo al comparar con el binario:** `-i` a secas NO identifica desde el modelo
diagonal. Monta su propio modelo (en m6: 61 slots, 46 libres, logL −1716.36, con
Σ diagonal y una transferencia por defecto) y lee las CCF de ESOS residuos. Para
comparar hay que pedirle `-0 -i` con las mismas restricciones. Confundir los dos
ajustes hace que las cifras no cuadren y parezca un fallo del puerto.
"""

import os
import re
import subprocess

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import build_cast_spec  # noqa: E402
from drtran.netid import (_bs_de_un_lado, identify_network,  # noqa: E402
                          report_network, residuals, write_guided)
from drtran.slots import build_slots, read_cns  # noqa: E402

REPO = "/home/david/Dropbox/SRC/drtran"
DATA = os.path.join(REPO, "tests/data/m6")
BIN = os.path.join(REPO, "bin/drtran")
M6 = ["EP", "EI", "EU", "EC", "EA", "P"]

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA, "M6_EP.pre")),
    reason="faltan los .pre de m6 del repo C")


def _cs(nombres=None):
    return build_cast_spec([drtran.load_pre(os.path.join(DATA, f"M6_{n}.pre"))
                            for n in (nombres or M6)])


# ── el bloque contiguo ───────────────────────────────────────────────────────
def test_la_estructura_es_el_bloque_contiguo_no_los_picos_sueltos():
    """Un pico aislado lejos del bloque es ruido: con bandas al 5 % se espera
    uno de cada veinte fuera por azar. Es la misma decisión que en la
    identificación bivariante, y la que evita disparates."""
    c = np.zeros(13)
    c[2] = 0.40; c[3] = 0.35          # bloque en k = 2..3
    c[9] = 0.45                        # pico aislado, MÁS alto
    b, s, _pico = _bs_de_un_lado(c, 12, 0.25)
    assert (b, s) == (2, 1), "el bloque es el contiguo desde b, no hasta el pico"


def test_sin_nada_significativo_no_hay_enlace():
    assert _bs_de_un_lado(np.full(13, 0.05), 12, 0.25) is None


# ── los residuos ─────────────────────────────────────────────────────────────
def test_los_residuos_los_da_elf_no_un_filtro_a_mano():
    """`elf` con `atf=True`: son los residuos EXACTOS, con su inicialización
    pre-muestral, no una aproximación truncada."""
    cs = _cs()
    a, ifault = residuals(drtran.x0_from_pre(cs), cs)
    assert ifault == 0
    assert a.shape == (64, 6)
    assert np.all(np.isfinite(a))


# ── homologación con el binario ──────────────────────────────────────────────
def _identificacion_del_C(out):
    """(covarianzas, enlaces) del bloque de identificación de un `.out`."""
    txt = open(out).read()
    cov = [(m[1], m[2], float(m[3])) for m in
           re.finditer(r"^\s+(\w+)\s+-\s+(\w+)\s+r\(0\) = ([+-][\d.]+)", txt, re.M)]
    enl = [(m[1], m[2], float(m[3]), int(m[4]), int(m[5])) for m in
           re.finditer(r"^\s+(\w+)\s+->\s+(\w+)\s+peak ([+-][\d.]+)\s+"
                       r"proposal\s+b=(\d+) r=0 s=(\d+)", txt, re.M)]
    return cov, enl


@pytest.mark.skipif(not os.path.exists(BIN), reason="falta el binario C")
def test_la_red_identificada_homologa_con_el_binario(tmp_path):
    """El caso de referencia: m6 entero, identificando desde el MISMO diagonal.

    Se compara contra el binario en vivo. La comparación es exigente a
    propósito: los mismos pares, en el mismo orden, con los mismos picos y las
    mismas propuestas de (b, s).
    """
    pre = [os.path.join(DATA, f"M6_{n}.pre") for n in M6]
    out = str(tmp_path / "i.out")
    r = subprocess.run([BIN, *pre, "-0", "-i", "-c", os.path.join(DATA, "m6.cns"),
                        "-o", out], capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr
    ll_C = float(re.search(r"^Log-likelihood\s*=\s*(-?\d+\.\d+)",
                           open(out).read(), re.M).group(1))
    assert ll_C == pytest.approx(-1709.511575, abs=1e-5), "cambió el binario C"
    cov_C, enl_C = _identificacion_del_C(out)
    assert cov_C and enl_C, "no se pudo leer la identificación del .out"

    # el puerto, EN EL MISMO PUNTO: se mapean los parámetros que reporta el C
    from collections import Counter, defaultdict
    pares, dentro = [], False
    for ln in open(out):
        if "Parameter" in ln and "Estimate" in ln:
            dentro = True
            continue
        if dentro:
            if ln.startswith("---") and pares:
                break
            if "(fixed by .pre" in ln:
                continue
            mm = re.match(r"^([A-Za-z_][^ ]*)\s+(-?\d+\.\d+)", ln)
            if mm:
                pares.append((mm.group(1), float(mm.group(2))))
    porn = defaultdict(list)
    for n, v in pares:
        porn[n].append(v)
    cs = _cs()
    t = build_slots(cs)
    read_cns(os.path.join(DATA, "m6.cns"), t)
    uso, x = Counter(), np.zeros(len(t))
    for i, s in enumerate(t.slots):
        vs = porn.get(s.name)
        assert vs, f"el C no reporta {s.name}"
        k = uso[s.name]; uso[s.name] += 1
        x[i] = vs[k] if k < len(vs) else vs[-1]

    red = identify_network(cs, x=x)
    nb = red.nombres

    assert red.banda == pytest.approx(2.0 / 8.0)      # n = 64
    assert red.nlags == 8                              # 2 × freq trimestral

    assert [(nb[i], nb[j], round(r0, 3)) for i, j, r0 in red.covarianzas] == \
           [(a, b, round(v, 3)) for a, b, v in cov_C]

    assert [(nb[c.inp], nb[c.out], round(c.pico, 3), c.b, c.s)
            for c in red.enlaces] == \
           [(a, b, round(v, 3), bb, ss) for a, b, v, bb, ss in enl_C]


def test_el_modo_guiado_escribe_el_BORRADOR_ciclo_incluido(tmp_path):
    """El `.dag` y el `.cns` se escriben sin podar, y el ciclo se ANOTA.

    En m6 la propuesta sale cíclica (EP → EC → EA → EP): leer las CCF par a par
    no impone aciclicidad. No es un fallo — es la parte del trabajo que le toca
    al analista. Esconderlo o podar por cuenta propia sería peor: el fichero es
    el borrador sobre el que se decide.

    Las covarianzas van con índice NUMÉRICO, no con nombre: el `.cns` no lee
    nombres en las `q`, y el propio `-i` del C las imprime con nombre bajo un
    rótulo que dice «pégalo en un fichero -c» — así no se puede pegar.
    """
    from drtran.network import read_dag

    cs = _cs()
    red = identify_network(cs, x=drtran.x0_from_pre(cs))
    assert red.ciclo is not None, "en m6 la propuesta cruda es cíclica"

    dag, cns = write_guided(red, str(tmp_path / "guia"))
    cabecera = open(dag).read()
    assert "CICLO" in cabecera and "Poda" in cabecera

    with pytest.raises(ValueError, match="ciclo"):
        read_dag(dag, cs.names)                # y el lector no lo deja pasar

    # Podar: quitar el enlace de pico menor de cada ciclo hasta que sea un DAG.
    # Hace falta más de una pasada — en m6 hay varios ciclos —, que es
    # justamente por lo que esto no lo hace la librería: decidir cuál cae es
    # juicio, no aritmética. Aquí se poda por el pico sólo para probar el ciclo
    # completo del método.
    podados = 0
    while red.ciclo is not None:
        ciclo = set(red.ciclo)
        fuera = min((c for c in red.enlaces
                     if c.out in ciclo and c.inp in ciclo),
                    key=lambda c: abs(c.pico))
        red.enlaces.remove(fuera)
        podados += 1
        assert podados < 10, "podando en círculos"
    assert podados >= 2, "en m6 hace falta podar más de un enlace"
    dag2, cns2 = write_guided(red, str(tmp_path / "podada"))

    links = read_dag(dag2, cs.names)
    assert len(links) == len(red.enlaces)
    cs2 = build_cast_spec([s.spec for s in cs.series], links=links)
    t = build_slots(cs2)
    n0 = t.n_free
    assert read_cns(cns2, t) == len(red.covarianzas)
    assert t.n_free == n0 + len(red.covarianzas)
    assert "q[" in open(cns2).read() and "q[EI" not in open(cns2).read()


def test_el_informe_dice_que_es_una_guia():
    """La red identificada es una GUÍA, no la final. Que el informe lo diga no
    es cortesía: es la doctrina de la escuela, y quitarlo invita a estimar el
    borrador."""
    cs = _cs()
    txt = report_network(identify_network(cs, x=drtran.x0_from_pre(cs)))
    assert "GUÍA" in txt and "podar" in txt
    assert "exogeneidad" in txt and "aciclicidad" in txt
