"""La RED de transferencias: el `.dag`, la tabla de slots y `expand_params`.

Tres cosas distintas, y las tres tienen que homologar con el binario C:

1. **el `.dag`** — quién alimenta a quién, con qué (b, r, s), y que sea acíclico;
2. **la tabla de slots** — el mapa de nombres, con las covarianzas naciendo fijas;
3. **`expand_params`** — libres → estructura completa, con fijos, compartidos,
   productos y combinaciones lineales.

El caso de validación NO es el m6 canónico: su serie EI lleva un determinista
`compimp` (impulso compensado) que el lector de fue Python degrada a `pulse`, y
eso mueve la verosimilitud 1.88 antes de que drtran entre en juego (ver
`docs/PORTE.md` §5.4). Se usan las cinco series limpias de m6 con una red que
ejercita lo mismo: cadena EC → EU → EP, una entrada con dos salidas, un
denominador r=1, covarianzas libres, un producto y una combinación lineal.
"""

import os
import subprocess

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import build_cast_spec  # noqa: E402
from drtran.estimate import loglik  # noqa: E402
from drtran.network import find_cycle, read_dag, write_dag  # noqa: E402
from drtran.slots import (ALIAS, FIXED, FREE, LINCOMB, PRODUCT,  # noqa: E402
                          build_slots, read_cns)

REPO = "/home/david/Dropbox/SRC/drtran"
DATA = os.path.join(REPO, "tests/data/m6")
BIN = os.path.join(REPO, "bin/drtran")
LIMPIAS = ["EP", "EU", "EC", "EA", "P"]        # m6 sin EI (ver el docstring)

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA, "M6_EP.pre")),
    reason="faltan los .pre de m6 del repo C")


# ── utilidades ───────────────────────────────────────────────────────────────
def _specs(nombres=None):
    return [drtran.load_pre(os.path.join(DATA, f"M6_{n}.pre"))
            for n in (nombres or LIMPIAS)]


def _cs(dag=None, nombres=None):
    specs = _specs(nombres)
    cs = build_cast_spec(specs)
    if dag is None:
        return cs
    return build_cast_spec(specs, links=read_dag(dag, cs.names))


def _escribe(tmp_path, nombre, texto):
    p = tmp_path / nombre
    p.write_text(texto)
    return str(p)


DAG = """# 1=EP 2=EU 3=EC 4=EA 5=P
EP <- EU   1 0 1
EP <- EC   1 0 2
EU <- EC   2 1 1
"""

CNS_LIBRE = "q[4,2] = free\nq[4,3] = free\n"

CNS_EXPR = """q[4,2] = free
q[4,3] = free
omega1[1] = omega1[0] * theta_2[B^1]
omega2[0] = omega2[1] + omega2[2]
"""


# ── 1. el .dag ───────────────────────────────────────────────────────────────
def test_el_dag_se_lee_por_nombres_no_por_posicion(tmp_path):
    """Un `.dag` no debe depender del orden de la línea de órdenes."""
    cs = _cs()
    links = read_dag(_escribe(tmp_path, "r.dag", DAG), cs.names)
    assert len(links) == 3
    assert (links[0].out, links[0].inp, links[0].b, links[0].s) == (0, 1, 1, 1)
    assert (links[2].out, links[2].inp, links[2].b, links[2].r) == (1, 2, 2, 1)


def test_el_dag_rechaza_lo_que_no_es_una_red(tmp_path):
    nombres = ["EP", "EU", "EC", "EA", "P"]
    with pytest.raises(ValueError, match="serie desconocida"):
        read_dag(_escribe(tmp_path, "a.dag", "EP <- NADA 1 0 1\n"), nombres)
    with pytest.raises(ValueError, match="a sí misma"):
        read_dag(_escribe(tmp_path, "b.dag", "EP <- EP 1 0 1\n"), nombres)
    with pytest.raises(ValueError, match="'<-'"):
        read_dag(_escribe(tmp_path, "c.dag", "EP -> EU 1 0 1\n"), nombres)
    with pytest.raises(ValueError, match="b, r, s"):
        read_dag(_escribe(tmp_path, "d.dag", "EP <- EU x 0 1\n"), nombres)


def test_un_ciclo_se_rechaza_y_se_dice_cual(tmp_path):
    """Con un ciclo no hay orden topológico: dejaría de ser un DAG recursivo
    para ser un sistema de ecuaciones simultáneas, que no es lo que el cast
    representa. El mensaje da el ciclo, no sólo su existencia."""
    ciclico = "EP <- EU  1 0 1\nEU <- EC  1 0 1\nEC <- EP  1 0 1\n"
    with pytest.raises(ValueError, match="ciclo"):
        read_dag(_escribe(tmp_path, "ciclo.dag", ciclico), LIMPIAS)

    from drtran.cast import Link
    assert find_cycle([Link(0, 1), Link(1, 2), Link(2, 0)], 5) is not None
    assert find_cycle([Link(0, 1), Link(1, 2)], 5) is None      # cadena: DAG


def test_el_dag_va_y_vuelve(tmp_path):
    cs = _cs()
    p = _escribe(tmp_path, "r.dag", DAG)
    links = read_dag(p, cs.names)
    q = str(tmp_path / "vuelta.dag")
    write_dag(q, links, cs.names)
    assert read_dag(q, cs.names) == links


# ── 2. la tabla de slots ─────────────────────────────────────────────────────
def test_la_tabla_tiene_los_nombres_del_C_y_las_covarianzas_nacen_fijas(tmp_path):
    cs = _cs(_escribe(tmp_path, "r.dag", DAG))
    t = build_slots(cs)

    # el binario reporta "Structural parameters: 48 (free: 40, fixed: 8)"
    assert len(t) == 48
    assert t.n_free == 48 - 10          # las 10 covarianzas de 5 series, fijas

    # transferencias: numeradas 1..n_link, como el .cns del C
    assert t.names[:4] == ["omega1[0]", "omega1[1]", "omega2[0]", "omega2[1]"]
    assert t.index("delta3[1]") >= 0            # el enlace con r=1
    # ARMA y deterministas, con el índice de SERIE 1-based
    assert t.index("theta_2[B^1]") >= 0
    assert t.index("omega_d1[1,0]") >= 0
    assert t.index("log(var5/var1)") >= 0

    for i in range(2, 6):
        for j in range(1, i):
            k = t.index(f"q[{i},{j}]")
            assert k >= 0 and t.slots[k].kind == FIXED and t.slots[k].value == 0.0


def test_liberar_una_covarianza_es_una_decision_no_un_bloque(tmp_path):
    """`q[i,j] = free`, en el mismo sitio y con el mismo lenguaje que el resto:
    el legacy m6-1 no libera las 15 de su sistema, libera TRES."""
    cs = _cs(_escribe(tmp_path, "r.dag", DAG))
    t = build_slots(cs)
    n0 = t.n_free
    assert read_cns(_escribe(tmp_path, "c.cns", CNS_LIBRE), t) == 2
    assert t.n_free == n0 + 2
    assert t.slots[t.index("q[4,2]")].kind == FREE
    assert t.slots[t.index("q[3,1]")].kind == FIXED       # las demás, intactas


def test_la_tabla_cuadra_con_el_vector_de_fue():
    """La red de seguridad del puerto: si la enumeración de nombres se separa de
    la de `fue._build_initial_x`, los nombres dejan de corresponder a las
    posiciones y el `.cns` restringiría el parámetro equivocado, en silencio."""
    cs = _cs()
    t = build_slots(cs)
    assert len(t) == cs.npar + cs.m * (cs.m - 1) // 2


# ── 3. las restricciones ─────────────────────────────────────────────────────
def test_las_cinco_naturalezas_de_un_slot(tmp_path):
    cs = _cs(_escribe(tmp_path, "r.dag", DAG))
    t = build_slots(cs)
    read_cns(_escribe(tmp_path, "c.cns", """
        q[4,2] = free
        omega1[0] = 0.25
        delta3[1] = theta_2[B^1]
        omega2[1] = -omega1[0] * theta_3[B^1]
        omega2[0] = omega2[1] + omega2[2] - theta_2[B^1]
    """), t)

    assert t.slots[t.index("q[4,2]")].kind == FREE
    assert t.slots[t.index("omega1[0]")].kind == FIXED
    assert t.slots[t.index("delta3[1]")].kind == ALIAS
    assert t.slots[t.index("omega2[1]")].kind == PRODUCT
    assert t.slots[t.index("omega2[0]")].kind == LINCOMB
    assert len(t.slots[t.index("omega2[0]")].terms) == 3

    x = t.expand(np.zeros(t.n_free))
    assert x[t.index("omega1[0]")] == 0.25

    # compartido: un solo grado de libertad en dos sitios
    xf = np.zeros(t.n_free)
    xf[t.free_of_slot[t.index("theta_2[B^1]")]] = 0.7
    x = t.expand(xf)
    assert x[t.index("delta3[1]")] == pytest.approx(0.7)
    # producto: -0.25 * theta_3, con theta_3 = 0
    assert x[t.index("omega2[1]")] == pytest.approx(0.0)
    # combinación lineal: omega2[1] + omega2[2] - theta_2
    assert x[t.index("omega2[0]")] == pytest.approx(0.0 + 0.0 - 0.7)


def test_el_cns_no_se_traga_lo_que_no_entiende(tmp_path):
    cs = _cs()
    t = build_slots(cs)
    with pytest.raises(KeyError, match="desconocido"):
        read_cns(_escribe(tmp_path, "a.cns", "no_existe = 1\n"), t)
    # a la derecha: ni un slot ni un número — como el C, «cannot parse»
    with pytest.raises(ValueError, match="no sé interpretar"):
        read_cns(_escribe(tmp_path, "b.cns", "log(var2/var1) = tampoco\n"), t)
    with pytest.raises(ValueError, match="consigo mismo"):
        read_cns(_escribe(tmp_path, "c.cns",
                          "log(var2/var1) = log(var2/var1)\n"), t)
    with pytest.raises(ValueError, match="sí mismo"):
        read_cns(_escribe(tmp_path, "d.cns",
                          "log(var2/var1) = log(var2/var1) * log(var3/var1)\n"), t)


def test_los_comentarios_y_las_lineas_vacias_no_estorban(tmp_path):
    cs = _cs()
    t = build_slots(cs)
    n = read_cns(_escribe(tmp_path, "c.cns",
                          "# todo esto es comentario\n\n"
                          "q[2,1] = free   # y esto también\n"), t)
    assert n == 1 and t.slots[t.index("q[2,1]")].kind == FREE


# ── 4. la covarianza no diagonal ─────────────────────────────────────────────
def test_una_covarianza_libre_cambia_sigma_y_una_no_PSD_se_rechaza(tmp_path):
    from drtran.cast import build_sigma
    from drtran.embed import cast_embedded

    cs = _cs()
    t = build_slots(cs)
    read_cns(_escribe(tmp_path, "c.cns", "q[2,1] = free\n"), t)

    x = drtran.x0_full(cs, t)
    x[t.index("q[2,1]")] = 0.1
    _phi, _th, _mu, _w, sigma, ifault = cast_embedded(x, cs)
    assert ifault == 0
    assert sigma[1, 0] == pytest.approx(0.1) and sigma[0, 1] == pytest.approx(0.1)

    # fuera de la región PSD se rechaza: el objetivo devuelve 1.0 y el
    # optimizador se aleja (Mauricio 1995 §3), en vez de evaluar lo imposible
    x[t.index("q[2,1]")] = 50.0
    _q, _idx, ifa = build_sigma(x, cs.npar - (cs.m - 1), cs.m)
    assert ifa != 0
    assert cast_embedded(x, cs)[5] != 0


# ── 5. homologación con el binario ───────────────────────────────────────────
def _optimo_del_C(dag, cns, tmp_path, nombres=LIMPIAS):
    """Corre el binario y devuelve `(cast_spec, tabla, x_C, logL_C)`.

    Se relanza EN VIVO en vez de guardar cifras: una referencia congelada deja
    de avisar en cuanto el C cambia, que es justo cuando hay que enterarse.
    """
    import re
    pre = [os.path.join(DATA, f"M6_{n}.pre") for n in nombres]
    out = str(tmp_path / "c.out")
    r = subprocess.run([BIN, *pre, "-n", dag, "-c", cns, "-o", out],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr

    ll_C = None
    pares, dentro = [], False
    for ln in open(out):
        if ln.startswith("Log-likelihood"):
            ll_C = float(re.search(r"-?\d+\.\d+", ln).group())
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

    cs = _cs(dag, nombres)
    t = build_slots(cs)
    read_cns(cns, t)

    # por nombre, respetando la k-ésima ocurrencia: el C repite nombres cuando
    # una serie tiene dos factores del mismo tipo (`theta_5[B^1]` en m6)
    from collections import Counter, defaultdict
    porn = defaultdict(list)
    for n, v in pares:
        porn[n].append(v)
    uso, x = Counter(), np.zeros(len(t))
    for i, s in enumerate(t.slots):
        vs = porn.get(s.name)
        assert vs, f"el C no reporta el slot {s.name}"
        k = uso[s.name]
        uso[s.name] += 1
        x[i] = vs[k] if k < len(vs) else vs[-1]
    return cs, t, x, ll_C


@pytest.mark.skipif(not os.path.exists(BIN), reason="falta el binario C")
def test_la_red_homologa_con_el_binario(tmp_path):
    """Cadena EC → EU → EP, una entrada con dos salidas, un denominador r=1 y
    dos covarianzas libres: el puerto reproduce al C en su propio óptimo."""
    dag = _escribe(tmp_path, "r.dag", DAG)
    cns = _escribe(tmp_path, "c.cns", CNS_LIBRE)
    cs, t, x, ll_C = _optimo_del_C(dag, cns, tmp_path)
    assert len(t) == 48 and t.n_free == 40          # lo que reporta el binario
    ll, ifault = loglik(x, cs, embed=True)
    assert ifault == 0
    assert ll == pytest.approx(ll_C, abs=1e-5)


@pytest.mark.skipif(not os.path.exists(BIN), reason="falta el binario C")
def test_los_productos_y_la_combinacion_lineal_homologan(tmp_path):
    """Y `expand` reconstruye los slots derivados a partir de SÓLO los libres:
    es la prueba de que el DSL no es decorativo."""
    dag = _escribe(tmp_path, "r.dag", DAG)
    cns = _escribe(tmp_path, "c.cns", CNS_EXPR)
    cs, t, xC, ll_C = _optimo_del_C(dag, cns, tmp_path)
    assert t.n_free == 38                            # 40 − 2 expresiones

    xexp = t.expand(t.pack(xC))
    assert np.max(np.abs(xexp - xC)) < 1e-5, "expand no reconstruye lo derivado"

    i0, i1, i2 = (t.index(f"omega2[{k}]") for k in (0, 1, 2))
    assert xexp[i0] == pytest.approx(xexp[i1] + xexp[i2])       # el (1−B)
    ip, ia, ib = t.index("omega1[1]"), t.index("omega1[0]"), t.index("theta_2[B^1]")
    assert xexp[ip] == pytest.approx(xexp[ia] * xexp[ib])       # el producto

    ll, ifault = loglik(xexp, cs, embed=True)
    assert ifault == 0
    assert ll == pytest.approx(ll_C, abs=1e-5)


@pytest.mark.skipif(not os.path.exists(BIN), reason="falta el binario C")
@pytest.mark.parametrize("cns,diana,tol", [
    # numeradores libres: se evalúa en los mismos números que imprime el C
    ("m6_net.cns", -1697.613401, 1e-4),
    # con EXPRESIONES la tolerancia sube: el informe del C redondea a 6 decimales
    # y los slots derivados se reconstruyen a partir de factores ya redondeados,
    # así que el error de entrada se propaga por el producto
    ("m6_net_prod.cns", None, 1e-3),       # + los 2 productos de MA compartida
    ("m6_net_full.cns", None, 1e-3),       # + el factor fijo (1−B) de EI←EU
])
def test_el_m6_canonico(cns, diana, tol, tmp_path):
    """El sistema de Relloso (1997, Tabla 4) entero, con las SEIS series.

    Estuvo bloqueado: la serie EI lleva un determinista `compimp` —el impulso
    compensado— que fue Python leía como un impulso a secas, y eso movía la
    verosimilitud 1.88 antes de que drtran entrara en juego. Arreglado en fue
    0.1.9 (BUG-0006), el puerto reproduce las dianas del C.
    """
    dag = os.path.join(DATA, "m6_net.dag")
    cs, t, x, ll_C = _optimo_del_C(dag, os.path.join(DATA, cns), tmp_path,
                                   ["EP", "EI", "EU", "EC", "EA", "P"])
    if diana is not None:
        assert ll_C == pytest.approx(diana, abs=1e-5), "cambió el binario C"
    ll, ifault = loglik(x, cs, embed=True)
    assert ifault == 0
    assert ll == pytest.approx(ll_C, abs=tol)


@pytest.mark.skipif(not os.path.exists(BIN), reason="falta el binario C")
def test_el_m6_diagonal_canonico(tmp_path):
    """El escalón anterior: seis univariantes juntos, Σ no diagonal, sin red."""
    import re
    import subprocess

    pre = [os.path.join(DATA, f"M6_{n}.pre")
           for n in ("EP", "EI", "EU", "EC", "EA", "P")]
    out = str(tmp_path / "diag.out")
    r = subprocess.run([BIN, *pre, "-0", "-c", os.path.join(DATA, "m6.cns"),
                        "-o", out], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr
    ll_C = float(re.search(r"^Log-likelihood\s*=\s*(-?\d+\.\d+)",
                           open(out).read(), re.M).group(1))
    assert ll_C == pytest.approx(-1709.511575, abs=1e-5), "cambió el binario C"

    cs = _cs(nombres=["EP", "EI", "EU", "EC", "EA", "P"])
    t = build_slots(cs)
    read_cns(os.path.join(DATA, "m6.cns"), t)
    assert t.n_free == 56, "el -0 del C libera las 15 covarianzas"


@pytest.mark.skipif(not os.path.exists(BIN), reason="falta el binario C")
def test_el_optimizador_llega_al_mismo_optimo_que_el_C(tmp_path):
    """Homologar en el óptimo del C prueba el CAST; esto prueba la BÚSQUEDA.

    Red pequeña (EC → EU → EP, 24 libres) para que quepa en la batería: el
    puerto arranca en el escalón diagonal y tiene que llegar donde llegó el C.
    """
    dag = _escribe(tmp_path, "r.dag", "EP <- EU   1 0 1\nEU <- EC   2 0 1\n")
    cns = _escribe(tmp_path, "c.cns", "q[3,1] = free\n")
    tres = ["EP", "EU", "EC"]
    cs, t, _x, ll_C = _optimo_del_C(dag, cns, tmp_path, tres)
    assert len(t) == 26 and t.n_free == 24

    f = drtran.fit(cs, slots=t, embed=True)
    assert f.ifault == 0
    # El criterio es LLEGAR, no cómo se para. `termcode` 3 es "paró sin mejora",
    # que en el óptimo es lo normal -- el propio C reporta ahí "STOPPED AT A
    # POINT WITH NO IMPROVEMENT" -- y basta un cambio al nivel del épsilon (aquí,
    # pasar del elf de Python puro al compilado, idénticos a 1e-13) para que la
    # búsqueda lineal deje de encontrar mejora y el 2 se vuelva 3. Fallo real es
    # 4-5. Lo que se exige es el valor, que es lo que se compara con el C.
    assert f.termcode in (1, 2, 3), f"no convergió: {f.estado}"
    assert f.loglik == pytest.approx(ll_C, abs=1e-5)
    assert len(f.xfree) == 24 and len(f.x) == 26


def test_sin_enlaces_ni_covarianzas_la_red_es_el_diagonal_de_siempre():
    """La tabla de slots no cambia el modelo por existir: con todo libre y las
    covarianzas en cero, `fit(slots=...)` parte del mismo sitio que sin ella."""
    cs = _cs()
    t = build_slots(cs)
    x_sin = drtran.x0_from_pre(cs)
    x_con = drtran.x0_full(cs, t)
    assert np.allclose(x_con[:len(x_sin)], x_sin)
    assert np.all(x_con[len(x_sin):] == 0.0)
    assert loglik(x_con, cs, embed=True)[0] == pytest.approx(
        loglik(x_sin, cs, embed=True)[0], abs=1e-9)
