"""
BUG-51 — el código tiene que compilar con el Python MÍNIMO que declara el paquete.

drtran declara `requires-python = ">=3.10"`, pero `mcp_server.py` usaba una barra
invertida dentro de la expresión de un f-string, que sólo es válida desde Python
3.12 (PEP 701). Con 3.10 y 3.11 el módulo no se importaba y `mtram` no arrancaba,
mientras pip lo instalaba sin avisar. Nadie lo veía porque se desarrolla con 3.12.

Esta prueba falla también EN 3.12, que es donde se desarrolla: en versiones
anteriores basta compilar; en 3.12+ el compilador acepta la sintaxis nueva, así
que se buscan en los tokens las construcciones de f-string que sólo existen desde
3.12 —una barra invertida, o la misma comilla, dentro de un campo `{…}`.
"""

import io
import pathlib
import re
import sys
import tokenize

import pytest

RAIZ = pathlib.Path(__file__).resolve().parents[1]
FUENTES = sorted((RAIZ / "src").rglob("*.py"))


def _minimo():
    txt = (RAIZ / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'requires-python\s*=\s*"\s*>=\s*(\d+)\.(\d+)', txt)
    return (int(m.group(1)), int(m.group(2)))


def _solo_312(src):
    """Líneas con sintaxis de f-string que sólo acepta Python 3.12+ (PEP 701)."""
    malas = []
    pila = []          # por cada f-string abierto: [comilla, profundidad de llaves]
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        tipo = tokenize.tok_name[tok.type]
        if tipo == "FSTRING_START":
            if pila and pila[-1][1] > 0 and tok.string.lstrip("rRfFbBuU")[:1] == pila[-1][0]:
                malas.append((tok.start[0], "la misma comilla dentro de un f-string"))
            pila.append([tok.string.lstrip("rRfFbBuU")[:1], 0])
        elif tipo == "FSTRING_END" and pila:
            pila.pop()
        elif pila and tipo == "OP" and tok.string in "{}":
            pila[-1][1] += 1 if tok.string == "{" else -1
        elif pila and pila[-1][1] > 0 and tipo == "STRING":
            if "\\" in tok.string:
                malas.append((tok.start[0], "barra invertida en la expresión de un f-string"))
            if tok.string.lstrip("rRbBuU")[:1] == pila[-1][0]:
                malas.append((tok.start[0], "la misma comilla dentro de un f-string"))
    return malas


def _problemas(path):
    src = path.read_text(encoding="utf-8")
    if sys.version_info < (3, 12):
        try:
            compile(src, str(path), "exec")
        except SyntaxError as e:
            return [(e.lineno, e.msg)]
        return []
    return _solo_312(src)


def test_la_deteccion_ve_el_caso_de_bug_51():
    if sys.version_info < (3, 12):
        pytest.skip("en <3.12 lo detecta compile(); la prueba del detector es para 3.12+")
    malo = "c = 1\ntxt = f\"{'+ ' if c >= 0 else '\\u2212 '}{abs(c):.4f}\"\n"
    bueno = "c = 1\nsigno = '+ ' if c >= 0 else '\\u2212 '\ntxt = f\"{signo}{abs(c):.4f}\"\n"
    assert _solo_312(malo)
    assert not _solo_312(bueno)


def test_todo_compila_con_el_python_minimo_declarado():
    if _minimo() >= (3, 12):
        pytest.skip("el mínimo declarado ya es 3.12")
    malos = {str(p.relative_to(RAIZ)): pr for p in FUENTES if (pr := _problemas(p))}
    assert not malos, (
        f"sintaxis que Python {_minimo()[0]}.{_minimo()[1]} —el mínimo que declara "
        f"pyproject.toml— no acepta: {malos} (BUG-51)")
