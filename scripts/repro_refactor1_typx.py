"""refactor=1: el fix `typx` in situ, sin monkeypatch.

`qnewtopt.c` fija el tamano tipico de parametro a 1. A refactor=1 los
deterministas valen ~1e-4, asi que el test de gradiente se vuelve una tolerancia
ABSOLUTA inalcanzable y el de paso una que se cumple de inmediato: el optimizador
llega al mismo optimo y no puede certificarlo (termcode 2 en vez de 1).

`drtran.fit(typx=...)` -> `raxopt(typx=...)`. typx=None reproduce el C.
"""
import copy
import numpy as np
import drtran
from drtran.cast import build_cast_spec, Link
from drtran.pre import PreSpec

C = '/home/david/Dropbox/SRC/drtran/tests/cases/'


def at_scale(path, factor):
    s = drtran.load_pre(path)
    m = copy.deepcopy(s.model)
    r = factor / float(m.refactor)
    m.refactor = factor
    for iv in m.interventions:
        iv.omega = [v * r for v in iv.omega]
    m.mu0 = float(m.mu0) * r
    return PreSpec(ts=m.series, model=m, path=s.path)


def run(fac, typx):
    specs = [at_scale(C + 'ES_CPI_m10.pre', fac), at_scale(C + 'WTI_ar1.pre', fac)]
    cs = build_cast_spec(specs, links=[Link(0, 1, b=0, r=0, s=1)])
    f = drtran.fit(cs, embed=True, typx=typx)
    # el jacobiano del reescalado uniforme: n_obs * m * log(razon de refactor)
    equiv = f.loglik - (0.0 if fac == 100.0 else 215 * 2 * np.log(100.0))
    print('  refactor=%6.0f  typx=%-6s  logL=%12.4f  (equiv %13.6f)  '
          'termcode=%d  iters=%d  omegas=%s'
          % (fac, str(typx), f.loglik, equiv, f.termcode, f.nit,
             np.array2string(np.asarray(f.xfree[:2]), precision=6)))
    return f


if __name__ == '__main__':
    print('el C (typx=None):')
    a = run(100.0, None)
    b = run(1.0, None)
    print('con el fix (typx=1e-3, el defecto de drtran.fit):')
    c = run(1.0, 1e-3)

    assert a.termcode == 1, 'refactor=100 deberia certificar por gradiente'
    assert b.termcode == 2, 'sin typx, refactor=1 cae al test de paso'
    assert c.termcode == 1, 'con typx, refactor=1 certifica por gradiente'
    np.testing.assert_allclose(c.xfree[:2], a.xfree[:2], atol=1e-6)
    print('\nOK: mismo optimo a 1e-6, y el certificado recuperado.')
