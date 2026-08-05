"""Mide el test de parada de umstop en el optimo, a las dos escalas."""
import copy
import numpy as np
import drtran
from drtran.cast import build_cast_spec, Link
from drtran.pre import PreSpec
from drtran.slots import build_slots
from drtran.estimate import _f1f2
from drvarma import _qnewt

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


for fac in (100.0, 1.0):
    specs = [at_scale(C + 'ES_CPI_m10.pre', fac), at_scale(C + 'WTI_ar1.pre', fac)]
    cs = build_cast_spec(specs, links=[Link(0, 1, b=0, r=0, s=1)])
    t = build_slots(cs)
    f = drtran.fit(cs, embed=True, slots=t)

    k = t.n_free
    x = np.zeros(k + 1)
    x[1:] = f.xfree
    f1_0, f2_0, _ = _f1f2(t.expand(f.xfree), cs, -1e-3, True)

    def obj(v):
        a, b, ifa = _f1f2(t.expand(np.asarray(v[1:k + 1], float)), cs, -1e-3, True)
        if ifa or not (a > 0 and b > 0):
            return 1.0
        return (a / f1_0) ** cs.m * (b / f2_0)

    g = np.zeros(k + 1)
    _qnewt.cdgrad(obj, k, x.copy(), _qnewt.MACHEPS, g)
    fk = obj(x)

    rel = [abs(g[i]) * (abs(x[i]) + 1.0) / (abs(fk) + 1.0) for i in range(1, k + 1)]
    worst = int(np.argmax(rel)) + 1
    fires = 'DISPARA' if max(rel) <= 1e-7 else 'NO dispara'
    print('refactor=%6.0f  max|g|=%.3e  relgrad=%.3e  (gradtol=1e-7 -> %s)'
          % (fac, np.max(np.abs(g[1:])), max(rel), fires))
    print('               peor slot: %-18s x=%+.6f  g=%+.3e  |x|+1=%.6f'
          % (t.names[t.slot_of_free[worst - 1]], x[worst], g[worst],
             abs(x[worst]) + 1.0))
    # el mismo test si el tamano tipico fuese |x| en vez de 1
    typx = [abs(g[i]) * max(abs(x[i]), 1e-8) / (abs(fk) + 1.0) for i in range(1, k + 1)]
    print('               con typx=|x| en vez de 1:  relgrad=%.3e' % max(typx))
