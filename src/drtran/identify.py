"""Identificación de (b, r, s) por preblanqueo y CCF — el paso 1 de Box-Jenkins.

Puerto de `prewhiten_and_identify` (`drtran.c`). El procedimiento clásico:

1. **Preblanquear la entrada** con su propio ARMA: aplicar φ_X(B) e invertir
   θ_X(B) hasta dejar ruido blanco `a_X`.
2. **Aplicar EL MISMO filtro a la salida** → `beta_Y`. Es el punto que suele
   olvidarse: filtrar sólo la entrada deja la CCF contaminada por la estructura
   propia de la salida.
3. **CCF entre ambas**. Con la entrada ya blanca, r(k) es proporcional a los
   pesos de la respuesta al impulso: ν(k) = r(k)·s_β/s_a.
4. **Leer (b, r, s)** del patrón de ν.
5. **Comprobar la exogeneidad** mirando el lado k<0 de la CCF: si la salida
   antecede a la entrada, hay retroalimentación y el modelo de transferencia con
   una sola entrada no se sostiene.

Fuentes
-------
* **Box & Jenkins (1976, cap. 11)** — el preblanqueo SIMPLE que se implementa
  aquí: filtrar la entrada con su ARMA y aplicar EL MISMO filtro a la salida, de
  modo que r(k) estime directamente los ν escalados.
* **Haugh & Box (1977)**, JASA 72(357) 121-130 — el artículo original. Proponen
  el DOBLE preblanqueo (cada serie con su propio modelo) y dan la distribución
  asintótica de las correlaciones cruzadas.
* **Tsay (1985)**, JBES 3(3) 228-237 — la crítica. Señala dos problemas del
  enfoque; ver más abajo cuál cubrimos y cuál no.

Dos decisiones del C que se replican porque son las que evitan disparates:

* La estructura es el **bloque CONTIGUO** que arranca en b. Con bandas al 5 % se
  espera que 1 de cada 20 retardos las cruce por azar, así que tomar el último
  significativo de toda la CCF disparaba s a valores absurdos (s=24).
* La exogeneidad se juzga con un **portmanteau** sobre k<0, no contando cuántos
  cruzan: contar dispara el aviso casi siempre por el mismo motivo.

Lo que dice la literatura, y qué hacemos con ello
-------------------------------------------------
**Haugh & Box (1977) §1.4**: `var{r_xy(k)} ≈ (N−k)⁻¹`, así que la banda debe
ensancharse con el retardo. El C usa `2/√N` constante. Medido en el caso
canónico, esa diferencia crea un falso positivo en k=24 — que es justamente el
que obligó a la heurística del bloque contiguo. Disponible como
`banda="haugh-box"`; el defecto sigue siendo `"constante"` para homologar.

**Tsay (1985) §2** señala dos problemas del preblanqueo:

1. *La unidireccionalidad rara vez es cierta; puede haber retroalimentación.*
   CUBIERTO: es exactamente el portmanteau de exogeneidad sobre k<0.
2. *En series no estacionarias, la transformación (diferenciar) puede DEBILITAR
   la relación entre las variables*, con conclusiones contradictorias entre el
   análisis en niveles y en series transformadas (cita Feige & Pearce 1974).
   **NO cubierto**: aquí se identifica sobre la serie ya diferenciada que fija el
   `.pre`. Es una limitación real del método, no del puerto. La alternativa de
   Tsay —aproximación VAR + mínimos cuadrados + método de la esquina, sin
   preblanquear ni diferenciar— queda fuera de alcance por ahora.

**Tsay (1985) ec. (2.2)**, de propina: la covarianza de las innovaciones de la
forma reducida es diagonal **si y sólo si b ≠ 0**. Con transferencia
contemporánea (b=0) la forma reducida NO puede tener Σ diagonal — que es
exactamente por lo que el cast empotrado necesita `normalize_phi0` y por lo que
el C guarda aparte la Q ESTRUCTURAL. Confirmación teórica del diseño.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Identificacion:
    """Lo que la CCF preblanqueada dice sobre un enlace."""

    lags: np.ndarray                 # retardos, de -nlags a +nlags
    ccf: np.ndarray                  # r(k)
    nu: np.ndarray                   # pesos ν(k) = r(k)·s_β/s_a
    threshold: float                 # el umbral en k=0
    b: int
    r: int
    s: int
    umbral: np.ndarray = None        # umbral por retardo (puede variar con |k|)
    alternativas: list = field(default_factory=list)   # [(b, r, s, motivo), …]
    exogena: bool = True
    Q_exogeneidad: float = 0.0
    p_exogeneidad: float = 1.0
    n_signif_negativos: int = 0

    @property
    def hay_relacion(self):
        return not (self.b == 0 and self.r == 0 and self.s == 0
                    and not self.significativos_no_negativos)

    @property
    def significativos_no_negativos(self):
        u = self.umbral if self.umbral is not None else self.threshold
        m = (self.lags >= 0) & (np.abs(self.ccf) > u)
        return self.lags[m].tolist()

    def __repr__(self):                                   # pragma: no cover
        ex = "exógena" if self.exogena else "¡RETROALIMENTACIÓN!"
        return (f"Identificacion(b={self.b}, r={self.r}, s={self.s}, "
                f"{ex} p={self.p_exogeneidad:.3f})")


def prewhiten(w, phi, theta):
    """Filtra `w` con el ARMA dado: aplica φ(B) e invierte θ(B).

        u[t]  = w[t] − Σ_k φ_k·w[t−k]
        a[t]  = u[t] + Σ_k θ_k·a[t−k]

    Los sumatorios se truncan al principio de la muestra (k < t), como en el C.
    """
    w = np.asarray(w, float)
    phi = np.asarray(phi, float).ravel()
    theta = np.asarray(theta, float).ravel()
    n = len(w)
    a = np.zeros(n)
    for t in range(n):
        u = w[t]
        for k in range(1, min(len(phi), t) + 1):
            u -= phi[k - 1] * w[t - k]
        a[t] = u
        for k in range(1, min(len(theta), t) + 1):
            a[t] += theta[k - 1] * a[t - k]
    return a


def _ccf(d1, d2, nlags):
    """corr(d1_t, d2_{t+k}) para k = 0..nlags (la convención de `diagnose.c:Ccf`)."""
    d1 = np.asarray(d1, float)
    d2 = np.asarray(d2, float)
    n = len(d1)
    x1 = d1 - d1.mean()
    x2 = d2 - d2.mean()
    s1 = math.sqrt(float((x1 * x1).sum()) / n)
    s2 = math.sqrt(float((x2 * x2).sum()) / n)
    out = np.zeros(nlags + 1)
    if s1 < 1e-12 or s2 < 1e-12:
        return out
    for k in range(nlags + 1):
        out[k] = float((x1[:n - k] * x2[k:]).sum()) / (n * s1 * s2)
    return out


def identify(cast_spec, link, x=None, nlags=None, banda="constante"):
    """Identifica (b, r, s) del `link` a partir de la CCF preblanqueada.

    `x` por defecto son las semillas del `.pre`: el preblanqueo usa el ARMA que
    fue estimó para la entrada, que es justo lo que la escalera pone ahí.

    `banda` elige el umbral de significación:

    * `"constante"` (por defecto) — `2/√N` para todos los retardos, que es lo que
      hace el C y con lo que este puerto homologa.
    * `"haugh-box"` — `2/√(N−|k|)`, que es lo que dice el artículo original.
      Haugh & Box (1977, §1.4) dan `var{r_xy(k)} ≈ (N−k)⁻¹` y proponen juzgar cada
      estimación «against an approximate standard error of N^(−½) **or
      (N−k)^(−½)**». La banda debe ENSANCHARSE con el retardo, porque hay menos
      solapamiento para estimarla.

      No es cosmético: en el caso canónico (N=215) el umbral en k=24 pasa de
      0.1364 a 0.1447, y el pico de −0.1423 deja de ser significativo. Ese pico es
      justamente el falso positivo que obligó al C a introducir la heurística del
      «bloque contiguo». Con la banda del artículo el falso positivo no llega a
      existir: **la heurística compensa una banda mal calibrada**.

      Se deja opcional para no romper la homologación con el binario.
    """
    from fue.cast_us import cast_us_py

    from .cast import x0_from_pre

    x = np.asarray(x0_from_pre(cast_spec) if x is None else x, float)

    # trozo univariante de cada serie
    idx = cast_spec.npar_links
    piezas = []
    for sc in cast_spec.series:
        piezas.append(x[idx:idx + sc.npar]); idx += sc.npar

    def preparar(i):
        p, q, phi, theta, mu, w, ifault = cast_us_py(piezas[i],
                                                     cast_spec.series[i].est_spec)
        return (np.asarray(phi, float)[:p], np.asarray(theta, float)[:q],
                np.asarray(w, float), int(ifault))

    phi_x, theta_x, w_x, if_x = preparar(link.inp)
    _phi_y, _theta_y, w_y, if_y = preparar(link.out)
    if if_x or if_y:
        raise ValueError("el cast univariante falló; revisa los .pre")

    n = min(len(w_x), len(w_y))
    w_x, w_y = w_x[len(w_x) - n:], w_y[len(w_y) - n:]

    if nlags is None:                       # nlags = min(n/4, 24), al menos 10
        nlags = max(10, min(n // 4, 24))

    # 1 y 2: preblanquear X y filtrar Y con EL MISMO filtro
    a_x = prewhiten(w_x, phi_x, theta_x)
    beta_y = prewhiten(w_y, phi_x, theta_x)

    s_a, s_b = a_x.std(), beta_y.std()
    lags = np.arange(-nlags, nlags + 1)
    ccf = np.zeros(2 * nlags + 1)
    if s_a < 1e-12 or s_b < 1e-12:
        return Identificacion(lags=lags, ccf=ccf, nu=ccf.copy(),
                              threshold=2.0 / math.sqrt(n), b=0, r=0, s=0)

    # 3: k >= 0 es "la entrada antecede a la salida" (la transferencia);
    #    k < 0 es "la salida antecede" (retroalimentación: no debería haber).
    cpos = _ccf(a_x, beta_y, nlags)
    cneg = _ccf(beta_y, a_x, nlags)
    for k in range(nlags + 1):
        ccf[nlags + k] = cpos[k]
        ccf[nlags - k] = cneg[k]

    nu = ccf * (s_b / s_a)                  # 4: pesos de la respuesta al impulso
    if banda == "haugh-box":
        # Haugh & Box (1977) §1.4: var{r_xy(k)} ~ (N-k)^-1
        umbral = np.array([2.0 / math.sqrt(n - abs(int(k))) for k in lags])
    elif banda == "constante":
        umbral = np.full(len(lags), 2.0 / math.sqrt(n))
    else:
        raise ValueError(f"banda desconocida: {banda!r}")
    threshold = float(umbral[nlags])         # el de k=0, para el informe

    # 5: exogeneidad por portmanteau sobre k<0 (no por recuento).
    # `diagnose.c:ChiTestC`: Q = n(n+2)·Σ_{i=1..lags} r_i² / (n − i + 1), saltando
    # el retardo 0. El divisor es n−i+1, no n−i.
    cn = np.array([ccf[nlags - k] for k in range(1, nlags + 1)])
    divisores = np.array([n - i + 1 for i in range(1, nlags + 1)], float)
    Q = float(n * (n + 2) * np.sum(cn ** 2 / divisores))
    try:
        from scipy.stats import chi2
        pval = float(1.0 - chi2.cdf(Q, nlags))
    except Exception:                                    # pragma: no cover
        pval = float("nan")
    nsig_neg = int(np.sum([abs(cn[i - 1]) > umbral[nlags - i]
                           for i in range(1, nlags + 1)]))

    # 6: b = primer k>=0 significativo; el bloque CONTIGUO desde ahí
    sig_pos = [k for k in range(nlags + 1)
               if abs(ccf[nlags + k]) > umbral[nlags + k]]
    if not sig_pos:
        return Identificacion(lags=lags, ccf=ccf, nu=nu, threshold=threshold,
                              umbral=umbral, b=0, r=0, s=0,
                              exogena=(pval >= 0.05),
                              Q_exogeneidad=Q, p_exogeneidad=pval,
                              n_signif_negativos=nsig_neg)

    b_hat = sig_pos[0]
    last = b_hat
    while last + 1 <= nlags and abs(ccf[nlags + last + 1]) > umbral[nlags + last + 1]:
        last += 1

    alternativas = []
    # Candidato A: todo peso significativo como un ω libre
    s1 = last - b_hat
    alternativas.append((b_hat, 0, s1, "cada peso significativo como ω libre"))

    # Candidato B: si la cola decae de forma aproximadamente geométrica, un
    # denominador de orden 1 la resume con un solo parámetro.
    nblock = last - b_hat + 1
    if nblock >= 3:
        q1 = abs(nu[nlags + last]) / (abs(nu[nlags + last - 1]) + 1e-12)
        q2 = abs(nu[nlags + last - 1]) / (abs(nu[nlags + last - 2]) + 1e-12)
        if q1 < 0.95 and q2 < 0.95 and abs(q1 - q2) < 0.25:
            s2 = max(0, (last - 2) - b_hat)
            alternativas.append(
                (b_hat, 1, s2,
                 f"la cola decae ~geométricamente (razón≈{0.5 * (q1 + q2):.2f})"))

    b, r, s, _motivo = alternativas[-1]      # el más parsimonioso propuesto
    return Identificacion(lags=lags, ccf=ccf, nu=nu, threshold=threshold,
                          umbral=umbral, b=b, r=r, s=s, alternativas=alternativas,
                          exogena=(pval >= 0.05), Q_exogeneidad=Q,
                          p_exogeneidad=pval, n_signif_negativos=nsig_neg)


def report(ident, nombres=("X", "Y")):
    """Informe de texto, al estilo del que imprime el C."""
    xi, yi = nombres
    L = ["=" * 61,
         "  IDENTIFICACIÓN — preblanqueo y CCF (Box-Jenkins)",
         f"  {xi}  ->  {yi}",
         "=" * 61,
         "  La entrada se preblanquea con su ARMA y el MISMO filtro se aplica",
         "  a la salida.  r(k) = corr(a_t, β_{t+k}):",
         f"    k > 0  ->  {yi} responde a {xi} con k periodos de retardo",
         f"    k < 0  ->  {yi} antecede a {xi} (retroalimentación: no debería haber)",
         "",
         f"  Pesos ν(k) = r(k)·s_β/s_a   (banda ±{ident.threshold:.4f})",
         "     k      r(k)       ν(k)",
         "    ------------------------------"]
    u = ident.umbral if ident.umbral is not None else np.full(len(ident.lags),
                                                             ident.threshold)
    for k, c, v, uk in zip(ident.lags, ident.ccf, ident.nu, u):
        if abs(c) > uk:
            L.append(f"   {int(k):4d}  {c:8.4f}  {v:9.4f}  *")
    if not any(abs(ident.ccf) > u):
        L.append("     (ninguno)")
    L += ["",
          "  Exogeneidad — portmanteau de la CCF en k < 0:",
          f"    Q({len(ident.lags) // 2}) = {ident.Q_exogeneidad:.4f}"
          f"   p = {ident.p_exogeneidad:.4f}   "
          f"[{ident.n_signif_negativos} significativos]"]
    L.append("    " + ("la entrada se comporta como exógena. OK" if ident.exogena
                       else "AVISO: la salida antecede a la entrada. Puede haber "
                            "RETROALIMENTACIÓN (Y -> X); el modelo de una sola "
                            "entrada supone que X es EXÓGENA."))
    L.append("")
    if ident.alternativas:
        L.append("  PROPUESTAS:")
        for i, (b, r, s, motivo) in enumerate(ident.alternativas):
            marca = " <-" if (b, r, s) == (ident.b, ident.r, ident.s) else "   "
            L.append(f"    [{chr(65 + i)}]  b={b}  r={r}  s={s}{marca}  {motivo}")
    else:
        L.append("  Sin CCF significativa en k >= 0: no se detecta relación.")
        L.append("  Propuesta: b=0, r=0, s=0 (sin transferencia).")
    return "\n".join(L)
