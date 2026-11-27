#!/usr/bin/env python3
"""Rodaje P07: `diagnose` da la Q del ruido a un solo retardo, fijo, y la
figura de `plot_residuals` da otra Q, a otro retardo y sin p.

`school.noise_adequacy` usa k = min(24, max(4, n // 5)) y `diagnose` no tiene
argumento para cambiarlo: con datos mensuales sale siempre Q(24). La escuela
(y el AGENTS.md del curso) lee la Q a 12, 24 y 39 (3s + 3); un MA estacional
que falta se ve a 12 y a 36. `plot_residuals` dibuja la figura de pyfug con
«Q(38) = 29.9» al pie (39 retardos menos 1 parametro), sin p-valor: dos Q
distintas del mismo residuo y solo una con p. G7 acabo programando la Q en
Python con el asistente.

Sale con 1 mientras `diagnose` no de la Q del ruido a 12, 24 y 39 (cada una
con su p) y no admita `lags`.
"""
from _p07 import ES, WTI, pre  # noqa: F401,E402 (datos y codigo de salida)
import inspect, os, re, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
import drtran.mcp_server as M

M.load_pre("QR", f"{ES},{WTI}", check=False)
M.set_network("QR", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
M.estimate("QR")
txt = M.diagnose("QR", 0)
lin = re.search(r"Ruido blanco \(Q\).*", txt)
qs = re.findall(r"Q\((\d+)\) = [\d.]+, g\.l\. = \d+, p = [\d.]+", lin.group(0)) if lin else []
params = list(inspect.signature(M.diagnose).parameters)
print("parametros de diagnose:", params)
print("Q del ruido que da diagnose, a los retardos:", qs)
print("esperado: Q(12), Q(24) y Q(39) en mensual (o un argumento `lags`), "
      "cada una con su p, como la figura")
# arreglado (2026-11-27): Q(12), Q(24) y Q(39), cada una con su p, y `lags`
sys.exit(0 if ({"12", "24", "39"} <= set(qs) and "lags" in params) else 1)
