"""Recorrido de mtram de punta a punta, por la superficie MCP.

Usa el caso canonico del repo del C (ES_CPI <- WTI), que es el que esta
homologado contra el binario y contra el oraculo TASTE.
"""
import os
import sys

from drtran import mcp_server as M

CASES = "/home/david/Dropbox/SRC/drtran/tests/cases"
ES = os.path.join(CASES, "ES_CPI_m10.pre")
WTI = os.path.join(CASES, "WTI_ar1.pre")

OK, FAIL = [], []


def step(label, fn, check=None):
    try:
        out = fn()
    except Exception as e:  # noqa: BLE001
        FAIL.append((label, "EXCEPCION: %s" % str(e)[:130]))
        return None
    if check:
        try:
            why = check(out)
        except Exception as e:  # noqa: BLE001
            why = "el check reventó: %s" % str(e)[:80]
        if why:
            FAIL.append((label, why))
            return out
    OK.append(label)
    return out


if __name__ == "__main__":
    if not os.path.exists(ES):
        sys.exit("faltan los .pre canónicos del repo del C")

    step("load_pre", lambda: M.load_pre("K", f"{ES},{WTI}"),
         lambda o: None if "ES_CPI" in o or "2" in o else "no confirma la carga")

    step("identify_link (propone b,r,s)",
         lambda: M.identify_link("K", input_index=1),
         lambda o: None if ("b=" in o or "(b" in o.lower()) else "no propone (b,r,s)")

    step("set_network (fija el enlace identificado)",
         lambda: M.set_network("K", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]'),
         lambda o: None if "1" in o else "no confirma el enlace")

    step("estimate", lambda: M.estimate("K"),
         lambda o: None if "log" in o.lower() else "sin verosimilitud")

    step("estimate: dice por qué paró",
         lambda: M.estimate("K"),
         lambda o: None if ("termcode" in o.lower() or "converg" in o.lower())
                   else "no reporta la convergencia")

    step("diagnose", lambda: M.diagnose("K"),
         lambda o: None if ("Q" in o or "adecua" in o.lower()) else "sin portmanteau")

    step("impulse_response", lambda: M.impulse_response("K"),
         lambda o: None if ("nu" in o.lower() or "ganancia" in o.lower()
                            or "gain" in o.lower()) else "sin nu(k)/ganancia")

    step("variance_decomposition", lambda: M.variance_decomposition("K"),
         lambda o: None if "%" in o else "sin porcentajes")

    step("forecast", lambda: M.forecast("K", horizon=6),
         lambda o: None if ("95" in o or "banda" in o.lower()) else "sin banda")

    step("evaluate (rolling)", lambda: M.evaluate("K", window=120, horizon=3),
         lambda o: None if ("RMSE" in o.upper() or "MAE" in o.upper())
                   else "sin métricas de error")

    step("build_model (informe completo)", lambda: M.build_model("K", horizon=6),
         lambda o: None if len(o) > 200 else "informe demasiado corto")

    print("\n%d pasos OK" % len(OK))
    for lab in OK:
        print("   ok   %s" % lab)
    if FAIL:
        print("\n%d FALLOS:" % len(FAIL))
        for lab, why in FAIL:
            print("   FALLO  %-40s %s" % (lab, why))
    sys.exit(1 if FAIL else 0)
