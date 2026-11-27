"""BUG-79: write_inp no pisa lo que escribió antes y deja el modelo conjunto
(.out, .dag, .cns) de modo que la CLI lo reproduce."""
import os
import re
import subprocess
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "tests", "data", "p07")
ES = os.path.join(D, "es_cpi_m10_clase.pre")
WTI = os.path.join(D, "wti_ar1_clase.pre")


def test_repro_cerrado():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts",
                                                     "repro_p07_ficheros_conjunto.py")],
                       capture_output=True, text=True, timeout=1200,
                       env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


def test_conjunto_reproducible(tmp_path):
    warnings.filterwarnings("ignore")
    import drtran.mcp_server as M

    cns = tmp_path / "in.cns"
    cns.write_text("omega1[2] = 0\nq[2,1] = free\n")
    M.load_pre("RT", f"{ES},{WTI}", check=False)
    M.set_network("RT", '[{"out":0,"inp":1,"b":0,"r":0,"s":2}]')
    M.estimate("RT", cns_path=str(cns))
    ll = M._FITS["RT"].loglik
    txt = M.write_inp("RT", outdir=str(tmp_path))
    dag, cnsw = tmp_path / "RT.dag", tmp_path / "RT.cns"
    assert dag.exists() and cnsw.exists() and (tmp_path / "RT.out").exists()
    assert "omega1[2] = 0.0" in cnsw.read_text() and "q[2,1] = free" in cnsw.read_text()
    assert str(dag) in txt
    r = subprocess.run([sys.executable, "-m", "drtran.cli", ES, WTI, "-n", str(dag),
                        "-c", str(cnsw), "-o", "-"], capture_output=True, text=True,
                       timeout=900, cwd=str(tmp_path),
                       env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")})
    assert r.returncode == 0, r.stderr[-2000:]
    m = re.search(r"[Ll]og-?likelihood\s*[=:]\s*(-?[\d.]+)", r.stdout)
    assert m and abs(float(m.group(1)) - ll) < 1e-4, (m and m.group(1), ll)
