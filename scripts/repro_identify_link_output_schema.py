"""BUG-4: `identify_link` is declared `-> str` but returns a LIST whenever it
manages to draw the CCF, so the MCP layer rejects its own output.

    def identify_link(name: str, input_index: int = 1, band: str = "constant",
                      ident_pre: str = "") -> str:
        ...
        return _con_figura("\\n".join(txt), grafico)

`_con_figura` returns the plain string only when there is NO figure; with one it
returns `[TextContent(...), ImageContent(...)]`, which is what the tool is for --
"EL GRAFICO VA CON LOS NUMEROS, no en otra llamada", says the comment above the
call. But FastMCP builds the structured-output schema from the return annotation,
so the declared contract is:

    identify_link -> {'properties': {'result': {'type': 'string'}}, ...}

and the list fails validation before it reaches the caller:

    Error executing tool identify_link: 1 validation error for identify_linkOutput
    result
      Input should be a valid string [type=string_type, input_value=[TextContent(...)]]

The failure is conditional in the worst possible direction: the tool works ONLY
when `plot_ccf` raises, because that is the branch that returns a string. When
everything goes right, the call fails. Node N1 -- the (b, r, s) decision -- is
the one the analyst is supposed to take by LOOKING at the CCF, so this takes out
the identification step of every transfer model built through mtram.

The sibling tools show the fix: `plot_ccf` and `plot_impulse_response` are
annotated `-> list` and get `outputSchema: None`, i.e. no structured validation,
which is why they pass the same content through untouched. `identify_link` is
the only tool that calls `_con_figura`.

Run:  python3 scripts/repro_identify_link_output_schema.py
"""
import asyncio
import os

from drtran import mcp_server as M

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PRES = ",".join([os.path.join(ROOT, "ES_CPI_m10.1.pre"),     # output
                 os.path.join(ROOT, "WTI_ar1.1.pre")])       # input
CASE = "repro_bug4"

print("=" * 78)
print("PART 1 — the declared contract vs what the function returns")
print("=" * 78)

schemas = {t.name: t.outputSchema
           for t in asyncio.run(M.mcp.list_tools())
           if t.name in ("identify_link", "plot_ccf", "plot_impulse_response")}
for name in ("identify_link", "plot_ccf", "plot_impulse_response"):
    print(f"  {name:24} outputSchema = {schemas[name]}")
print()
print("  identify_link is the only one of the three that declares a schema, and")
print("  the only one that returns content blocks through _con_figura.")
print()

M.load_pre(CASE, PRES, check=False)
raw = M.identify_link(CASE, input_index=1)
print(f"  identify_link(...) returns: {type(raw).__name__}", end="")
if isinstance(raw, list):
    print(f"  {[type(c).__name__ for c in raw]}")
else:
    print()

print()
print("=" * 78)
print("PART 2 — the symptom: the call fails through the MCP layer")
print("=" * 78)
try:
    asyncio.run(M.mcp.call_tool("identify_link", {"name": CASE, "input_index": 1}))
    print("  call_tool -> returned normally  (bug not reproduced)")
except Exception as exc:                                    # noqa: BLE001
    first = str(exc).strip().splitlines()[0]
    print(f"  call_tool -> {type(exc).__name__}: {first}")

print()
print("=" * 78)
print("PART 3 — the mechanism: it only survives when the figure FAILS")
print("=" * 78)


def _boom(*a, **k):
    raise RuntimeError("simulated plotting failure")


real_plot_ccf = M.plot_ccf
M.plot_ccf = _boom
try:
    out = asyncio.run(M.mcp.call_tool("identify_link", {"name": CASE, "input_index": 1}))
    print("  with plot_ccf broken   -> call_tool returned normally")
    print("     (the string branch of _con_figura is the only one that validates)")
except Exception as exc:                                    # noqa: BLE001
    print(f"  with plot_ccf broken   -> {type(exc).__name__}: {str(exc).splitlines()[0]}")
finally:
    M.plot_ccf = real_plot_ccf

print()
print("  So the tool is available exactly when it cannot do its job, and")
print("  unavailable when it can. Fix: annotate `-> list`, as plot_ccf and")
print("  plot_impulse_response already are.")
