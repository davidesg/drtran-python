#!/usr/bin/env python3
"""Generate the MCP tool reference FROM THE DOCSTRINGS.

    python3 tools/gen_tools_md.py art.mcp_server art docs/TOOLS.md

Why generated and not written beside them: in an MCP server the docstring is
what the MODEL reads, so the documentation has two audiences — the analyst and
Claude — and the failure mode is DIVERGENCE. A rule that lives in TOOLS.md and
not in the docstring is a rule the model never applies; a docstring promising
what the tool no longer does is a lie told to the analyst through the model.
Generating removes the possibility.

It reads the REGISTERED tools, not the source, so a tool that exists and was
never registered shows up as missing here too — that has happened (BUG-4).
"""
import asyncio
import importlib
import sys


def render(mod_name: str, server: str) -> str:
    mod = importlib.import_module(mod_name)
    tools = sorted(asyncio.run(mod.mcp.list_tools()), key=lambda t: t.name)

    out = [f"# `{server}` — MCP tool reference", "",
           f"*Generated from the docstrings by `tools/gen_tools_md.py`. Do not "
           f"edit by hand — edit the docstring.*", "",
           f"**{len(tools)} tools.** In an MCP server the docstring is what the "
           f"model reads, so this page and the instruction the model receives "
           f"are the same text by construction.", "", "---", ""]

    out.append("| tool | what it answers |")
    out.append("|---|---|")
    for t in tools:
        first = (t.description or "").strip().split("\n")[0] or "—"
        out.append(f"| [`{t.name}`](#{t.name.replace('_', '-')}) | {first} |")
    out += ["", "---", ""]

    for t in tools:
        out.append(f"## `{t.name}`")
        out.append("")
        props = ((t.inputSchema or {}).get("properties") or {})
        req = set((t.inputSchema or {}).get("required") or [])
        if props:
            out.append("**Arguments**")
            out.append("")
            out.append("| name | type | required | default |")
            out.append("|---|---|---|---|")
            for n, sp in props.items():
                out.append("| `%s` | %s | %s | %s |" % (
                    n, sp.get("type", "—"), "yes" if n in req else "no",
                    "`%s`" % sp["default"] if "default" in sp else "—"))
            out.append("")
        body = (t.description or "").strip()
        if body:
            out.append(body)
            out.append("")
        out.append("---")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    mod_name, server, dest = sys.argv[1:]
    open(dest, "w", encoding="utf-8").write(render(mod_name, server))
    print("written %s" % dest)
