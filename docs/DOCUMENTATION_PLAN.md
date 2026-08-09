# Documentation: the diagnosis, and a proposal

*Everything about `drtran` and `mtram` should be documented and reachable, by
analysts and by developers. This is what is wrong, what exists, and what I
propose. The decisions marked **[David]** are not mine to take.*

---

## 1. Why the PyPI links are broken — verified, not guessed

```
drtran-python   https://github.com/davidesg/drtran-python     404
drtran (C)      git@github.com:davidesg/drtran                (SSH, not checkable)
art-python      https://github.com/davidesg/art-python        200
```

**The repository is private.** Pushes work because they authenticate; anonymous
visitors — and PyPI's readers — get a 404. Two consequences, and the second is
the one that is easy to miss:

1. `[project.urls]` declares Homepage, Repository and Issues, and **all three are
   dead** for anyone who is not David.
2. `README.md` links to `docs/PORTE.md` and `TODO.md` **relatively**. PyPI
   renders the README on its own page and resolves those against the project's
   Homepage — which 404s. So even if the repo were public, the doc links would
   need to be absolute, or they break the moment the README leaves the repo.

`art-python` is public and its links work. That is the whole difference, and it
is why `art-tseries` looks fine on PyPI and `drtran` does not.

---

## 2. What exists, sorted by who it is for

There is a lot of good material. It is not missing — it is unreachable and
unsorted.

**For ANALYSTS** — how to build a model and how to read what comes out:

| document | where | what it gives |
|---|---|---|
| `SCHOOL_PRACTICE_STUDY.md` | py | the school's practice, tier by tier |
| `DECISION_NODES.md` | py | where the analyst decides, and on what |
| `LADDER_AS_OPTIMISATION.md` | py | what `.inp` / `.out` / `.pre` guarantee — the conceptual core |
| `FORECAST_DIAGNOSIS.md` + `.tex/.pdf` | C | the out-of-sample criterion and its Diebold-Mariano test |
| `CASO_EMPLEO.md`, `M6_*.md` | C | the worked example (Relloso's employment model) |
| `drtran-note.tex/.pdf` | C | the technical note |

**For DEVELOPERS** — how it is built and what may not be broken:

| document | where |
|---|---|
| `ARCHITECTURE_MCP.md`, `PORTE.md` | py |
| `BUGS.md` | py — the honest record, including the ones that were NOT bugs |
| `LEVEL_TRANSFER_PLAN.md`, `PASSTHROUGH_MEG_BANK.md` | py — BUG-8 end to end |
| `OPTIMIZER_STOPPING_STUDY.md`, `FUF_FORECAST_BUG.md` | C |

**Working notes, NOT for publication as they stand**: `SESION_*.md` in the C
repo. They are dated logs and reading them as documentation would mislead.

---

## 3. The proposal

### 3.1 Fix the links first — it is an afternoon, not a project

* Make `[project.urls]` absolute and point at something that exists.
* **Rewrite every README link as an absolute URL.** Relative links are correct
  inside a repo and wrong everywhere the README travels — PyPI, mirrors,
  aggregators. This is worth doing whatever else is decided.
* Add `Documentation = ...` to `[project.urls]`, which neither project declares
  and which is the field PyPI shows most prominently.

### 3.2 One site for the family, not three **[David]**

`drtran` (C), `drtran-python`/`mtram`, `art`/`art-tseries` and `fue` are one
suite with one ladder, and an analyst climbing it does not care which repository
a page lives in. I propose **one documentation site covering the suite**, with
the per-package READMEs kept short and pointing into it.

Tooling: **MkDocs with Material**. Reasons: the sources are already Markdown, so
nothing is rewritten; it builds to static HTML that GitHub Pages serves for
free; and `mkdocs.yml` is the only new file. Sphinx would be the answer if the
docs were mostly API reference generated from docstrings — they are not, they
are prose with mathematics.

Proposed sections, which are just the table in §2 given an order:

```
Start here            what the suite is, the ladder in one page, install
For analysts          practice, decision nodes, the worked example, forecasting
For developers        architecture, the port, the record of defects
Reference             CLI, the MCP tools, the file formats (.inp/.out/.pre)
The record            studies: BUG-8, the optimiser, the MEG bank
```

**The file formats deserve a page of their own and do not have one.** The
`.inp`/`.out`/`.pre` convention is the suite's central contract — §1 of
`LADDER_AS_OPTIMISATION.md` states it, but as part of an argument rather than as
a specification someone can implement against.

### 3.3 What has to be decided **[David]**

1. **Public or not.** Hosted documentation for a private repository means either
   making it public or publishing the docs somewhere else. For the C there is a
   further consideration that is not a formality: `raxopt`/`qnewtopt` and `elf`
   are Mauricio's published, refereed work, and anything that ships them keeps
   his criteria and his announcements. That is a question for him, not a
   packaging decision.
2. **Ship the docs inside the wheel?** It makes them available offline and
   version-locked to the code, at the cost of wheel size. My recommendation:
   ship the analyst set, not the record.
3. **Language.** The material is mixed. A suite documented half in Spanish and
   half in English is worse than either. My recommendation: **English for the
   reference and the developer material, Spanish kept for the worked examples**
   where the sources (Relloso, Muñoz) are Spanish and the audience is the
   school's.

---

## 4. mtram is behind drtran, and here is the list

The engine grew and the MCP did not follow. Concretely:

| what drtran gained | what mtram shows | proposal |
|---|---|---|
| **the dispatch** (`effective_embed`) | `estimate(embed=True)`, and the flag is silently overridden | say so in the result: which cast ran and why |
| `delta_operator`, `check_operators` | nothing | a `check_operators` tool: Δ(1), the arm, and what it does to the gain |
| `common_window` | nothing | report the window on `load_pre`, and the spare history |
| `forecast_by_parts` | `forecast` / `plot_forecast` return the right numbers | say which route produced them |
| the backcast pre-sample | — | nothing to expose; internal |

The first is the one that matters. `estimate(embed=True)` returning a fit that
was NOT embedded, without saying so, is precisely the class of silence BUG-8
was. And `load_pre` should carry the window reconciliation of §3 in
`PASSTHROUGH_MEG_BANK.md`: it already re-estimates each series on the way in, so
trimming to the common window first is a small change and it removes a trap that
passed every existing check.

---

## 5. Order

1. Absolute URLs and a `Documentation` entry — independent of every decision
   below, and it fixes what is visibly broken today.
2. The mtram catch-up (§4). It is code, it is testable, and the documentation
   describes it afterwards rather than describing something that does not exist.
3. The formats page — the missing specification.
4. `mkdocs.yml`, the section structure, and a first build.
5. Publication, once §3.3 is decided.
