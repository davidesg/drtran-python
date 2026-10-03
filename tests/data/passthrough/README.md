# Passthrough fixtures

Two univariate optima, IPC_ES and WTI, monthly 02/2002–12/2019, from
`levels_2002_2019.csv` (passthrough study). They are fixed points of fue.

- `WTI_autonomo.pre` — WTI as modelled in art's **autonomous lane**
  (2026-10-03, art 0.2.3.dev0 after BUG-0199..0203). This is the model the
  analyst would hand to mtram:
  - λ=0, d=1, D=0, no harmonics, μ;
  - AR(1) noise with φ=+0.138;
  - two permanent steps with three ω each: 10/2008 (the post-Lehman crash)
    and 11/2014 (OPEC's decision not to cut);
  - ℓ=−730.657424.

  The lane chose AR(1) over the AR(3) and MA(1) it tied with, by the domain's
  expectation of gradual adjustment (φ>0) and by BIC. Its guion recorded every
  node. In the run, the ladder's rung 2 for 2014 was built by hand, because
  `suggest_intervention_form(form="auto")` built a 1-ω step instead (an art
  defect, reported separately).
- `IPC_ES_cadena.pre` — IPC_ES as the chain test builds it (`art.batch_build`,
  then fue): λ=0, d=1, 11 harmonics; ℓ=−7.297333.

Tests: `tests/test_passthrough_fixtures.py`.
