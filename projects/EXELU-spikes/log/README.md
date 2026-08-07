# Log

What was decided and why, what went wrong, what is still open. So a choice made weeks ago can
be defended without re-deriving it.

**Not read automatically.** Point an assistant at this folder at the start of a session.

## Convention

One file per working session: `YYYY-MM-DD_short-topic.md`. Newest first below.

**Skeletal.** Bullets and tables, not prose. An entry carries only what is expensive to
reconstruct:

- **Errors** — what was wrong, how it was caught, the numbers before and after. Highest value:
  these stop a corrected mistake from being reintroduced.
- **Decisions** — the choice and the reason, one line each.
- **Numbers** — current, with the file they came from, so a stale figure elsewhere is
  identifiable as stale.
- **Open items** — blocking or not, and who answers.

Explanation of *how a method works* belongs in `../methods/`, not here. Explanation of how a
function works belongs in its docstring. Do not log routine edits or anything in `git log`.

## Deliverables

| file | what |
|---|---|
| `../notebooks/EXELU-spike-EDA.ipynb` | the worked example, one session, ~50 s. Shipped unexecuted |
| `../scripts/EXELU_*.py` | this experiment only |
| `../../../scripts/*.py` | shared, reusable on any dataset |
| `../../../modules/ephyslink/` | the Session object and its format; `EphysLink.jl` is the Julia half |
| `../methods/*.md` | full derivations for the analyses that will be scrutinised |
| `../output/<session>/<session>.h5` | full working session: raw + derived |
| `../output/_results/<session>.h5` | results only, ~859 KB — what a cross-session comparison reads |
| `../output/<session>/*.csv` | the same derived tables, for spreadsheets |
| `../output/<session>/figures/*.png` | numbered in notebook order, cleared each run |
| `REPORT_2026-04-14_11-46-55.md` | report for the professor — **stale**, see O5 |

## Index

- [2026-08-07 · the session object carries the pipeline](2026-08-07_session-carries-the-pipeline.md)
- [2026-08-07 · ephyslink, and the script layout](2026-08-07_ephyslink-and-script-layout.md)
- [2026-08-07 · review round 2 (points 1–9)](2026-08-07_review-round-2.md)
- [2026-08-07 · review points 1–7](2026-08-07_review-points-1-7.md)
