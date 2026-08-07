# 2026-08-07 · review round 2 (points 1–9)

Session `2026-04-14_11-46-55`. Analysis moved out of the notebook into plain `.py` files —
no package. Notebook is narrative plus calls.

## Layout

Split by **reusability**, enforced by one rule: *a shared module may not import a project
module.* That forced real changes, not file moves — shared functions take `fs` explicitly
instead of importing a constant, take plain dicts instead of `Params`/`Trigger`, and return
only what they compute (region/depth/cell type are joined back on by
`units.attach_metadata`).

| where | what |
|---|---|
| `neuroelectrophysiology/scripts/` | `spiketrains`, `correlograms`, `contamination`, `waveforms`, `psth`, `responders`, `entrainment`, `statstools`, `kilosort_io`, `plotstyle` |
| `projects/EXELU-spikes/scripts/` | `config`, `paths`, `records`, `regions`, `triggers`, `session`, `units`, `population`, `figures`, `validation` |

## Errors

- **E5 · burst index read 275, 365, 137** for near-silent units — a 0.1 Hz unit contributes
  ~5 pairs to the 200–300 ms shoulder, so the denominator is empty. Now NaN below 20 pairs
  per band. 6/45 units affected. **Do not replace the NaN with a number.**
- **E6 · `row.pct_change` silently returned a pandas method**, not the column. Attribute access
  never raised. All colliding names now use bracket access. Worth a grep before trusting any
  `row.<name>`.
- **E7 · Kilosort's threshold is 20 %, not the 10 % used.** Empirical across all 15 sessions:
  max ContamPct on any `good` cluster 17.7–20.0, min on any `mua` 20.0. Converse does not hold
  (KS also rejects on count and amplitude). **Clean set 35 → 45 units — moves every headline
  count.** Open item.
- **E9 · width-based cell type applied to waveforms it does not describe.** 8/45 are
  positive-dominant, 2 never repolarise. Caught from the waveform insets, not the numbers —
  an argument for putting the waveform on the figure. Non-repolarising → `unclassified`;
  positive-dominant → labelled but flagged. 13/32 → **12 interneuron / 31 principal /
  2 unclassified**.
- **E3, E4 from round 1 applied.** Boundary 1080 → 1060 µm.

## Decisions

| choice | why |
|---|---|
| plain `.py`, split shared/project | the batch script must be the same code; this is not the only project that will make PSTHs |
| shared modules take `fs` explicitly | a module importing this project's `FS` is not reusable |
| shared tables carry no region/depth | those are this experiment's concepts |
| clean set at Kilosort's own 20 % | the set is then exactly "what KS called a single unit" |
| per-trial baseline **subtraction**, pooled SD | per-trial SD is 0 for most trials of most units |
| responder = paired Wilcoxon on trial-wise rates, BH-FDR | trials are paired; counts are skewed |
| rates, not counts | windows are 2 s and 4 s — counts would find significance from length |
| permutation as confirmation only | its p floors at 1/(n+1); cannot be FDR-corrected |
| cell type from waveform width alone, fixed threshold | a fitted threshold means a different label per session |
| templates un-whitened before measuring width | Phy's convention; whitening can move the peak channel |
| `FigureSaver` clears the directory first | renumbered runs otherwise interleave with the old set |

Method derivations: `../methods/`.

## Numbers

From `output/2026-04-14_11-46-55/*.csv`. Clean set 45 `good` units at 20 %.

- regions good 20 CTX / 25 HPC · mua 56 / 106
- cell type 12 interneuron / 31 principal / 2 unclassified; 11 flagged, 6 unreliable rhythmicity
- contamination 47/207 accepted at KS 20 %, 26 at Hill 10 %, 164 saturate Hill; 21/45 `good`
  disagree, all one-directional
- baseline CV median **1.02**
- responders (FDR<0.05) 20 Hz CTX 13/20, HPC 3/25 · 40 Hz CTX 9/20, HPC 0/25 ·
  phase-reversing 0/45 · static CTX 7/20, HPC 1/25
- Wilcoxon vs permutation 165/180 agree, ρ = 0.86
- locking 20 Hz CTX **15/17** PPC 0.028 · HPC 2/17 PPC −0.0002 · 40 Hz CTX 7/15 PPC 0.002 ·
  HPC 0/18 · phase-reversing 0/31
- harmonic control 0 units locked at 2f₀ without f₀
- pooled MUA 20 Hz CTX +14.9 % (p 0.029) HPC +19.1 % (p 0.003) · 40 Hz +7.7 % / +9.3 % (n.s.)
- guards: ContamPct matches KS to 0.050 pp; flat train MD 1.152, locked VS 0.863, two-peak
  VS_f₀ 0.005 / VS_2f₀ 0.768

**Two results needing a decision:** 20 Hz outperforms 40 Hz in cortex on both rate and locking
despite being the control frequency; pooled hippocampal MUA shows *larger* % upregulation than
cortex while showing zero phase locking — a rate/timing dissociation, not "hippocampus
responds".

## Open

- **O4** report §3.1 leads with Spearman ρ = 0.87; the 21/45 one-directional count is honest.
- **O5** `REPORT_2026-04-14_11-46-55.md` is stale — figure paths, 35-unit set, 10 % threshold.
- **O6** batch script not written, as instructed. `2026-04-08_14-12-45` has an empty
  particulars file and will raise on `boundary_um`; the batch must catch and skip.
- **O7** `running_wheel` (53 416 events) unused — the only way to separate visually driven from
  locomotion-driven rate change, which is the stated motivation for point 6.
- **O8** no transient-window responder variant. `stim_s=0.5` is one keyword.
- Clean set at 20 % or 10 %? See E7.
- Does the continuous LFP exist? Determines whether "gamma-modulated" is measurable at all.
