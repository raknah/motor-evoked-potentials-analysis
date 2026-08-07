# 2026-08-07 · review points 1–7

Session `2026-04-14_11-46-55`. Point 8 (shuffle control) not implemented, as instructed.
Superseded in most respects by `2026-08-07_review-round-2.md`.

## Errors

- **E1 · folding artifact inflated every modulation depth.** Peak/trough was read from a
  3.5-cycle window folded modulo the period. 3.5 is not a whole number → 4:3 coverage step →
  a flat train read MD 1.33 from geometry. Caught because pooled MUA read ~1.35 for *every*
  condition. Fixed by building the cycle histogram from `cycle_phases`. Pooled CTX
  phase-reversing: 1.366 → **1.056**. VS/PPC/Rayleigh never affected.
  **Do not reintroduce a `% period` fold over a non-integer number of cycles.**
- **E2 · harmonic control asked the wrong question.** "Which harmonic is larger?" is a coin
  flip for an unmodulated unit. Correct test: locked at 2f₀ but *not* f₀. Answer: zero units.
- **E3 · `units` self-merge** — cell was not idempotent. Fixed in round 2.
- **E4 · report claimed no unit sits near the CTX/HPC boundary.** False. `thetaCh` is 1-based;
  boundary 1080 → 1060 µm moves u77 (20 Hz-locked) and u76 across. Fixed in round 2.

## Decisions

| choice | why |
|---|---|
| pitch 20 µm, not 30 | `channel_positions.npy` runs 0, 20 … 2580 |
| KS `ContamPct` as the QC metric | it is what assigned the `good`/`mua` labels |
| re-implement KS ContamPct rather than use a library | no library computes KS's R₁₂ |
| PPC for any CTX-vs-HPC claim | VS is biased by spike count; the regions differ in rate |
| block PSTH 25 ms, entrainment 1 ms | one 25 ms bin *is* one 40 Hz cycle |
| static ON = event followed 4.000–4.002 s later by its OFF | events are ON/OFF pairs |
| notebook shipped unexecuted | executed retina figures made a ~100 MB blob |

## Waiting on the professor

1. `photodiode` is bit-identical to `flicker_20Hz` — no latency claim is defensible.
2. Nothing follows the phase-reversing grating (0/44 units). Is that TTL what we think?
3. 20 Hz drives cortex harder than 40 Hz, and 20 Hz is the nominal control.
4. Settling cutoff assumed 300 s.
5. Isolated events at 283–348 s assumed calibration, dropped.
