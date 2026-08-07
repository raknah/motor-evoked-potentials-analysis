# Modulation depth and the transition-triggered PSTH — full derivation

**Code:** `scripts/entrainment.py` (shared), `projects/EXELU-spikes/scripts/EXELU_validation.py`
**Notebook section:** 8
**Status of this document:** written to be read line by line before the analysis is
re-implemented in Julia. The section on peak/trough detection is the one the professor asked
for; it is section 3.

Tags: `[established]` = standard, cited. `[contested]` = genuine method-dependence.
`[inference]` = my reasoning from this dataset, needs checking.

---

## 0 · The question, and why it is not the previous question

The block PSTH asks *did the stimulus change the firing rate?* This asks *does the unit fire
at a consistent point within each flicker cycle?* They are independent: a unit can double
its rate with no cycle-by-cycle structure, and lock perfectly to the cycle while its mean
rate falls. In this session both happen.

The bin width alone forces the separation. The block PSTH uses 25 ms bins — and one 25 ms
bin **is** one 40 Hz cycle, so that histogram is structurally incapable of showing 40 Hz
following. Any within-cycle structure must be integrated away by it. This is the classic
trap: choosing a bin width that silently makes the question unanswerable. `[established]`

---

## 1 · Three layers, in increasing order of trustworthiness

### Layer 1 — the display histogram (looks only, **no number comes from it**)

Every transition is a trigger. Spikes are gathered over $\pm 1.75$ cycles and binned at
1 ms — twenty-five bins per cycle at 40 Hz.

**The windows overlap.** Triggers are one cycle apart and the window is 3.5 cycles wide, so
each spike falls inside about three and a half windows. Consequences:

- The rate axis is still correct: dividing by the number of triggers accounts for the
  multiplicity exactly.
- Adjacent cycles of the plot are **not statistically independent**, and the histogram is
  periodic **by construction** even for a completely unmodulated train. Seeing three
  repetitions of the same shape is not evidence of rhythmicity.

The 3.5 cycles are shown because the professor asked for three to four transitions to be
visible. The informative object is one cycle.

### Layer 2 — the one-cycle histogram (this is where modulation depth comes from)

Each spike is assigned to **the single transition immediately preceding it**:

$$
k(i) = \max\{k : \tau_k \le t_i\},
\qquad
\Delta_i = t_i - \tau_{k(i)},
\qquad
\text{keep } i \text{ iff } 0 \le \Delta_i < T
$$

$$
\theta_i = 2\pi \frac{\Delta_i}{T} \in [0, 2\pi)
$$

with $T$ the stimulus period. Two things happen at once and both are load-bearing:

- **Every spike is counted exactly once.** No overlap, no multiplicity.
- **Only in-block spikes survive.** A spike in the inter-block interval has
  $\Delta_i \ge T$ and is dropped without needing a separate mask.

*Implementation:* `entrainment.cycle_phases`.

### Layer 3 — the phase statistics (no histogram, no binning, no smoothing)

Computed from $\{\theta_i\}$ directly, so no binning or smoothing choice can influence them.

$$
R = \left| \sum_{i=1}^{N} e^{\mathrm{i}\theta_i} \right|,
\qquad
\mathrm{VS} = \frac{R}{N},
\qquad
\mathrm{PPC} = \frac{R^2 - N}{N(N-1)}
$$

**Vector strength** is the length of the mean unit phase vector: 0 for uniform phases, 1 for
perfect locking. `[established — Goldberg & Brown 1969; reviewed critically in Heil &
Peterson 2015, *Synapse* 71:5–36, doi:10.1002/syn.21925]`

**PPC** is the average pairwise cosine similarity between phase vectors. `[established —
Vinck et al. 2010, *NeuroImage* 51:112–122, doi:10.1016/j.neuroimage.2010.01.073]`

**Why PPC is the metric for any cortex-versus-hippocampus claim.** VS is biased upward at
low $N$. Under the null of uniform phases, $R^2$ has expectation $N$, so
$\mathbb{E}[\mathrm{VS}^2] \approx 1/N$ — a unit with 200 spikes reads VS $\approx 0.07$ from
nothing at all, and a unit with 20 000 reads $0.007$. Cortex and hippocampus here differ
systematically in firing rate, so a raw VS comparison between them is **partly a firing-rate
comparison**. PPC subtracts exactly that $N$: the numerator $R^2 - N$ has expectation zero
under the null, which is why PPC can and does go slightly negative. A negative PPC is not an
error; it is the estimator being unbiased.

**Rayleigh test.** $p \approx \exp\!\left(\sqrt{1 + 4N + 4(N^2 - R^2)} - (1 + 2N)\right)$,
the standard closed form, accurate above $N \approx 50$ and conservative below.
`[established]`

**A warning that matters for the pooled traces.** At $N \approx 10^5$ — which pooled
multi-unit trains reach — a Rayleigh p-value is a statement about *sample size*, not about
effect size. Pooled hippocampal PPC at 40 Hz here is $6 \times 10^{-5}$ with $p \approx 0$.
That is "significant" and functionally zero. **Report PPC, not p.**

---

## 2 · The folding artifact — the error this design exists to prevent

**Do not reintroduce a `% period` fold over a non-integer number of cycles.**

An earlier version built the cycle histogram by folding the *display* window modulo the
period. The display window spans 3.5 cycles. 3.5 is not a whole number, so phases below 0.75
of a cycle were covered **four** times and phases above it **three** times. A perfectly flat
spike train therefore folds to a 4:3 step and reads

$$
\mathrm{MD} = \frac{4}{3} = 1.33
$$

**from geometry alone**, with no neural modulation whatsoever.

- **How it was caught:** the pooled multi-unit trace read MD ≈ 1.33–1.37 for *every*
  condition — including phase-reversing gratings, where nothing is locked. A number that is
  the same everywhere is measuring the method, not the data.
- **How it was confirmed:** folding uniform random numbers. 3.5 cycles → 1.34.
  2, 3 or 4 cycles → 1.01.
- **Effect of the fix**, pooled cortical MUA, phase-reversing: MD 1.366 → **1.056**.
- **Never affected:** VS, PPC and Rayleigh, which are computed from per-spike phase and
  never touch the fold.
- **Guard:** `EXELU_validation.modulation_depth_guard` runs three synthetic trains through the
  identical estimator on every notebook run and raises if a flat train reads above 1.25.

---

## 3 · How peak and trough are detected — the professor's question

### 3.1 The procedure, exactly

1. **Bin the phases into one cycle.** $n_{\text{bins}} = \lfloor T / \Delta \rceil$ where
   $\Delta = 1$ ms, so 25 bins at 40 Hz and 50 at 20 Hz.

2. **Convert to a rate.** Each transition contributes exactly one cycle, so

   $$
   \rho_b = \frac{c_b}{\Delta \cdot K}
   $$

   with $c_b$ the count in bin $b$ and $K$ the number of transitions. The result is in Hz and
   directly comparable with the block PSTH.

3. **Smooth with a 3-bin circular boxcar.**

   $$
   \tilde{\rho}_b = \frac{\rho_{b-1} + \rho_b + \rho_{b+1}}{3},
   \qquad \text{indices modulo } n_{\text{bins}}
   $$

   **Circular** because phase $0$ and phase $2\pi$ are the same place. A linear filter would
   corrupt both ends of the cycle, and the ends are exactly where a transition-locked
   response sits.

   **Why smooth at all.** Without it, a single noisy bin sets the peak, and a single empty
   bin sets the trough — the estimate then reports the noisiest bin rather than the response.

   **Why only 3 bins.** The kernel is a low-pass filter, and it attenuates real structure as
   well as noise. At 40 Hz, 3 bins is 3 ms out of a 25 ms cycle — a twelfth of a cycle. Wider
   smoothing would start flattening a genuinely sharp response, which is the opposite failure
   and harder to notice.

4. **Read off the extremes.**

   $$
   P = \max_b \tilde{\rho}_b, \qquad Q = \min_b \tilde{\rho}_b
   $$

   The **global** maximum and minimum of the smoothed cycle. Not a local-maximum finder, not
   a template fit, not a peak-prominence criterion. Deliberately the simplest thing that can
   be stated in one line, because the professor's definition of modulation depth is
   $\mathrm{FR}_{\text{peak}}/\mathrm{FR}_{\text{trough}}$ and any cleverness here would make
   the reported number stop matching the definition.

*Implementation:* `entrainment.cycle_histogram`, `entrainment.modulation_depth`.

### 3.2 What is wrong with reading off extremes, quantitatively

$P$ and $Q$ are **order statistics** — the maximum and minimum of $n_{\text{bins}}$ noisy
values. Sampling noise alone pushes the maximum above the mean and the minimum below it, so
$P/Q > 1$ even with no modulation at all.

Order of magnitude. With $N$ spikes spread over $n$ bins, each bin holds $\approx N/n$
counts with Poisson standard deviation $\sqrt{N/n}$, i.e. a relative noise of
$\sqrt{n/N}$. Three-bin smoothing divides that by $\sqrt{3}$. The expected spread between
the max and min of $n$ approximately normal values is around $3.5\sigma$ for $n = 25$, so

$$
\frac{P}{Q} \approx 1 + 3.5\sqrt{\frac{n}{3N}}
$$

The guard confirms it: an unmodulated Poisson train with $N = 19\,200$ over 25 bins reads
$\mathrm{MD} = 1.152$, against a prediction of $1 + 3.5\sqrt{25/57600} = 1.073$ — the right
order, and the residual is because the bin counts are not quite normal and the smoothing
introduces correlation between neighbouring bins. `[inference — the scaling argument is
standard, the specific coefficient is fitted to this session's guard output]`

**The consequence, stated plainly.** $\mathrm{MD}$ *decreases with firing rate* even when
nothing about the neuron changes. Units in this dataset differ in rate by more than 400×.
**A modulation-depth comparison between a cortical interneuron and a dentate granule cell is
substantially a firing-rate comparison.** This is the same failure mode as raw vector
strength, for the same reason.

`tt_min_spikes = 200` suppresses the metric below 200 in-cycle spikes, which caps the worst
of it, but does not remove the trend above that threshold.

### 3.3 The three modulation depths, and which to quote

| column | formula | bounded? | peak-picking? | biased by $N$? |
|---|---|---|---|---|
| `mod_depth` | $P/Q$ | no — diverges as $Q \to 0$ | yes | yes, strongly |
| `mod_depth_norm` | $(P-Q)/(P+Q)$ | $[0,1]$ | yes | yes, less |
| `mod_depth_fourier` | $2 \cdot \mathrm{VS}$ | $[0,2]$ | **no** | mildly (via VS) |
| `ppc` | $(R^2-N)/(N(N-1))$ | — | **no** | **no** |

**Where `mod_depth_fourier` comes from.** The classical definition of modulation depth is
the amplitude of the first Fourier harmonic of the cycle histogram divided by its mean. For a
point process the first harmonic of the phase distribution is

$$
a_1 = \frac{1}{N}\sum_i e^{\mathrm{i}\theta_i}
$$

and the cycle histogram, written as a Fourier series with mean $\rho_0$, has first-harmonic
amplitude $2\rho_0 |a_1|$. Dividing by the mean:

$$
\frac{\text{first-harmonic amplitude}}{\text{mean}} = 2|a_1| = 2\,\mathrm{VS}
$$

So the classical modulation depth **is exactly twice the vector strength**, requires no peak
picking, and is therefore free of the order-statistic bias. For a perfectly sinusoidal
response it equals $(P-Q)/(P+Q)$; for a sharp, non-sinusoidal one it is smaller, because the
energy has gone into higher harmonics.

**Recommendation:**

- quote **`mod_depth`** where the professor's definition is what is being reported;
- quote **`mod_depth_norm`** for anything averaged over units;
- quote **`ppc`** for any cortex-versus-hippocampus claim — that is the one that survives the
  firing-rate objection.

---

## 4 · The vector-strength blind spot, and the control for it

**The failure mode.** `[established]` VS reads $\approx 0$ for clean **n:1 locking**. A unit
firing twice per cycle at opposite phases contributes phase vectors at $\theta$ and
$\theta + \pi$, which cancel exactly. The unit is perfectly entrained and the metric says
nothing is happening.

**Why this is live rather than hypothetical here.** The analog line logs 9 600 events per
60 blocks of 4 s = 40 Hz for a "40 Hz" flicker whose screen physically changes **twice** per
cycle — on and off. One logged edge per physical cycle-pair means a unit responding to both
edges is exactly the cancelling case.

**The correct diagnostic.** Not "which harmonic is larger on average" — for an unmodulated
unit both harmonics are noise and that comparison is a coin flip carrying no information.
(An earlier version asked exactly that and got 46 %, which is a coin flip *by construction*.)

The right question is: **is any unit significantly locked at $2f_0$ while not locked at
$f_0$?** That is the only signature of cancellation. Recompute $\theta$ against $T/2$ and
run the same Rayleigh test.

**Answer in this session: zero units, at either frequency.** So the weak 40 Hz locking is a
real failure to follow the flicker cycle by cycle, not an artefact of the metric. At 20 Hz
several units are significant at both harmonics, which is simply the harmonic content of a
sharp non-sinusoidal peak — VS at $f_0$ exceeds VS at $2f_0$ in every locked unit.

*Implementation:* `entrainment.harmonic_table`.

---

## 5 · What this analysis cannot tell you

- **Preferred phase is not interpretable in absolute terms.** `photodiode` is bit-identical
  to `flicker_20Hz`, so display latency is unmeasured. Relative phase *between units* is
  fine; absolute phase relative to the screen is not. *Blocking for latency claims.*
- **This is spike-to-stimulus phase, not spike-field coherence.** There is no LFP in this
  dataset. The published 40 Hz results this is being compared against are largely **LFP power**
  results, and LFP power and single-unit phase locking routinely disagree — that disagreement
  is the substance of the field's controversy, not a flaw in either measurement.
  `[contested — Soula et al. 2023 vs Adaikkan et al. 2019]`
- **A null result here is a null for spike entrainment only.** It does not bear on the
  amyloid claims, which have proposed non-oscillatory mechanisms.
- **No shuffle control**, as instructed. The Rayleigh test provides an analytic null; a
  trigger-shuffle null would additionally absorb any periodicity in the recording that is not
  stimulus-driven.

---

## 6 · Current numbers, this session (`output/2026-04-14_11-46-55/transition_good.csv`)

Reliable units only ($\ge 200$ in-cycle spikes), Rayleigh at BH-FDR < 0.05.

| condition | region | units | locked | median PPC | median MD | median MD-norm |
|---|---|---|---|---|---|---|
| flicker_20Hz | CTX | 17 | **15** | 0.0277 | 3.63 | 0.57 |
| flicker_20Hz | HPC | 17 | 2 | −0.0002 | 1.52 | 0.21 |
| flicker_40Hz | CTX | 15 | 7 | 0.0019 | 1.45 | 0.18 |
| flicker_40Hz | HPC | 18 | 0 | −0.0002 | 1.27 | 0.12 |
| phase_reversing_1 | CTX | 14 | 0 | −0.0001 | 1.27 | 0.12 |
| phase_reversing_1 | HPC | 17 | 0 | −0.0002 | 1.23 | 0.10 |

Two things to notice, both of which need the professor's input.

1. **The hippocampal and phase-reversing median MDs sit at 1.23–1.27, not 1.00.** That is
   the order-statistic floor of section 3.2, not weak entrainment. Their median PPC is
   $\approx 0$, which is the honest reading. **This is why MD must not be quoted as evidence
   of weak modulation.**
2. **20 Hz drives cortical locking far harder than 40 Hz** — 15/17 versus 7/15, median PPC
   0.028 versus 0.002 — and 20 Hz is the nominal *control* frequency. The harmonic control
   rules out the obvious artefact. It is consistent with the temporal low-pass behaviour of
   the early visual system `[established — Schneider et al. 2023, *Cell Reports* 42:112492,
   capacitive low-pass model]` and does not contradict the 40 Hz amyloid literature, which
   rests on LFP power rather than spike phase. **Raised with the professor.**

---

## Sources

- Vinck, M. et al. (2010). The pairwise phase consistency: a bias-free measure of rhythmic
  neuronal synchronization. *NeuroImage* **51**:112–122. doi:10.1016/j.neuroimage.2010.01.073
- Heil, P. & Peterson, A. J. (2015). Basic response properties of auditory nerve fibers: a
  review. *Synapse* **71**:5–36. doi:10.1002/syn.21925 — critical review of vector strength.
- Schneider, M. et al. (2023). *Cell Reports* **42**:112492.
  doi:10.1016/j.celrep.2023.112492 — laminar PPC, LGN→V1→CA1 attenuation, capacitive
  low-pass model. Methodological template.
- Soula, M. et al. (2023). *Nature Neuroscience* **26**:570–578.
  doi:10.1038/s41593-023-01270-2 — matched design (silicon probe, V1/EC/hippocampus, 5xFAD,
  40 Hz); artifact controls.
- Adaikkan, C. et al. (2019). *Neuron* **102**:929–943. doi:10.1016/j.neuron.2019.04.011 —
  the strongest pro-entrainment claim, for direct comparison of metrics.
