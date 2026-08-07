# The responder metric — full derivation

**Code:** `scripts/responders.py`, `scripts/statstools.py` (both shared)
**Notebook section:** 7
**Status of this document:** written to be read line by line before the analysis is
re-implemented in Julia. Nothing here is asserted without the reason it is true.

Tags: `[established]` = standard, cited. `[contested]` = genuine method-dependence.
`[inference]` = my reasoning from this dataset, needs checking.

---

## 0 · The question, stated precisely

> Which units changed their firing rate when the stimulus came on?

Three separate decisions hide inside that sentence, and each one has been got wrong in a
published paper at some point:

1. **Changed relative to what?** A unit's own baseline, or the other units?
2. **Changed by how much, measured how?** A difference, a ratio, or a standardised score?
3. **Changed *significantly* by what test?** And significant against which null?

The rest of this document answers those three in order, then states what the answer cannot
support.

---

## 1 · The raw material: two numbers per trial

For each unit $u$ and condition $c$ there are $T \approx 60$ trials. For trial $t$:

$$
n^{\text{pre}}_{t} = \#\{\text{spikes in } [\text{onset}_t - 2\,\mathrm{s},\ \text{onset}_t)\},
\qquad
n^{\text{stim}}_{t} = \#\{\text{spikes in } [\text{onset}_t,\ \text{onset}_t + 4\,\mathrm{s})\}
$$

The windows are **half-open** so that consecutive windows partition time and a spike landing
exactly on the boundary is counted once.

They are then converted to **rates**, not left as counts:

$$
r^{\text{pre}}_{t} = \frac{n^{\text{pre}}_{t}}{2\,\mathrm{s}},
\qquad
r^{\text{stim}}_{t} = \frac{n^{\text{stim}}_{t}}{4\,\mathrm{s}}
$$

**Why this matters and is not pedantry.** The two windows have different lengths. A paired
test on raw counts would find $n^{\text{stim}} > n^{\text{pre}}$ for *every unit in the
dataset*, with a vanishing p-value, and the effect would be entirely the factor of two in
window length. Any counts-based responder analysis on unequal windows is simply wrong.

*Implementation:* `responders.trial_rates`.

---

## 2 · Why a z-score, and against what

### 2.1 The problem it solves

Units in this dataset differ in firing rate by a factor of more than 400 — from 0.09 Hz to
37 Hz. A raw change of $+2$ Hz is a rounding error for the fastest unit and a fivefold
increase for the slowest. Some common scale is needed before "which units responded most"
means anything.

### 2.2 The choice of reference distribution — this is the whole argument

A z-score is $(x - \mu)/\sigma$. Everything depends on what $\mu$ and $\sigma$ describe.

**Option A — the population of units** (the professor's earlier rule).

$$
r_u = 100 \cdot \frac{\mathrm{FR}^{\text{stim}}_u - \mathrm{FR}^{\text{pre}}_u}{\mathrm{FR}^{\text{pre}}_u},
\qquad
z_u = \frac{r_u - \bar r}{\mathrm{SD}(r)},
\qquad
\text{responder} \iff |z_u| > 2
$$

Three failure modes, all structural rather than bad luck:

- **It assumes most units do not respond.** $\mathrm{SD}(r)$ is computed over all units
  including the responders. If half of V1 responds at 40 Hz — the expected situation — the
  responders inflate the SD, and can then fall below their own threshold. The metric
  actively hides a strong, widespread effect. `[established]` This is the standard
  criticism of population-relative thresholds.
- **It manufactures responders from noise.** Any finite sample has a tail. Applying
  $|z| > 2$ to 45 pure-noise values returns 1–2 "responders" roughly every time, and no
  amount of data fixes it, because the threshold is defined relative to the sample.
- **The ratio is unstable.** $\mathrm{FR}^{\text{pre}}$ is in the denominator, and several
  clean units here fire below 0.2 Hz — a handful of spikes over the whole baseline. $r_u$ is
  bounded below at $-100\%$ and unbounded above, so its distribution is strongly
  right-skewed and a z-score, which presumes symmetry, is not a meaningful summary of it.

In this session the population rule flags **2 units** at 20 Hz in cortex while the
per-unit test flags **13**, and 15 of 17 are phase-locked at that frequency. The population
rule is not being conservative; it is looking in the wrong place. `[inference, but the
15/17 locking figure is independent corroboration]`

**Option B — the unit's own baseline trials.** Used here.

$$
\mu_u = \frac{1}{T}\sum_t r^{\text{pre}}_{u,t},
\qquad
\sigma_u = \max\!\left(\mathrm{SD}_t\!\left(r^{\text{pre}}_{u,t}\right),\ \sqrt{\mu_u / w}\right),
\qquad
z^{\text{stim}}_{u,t} = \frac{r^{\text{stim}}_{u,t} - \mu_u}{\sigma_u}
$$

with $w = 2\,\mathrm{s}$ the baseline window length. Now "z = 2" means *two of this unit's
own baseline standard deviations* — a statement about that neuron, independent of what the
other 44 are doing, and therefore comparable across units, regions, sessions and animals.

*Implementation:* `responders.z_against_baseline`.

### 2.3 The Poisson floor on $\sigma$

$\sigma_u$ is floored at $\sqrt{\mu_u / w}$. **Derivation:** if the spike train were a
homogeneous Poisson process of rate $\mu$, the count in a window of length $w$ has mean
$\mu w$ and — a defining property of the Poisson distribution — variance also $\mu w$. The
rate estimate $n/w$ therefore has variance $\mu w / w^2 = \mu / w$ and standard deviation
$\sqrt{\mu/w}$.

**Why a floor is needed.** A unit firing at 0.1 Hz produces 0 spikes in its 2 s baseline on
about 82 % of trials. If it happens to produce 0 on *all* 60, the empirical SD is exactly 0
and $z$ is $\pm\infty$. The floor says: you cannot claim to have measured a spread smaller
than counting noise alone would produce. It only ever *raises* $\sigma$, so it can only ever
make the analysis more conservative. `[established — this is the standard shot-noise floor]`

---

## 3 · The test

### 3.1 Paired, because the trials are paired

$r^{\text{pre}}_t$ and $r^{\text{stim}}_t$ come from the same trial of the same unit,
seconds apart. They share the animal's state on that trial, which — as section 6 of the
notebook shows — varies enormously: the median coefficient of variation of the per-trial
baseline is **1.02**, so a unit's baseline routinely doubles or halves between trials.

An unpaired test would treat that shared variation as noise and lose almost all its power.
The paired test differences it away. `[established]`

### 3.2 Non-parametric, because counts are not normal

Trial-wise spike counts are non-negative integers, right-skewed, and heteroscedastic — for a
Poisson-like process the variance grows with the mean, so the stimulus period has both a
different mean *and* a different variance. A paired $t$-test assumes normal differences with
constant variance; both assumptions fail, and they fail worst for exactly the low-rate
hippocampal units where the scientific question is most contested.

**Wilcoxon signed-rank** replaces the differences with their signed ranks, which requires
only that the null distribution of differences be **symmetric about zero**. `[established]`

`zero_method="zsplit"` keeps tied pairs in the ranking rather than discarding them. Ties are
common here — a low-rate unit produces the same count in both windows on many trials.
Discarding them (the default `"wilcox"`) shrinks $n$ and inflates significance; splitting
them keeps every trial and is the conservative choice.

### 3.3 The permutation confirmation

Signed-rank still assumes symmetry. A **sign-flip permutation test** assumes only that,
under the null, the sign of each paired difference is equally likely to be $+$ or $-$ —
which is precisely what "the stimulus did nothing" means.

Procedure: take $d_t = r^{\text{stim}}_t - r^{\text{pre}}_t$; draw $10^4$ independent sign
vectors $s \in \{-1, +1\}^T$; the null distribution is $|\overline{s \odot d}|$; the
two-sided p-value is

$$
p = \frac{\#\{\text{null} \ge |\bar d|\} + 1}{10^4 + 1}
$$

The $+1$ in numerator and denominator is not cosmetic — it includes the observed
arrangement in its own null, which is what makes the test exact rather than anti-conservative
at small counts. `[established — Phipson & Smyth 2010, *Stat Appl Genet Mol Biol* 9:39]`

**Why it confirms rather than replaces.** A permutation p-value cannot fall below
$1/(10^4+1) \approx 10^{-4}$. After correcting across 45 units the smallest achievable
corrected value is bounded, so strongly responding units would all pile up at the floor and
become unrankable. The Wilcoxon carries the ranking; the permutation test says whether to
believe it.

**What the comparison found in this session:** across 180 unit × condition cells, the two
agree on 165. Ten are significant by Wilcoxon only, five by permutation only, Spearman
$\rho = 0.86$ between the p-values. No systematic direction, so the Wilcoxon's symmetry
assumption is not obviously being violated. `[inference from this session only]`

*Implementation:* `statstools.paired_wilcoxon`, `statstools.sign_flip_permutation`.

### 3.4 Multiple comparisons

45 units × 4 conditions = 180 tests. At $\alpha = 0.05$ with nothing happening, 9 come back
"significant".

**Benjamini–Hochberg** controls the *false discovery rate*: the expected proportion of the
rejected hypotheses that are false. That is the right guarantee for "which units responded".
Bonferroni controls the probability of *any* false positive, which is the right guarantee for
a single confirmatory claim and is far too strict here — it would cost most of the real
effects. `[established — Benjamini & Hochberg 1995, *JRSS-B* 57:289]`

**The family is one condition.** The correction is applied within each condition separately,
because the question asked is "which units responded to 40 Hz flicker", not "which unit ×
condition combinations are unusual". Pooling all 180 into one family would be answering a
question nobody asked and would cost power for no gain.

*Implementation:* `statstools.benjamini_hochberg`.

---

## 4 · What is reported

| column | definition | read it for |
|---|---|---|
| `fr_pre_hz`, `fr_stim_hz` | mean rate over trials in each window | the raw fact |
| `delta_hz` | $\mathrm{FR}^{\text{stim}} - \mathrm{FR}^{\text{pre}}$ | interpretable, but scale-dependent |
| `pct_change` | $100(\mathrm{FR}^{\text{stim}} - \mathrm{FR}^{\text{pre}})/\mathrm{FR}^{\text{pre}}$ | scale-free, unstable at low rate |
| `mod_index` | $(S-P)/(S+P) \in [-1, 1]$ | bounded, well-behaved at low rate — **the effect size to quote** |
| `z_stim` | mean of $z^{\text{stim}}_{t}$ | how far the response stands out of this unit's own trial-to-trial variability |
| `p_wilcoxon`, `q_wilcoxon` | raw and BH-corrected | the decision |
| `p_permutation` | sign-flip | whether to believe the decision |
| `responder` | `q_wilcoxon < 0.05` | the headline |
| `direction` | sign of `delta_hz`, `"none"` if not a responder | up or down |
| `responder_legacy` | the old population z-rule | comparison only |

**`mod_index` rather than `pct_change` as the effect size.** $(S-P)/(S+P)$ is bounded in
$[-1, 1]$, symmetric under swapping the two windows, and finite whenever either window has a
spike. `pct_change` is unbounded above, bounded at $-100\%$ below, and undefined when
$\mathrm{FR}^{\text{pre}} = 0$. Both are reported because the professor's phrasing uses
percent change; only one is safe to average across units.

---

## 5 · What this metric cannot tell you

Stated explicitly so it does not get overclaimed in a report.

- **A rate change is not entrainment.** `[established]` A unit can double its rate with no
  temporal relationship to the flicker cycle whatsoever, and a unit can lock perfectly to
  the cycle while its mean rate falls. Both happen in this session: at 20 Hz in cortex the
  median rate change is $-13.6\%$ while 15 of 17 units are phase-locked, and the two most
  strongly locked units change rate by $+61\%$ and $-56\%$. Rate and timing are separate
  questions and are answered by separate analyses. See
  `modulation-depth-calculation.md`.
- **A responder is not necessarily driven by the stimulus.** The mouse can move in response
  to the screen changing. Without the running-wheel channel in the model, a
  locomotion-driven rate change is indistinguishable from a visually driven one.
  `[inference — the `running_wheel` analog channel exists (53 416 events) and is not yet
  used. This is the single most valuable unused variable in the dataset.]`
- **Direction is a per-unit statement, not a population one.** "7 up, 6 down" does not mean
  the population did nothing; it means the population response is heterogeneous, which is
  itself a result.
- **Absolute latency is not available.** The `photodiode` channel is bit-identical to
  `flicker_20Hz`, so there is no independent measurement of when the screen actually
  changed. Nothing here supports a claim about response latency. *Blocking, raised with the
  professor.*

---

## 6 · Open items on this metric

1. **Shuffle control not implemented, as instructed.** It remains the correct null: draw
   pseudo-trigger times from the non-exposure periods, run the identical pipeline, and read
   off the empirical false-positive rate. That would replace both the population z-score and
   the parametric assumptions of the Wilcoxon with something that has the right error rate
   by construction.
2. **Locomotion is not controlled for.** See above. The cheapest version is to split trials
   by whether the wheel moved in the pre-window and check that responders survive within the
   stationary subset.
3. **The window pair is fixed at (2 s pre, 4 s stim).** A unit with a fast transient that
   adapts within 500 ms is diluted by a factor of eight over a 4 s average. The block PSTH
   shows exactly such transients in the pooled cortical trace at 20 Hz. A transient-window
   variant (`stim_s=0.5`) is one keyword argument away and has not been run.

---

## Sources

- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS-B*
  **57**:289–300.
- Phipson, B. & Smyth, G. K. (2010). Permutation p-values should never be zero.
  *Stat Appl Genet Mol Biol* **9**:39.
- Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*
  **1**:80–83.
- Soula, M. et al. (2023). *Nature Neuroscience* **26**:570–578. doi:10.1038/s41593-023-01270-2
  — for the framing that firing-rate modulation and phase locking must be reported
  separately, and for the region-wise rate statistics this analysis is comparable to.
