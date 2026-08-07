# Cell-type classification — principal cells versus interneurons

**Code:** `scripts/waveforms.py` (shared), `projects/EXELU-spikes/scripts/EXELU_units.py`
**Notebook sections:** 4 and 5
**Why this document exists:** the label is a threshold on a single number, and thresholds
are the first thing a reviewer attacks. This states exactly which number, which threshold,
where the threshold came from, and what it cannot support.

Tags: `[established]` = standard, cited. `[contested]` = genuine method-dependence.
`[inference]` = my reasoning from this dataset, needs checking.

---

## 1 · Why waveform width separates the two classes at all

An extracellular spike waveform is the field produced by transmembrane currents at the
recording contact. It has two dominant phases:

1. a sharp **negative trough** — the sodium influx during the action potential, seen from
   outside as a current sink;
2. a slower **positive rebound** — the potassium efflux during repolarisation, seen from
   outside as a source.

The interval between them, the **trough-to-peak duration**, is therefore a direct readout of
how fast the cell repolarises. Fast-spiking parvalbumin-positive interneurons express
Kv3-family potassium channels with unusually fast kinetics — that is what lets them sustain
high firing rates without depolarisation block — so their repolarisation is fast and the
trough-to-peak interval is short. Pyramidal cells repolarise more slowly and the interval is
longer. `[established — Barthó et al. 2004, *J Neurophysiol* 92:600–608; McCormick et al.
1985]`

This is a property of the membrane, not of the animal's behavioural state, which is why it is
the primary criterion and firing rate is not.

---

## 2 · Where the waveform comes from, and what is wrong with it

**The raw voltage traces are not available.** `kilosort4/params.py` records
`dat_path = 'D:/2026-04-14_11-46-55/...dat'` — the acquisition machine. So the waveform
cannot be an average of raw snippets and must come from Kilosort's **templates.npy**.

Two consequences, both real:

1. **A template is a fitted basis reconstruction, not a mean waveform.** Kilosort represents
   every spike as a low-rank combination of learned temporal components; the template is the
   cluster's fitted shape. It is smoother than the true mean waveform, and fine features can
   be suppressed. It is what Phy plots and what a large fraction of published width-based
   classifications actually use, but it is not the same object as a raw-snippet average.
   `[established as common practice; the distinction is rarely stated]`

2. **Templates are stored in Kilosort's whitened space.** Kilosort spatially whitens the
   data before sorting; the templates inherit that transform. They are un-whitened here by
   right-multiplying with `whitening_mat_inv`, exactly as Phy does:

   $$
   W^{\text{data}}_{t,c} = \sum_{c'} W^{\text{whitened}}_{t,c'} \, M^{-1}_{c',c}
   $$

   In this session the whitening matrix is **99.4 % diagonal** (mean off-diagonal magnitude
   $1.7\times10^{-4}$ against a mean diagonal of $2.9\times10^{-2}$), so the effect on a
   single channel's time course is small. But it is not zero, and it can change *which
   channel is the peak channel*, which then changes the measured width. Across the 207
   templates the two versions correlate at $r = 0.94$ with a worst-case difference of
   0.63 ms — driven entirely by peak-channel reassignment. Un-whitened is the correct choice
   and is what is used. `[inference — verified empirically on this session]`

### 2.1 Peak channel and the measurement

- **Peak channel** = the channel with the largest peak-to-peak amplitude. Not the largest
  absolute value: some templates are positive-going on some channels (axonal or
  reference-artefact contributions), and $\max|V|$ would select a positive lobe on the wrong
  contact.
- **Trough** = the global minimum of the waveform on that channel.
- **Peak** = the maximum *after* the trough. Searching after the trough, not globally, is
  what makes this a repolarisation measurement rather than a "distance between the two
  biggest excursions" measurement.
- **Half-width** = the width of the trough at half its depth, reported as an independent
  second shape measure. It is not used for the label.

*Implementation:* `waveforms.waveform_metrics`.

### 2.2 Two waveform shapes the rule cannot handle, and what is done about them

Eight of the 45 clean units in this session have a **positive-dominant** waveform: the
positive lobe is larger than the trough. Physically that usually means the contact is not
near the soma — an axonal spike, or the return current seen from a distance. The width may
still carry information but it is no longer the clean membrane measurement the rule assumes.
These units keep their label and are marked `cell_type_consistent = False`.

One unit (u97, 0.117 Hz) has a **negative** `peak_trough_ratio`: its waveform never returns
above baseline after the trough within the 2.03 ms template window. For it the
trough-to-peak interval is not a repolarisation time at all, so the rule is not uncertain,
it is **undefined**. Such units are set to `unclassified` rather than given a label the
measurement cannot support.

Both cases are columns in `units_clean.csv` (`positive_dominant`, `repolarises`), not
silent internal state.

### 2.3 Cluster to template

Usually the cluster id and the template id coincide, and in this session
`spike_templates == spike_clusters` exactly. They diverge after manual curation, so the code
takes the **modal template of the cluster's own spikes** rather than trusting the id.

---

## 3 · The threshold

$$
\text{cell\_type} = \begin{cases}
\text{interneuron} & \text{if } t_{2p} < 0.45\ \mathrm{ms} \\
\text{principal} & \text{otherwise}
\end{cases}
$$

**Where 0.45 ms comes from.** The 0.4–0.5 ms range is the standard extracellular convention
for rodent cortex and hippocampus. `[established — Barthó et al. 2004; Sirota et al. 2008;
Stark et al. 2013; used with values in this range across the Allen Institute Neuropixels
pipeline and most Buzsáki-lab work]` It is **contested** at the level of the exact number:
different labs use 0.4, 0.43, 0.45 or 0.5 ms, and the choice moves a handful of units in any
dataset. `[contested]`

**Why the threshold is fixed rather than fitted.** A per-session fitted threshold would make
the label mean something slightly different in every session, which destroys the ability to
pool 15 sessions. A fixed literature value keeps the label comparable at the cost of being
slightly wrong in each session — the right trade for a cross-session design.

**But the data is checked against it.** `waveforms.kde_antimodes` finds every prominent dip in
a kernel density estimate of the width distribution, where "prominent" means the dip's
density is below 0.75 × the smaller of the two peaks flanking it. This session returns

$$
\{0.383,\ 0.834,\ 0.970\}\ \mathrm{ms}
$$

The first, **0.38 ms**, sits close to the 0.45 ms default and corroborates it. `[inference]`
The other two separate wide from very-wide waveforms — a different distinction, possibly
cortical versus hippocampal pyramidal morphology, and **not** the interneuron boundary.
Reading the deepest dip as "the" split would have been wrong; this is why the function
returns all of them.

---

## 4 · The two supporting measurements, and why they do not vote

Both come from the long-window autocorrelogram (±500 ms, 1 ms bins), so they are independent
of the waveform.

### 4.1 Burst index

$$
\mathrm{BI} = \frac{\overline{\mathrm{ACG}}(3\text{–}5\ \mathrm{ms})}{\overline{\mathrm{ACG}}(200\text{–}300\ \mathrm{ms})}
$$

Numerator: spike pairs at the short lags characteristic of a burst. Denominator: the far
shoulder, outside any post-spike dynamics, which estimates the unit's own baseline pair
density. Above 1 means the unit fires in bursts. `[established — Royer et al. 2012,
*Nat Neurosci* 15:769–775]`

### 4.2 Theta modulation index

$$
\mathrm{TMI} = \frac{\overline{\mathrm{ACG}}(100\text{–}140\ \mathrm{ms}) - \overline{\mathrm{ACG}}(50\text{–}70\ \mathrm{ms})}
{\overline{\mathrm{ACG}}(100\text{–}140\ \mathrm{ms}) + \overline{\mathrm{ACG}}(50\text{–}70\ \mathrm{ms})}$$

A theta-modulated unit has an autocorrelogram side-peak at one theta period (100–140 ms,
i.e. 7–10 Hz) and a trough at half a period (50–70 ms). Bounded in $[-1, 1]$ and independent
of firing rate, because it is a ratio of two bands of the same histogram. `[established —
same source]`

### 4.3 The empty-denominator trap, and the guard

A 0.1 Hz unit contributes roughly **five** spike pairs to the entire 200–300 ms shoulder over
a 42-minute recording. Dividing by that produces burst indices in the hundreds. The first
version of this analysis reported burst indices of 275, 365 and 137 for three near-silent
hippocampal units — an artefact of an almost-empty denominator, not a bursty neuron.

**Guard:** both indices return `NaN` unless every band they use contains at least 20 spike
pairs, and a `rhythmicity_reliable` column records which units failed. Six of the 45 clean
units in this session fail. Reporting NaN is the honest answer; reporting 365 is not.

### 4.4 Why these do not enter the classifier

Firing rate and burst index depend on the animal's behavioural state, on how much it moved,
and on the recording's length. Folding them into the label would make the label
state-dependent and would hide the disagreements that are scientifically interesting. Instead
they are used only to raise a flag:

```
cell_type_consistent = False  if  (narrow AND rate < 2 Hz AND burst index > 1.5)
                              or  (wide AND rate > 15 Hz)
```

Four units in this session are flagged. They keep their waveform-based label and are ringed
in panel (b) of the cell-type figure.

---

## 5 · What the classification looks like in this session

45 clean units: **12 interneurons, 31 principal cells, 2 unclassified**
(interneuron CTX 6 / HPC 6; principal CTX 14 / HPC 17; both unclassified in HPC).
11 units carry `cell_type_consistent = False`, 8 of them because the waveform is
positive-dominant. 6 units have unreliable rhythmicity indices.

| | trough-to-peak | half-width | firing rate | burst index | theta index |
|---|---|---|---|---|---|
| interneuron | 0.233 ms | 0.150 ms | 8.27 Hz | 1.29 | −0.022 |
| principal | 0.733 ms | 0.233 ms | 3.26 Hz | 0.60 | +0.019 |

All five medians move in the predicted direction. The two that were **not** used to make the
label — firing rate 2.5× higher and burst index 2.1× higher in interneurons — move as the
physiology predicts, which is genuine independent corroboration rather than a restatement of
the threshold. `[inference, but a strong one]`

The theta index difference (−0.022 vs +0.019) is in the right direction but is **small and
should not be leant on**: it is measured on the units' own spike trains, and 6 of 45 units
do not have enough spikes for it to mean anything at all.

---

## 6 · What this cannot support

- **"Gamma-modulated interneurons" cannot be verified here.** `[blocking]` The professor's
  description of interneurons includes gamma modulation. Gamma modulation is a
  spike-to-LFP-phase property, and **there is no LFP in this dataset**. The theta index is
  rhythmicity of the unit's *own spike train*, which is a weaker and different thing. Any
  gamma-modulation claim needs the continuous LFP. This is open confirmation item 2 in the
  reference note.
- **"Interneuron" here means "narrow-waveform unit".** It is not optotagging. Schneider et
  al. (2023) show by optotagging that gamma-rhythmic flicker preferentially entrains
  fast-spiking PV+ *and* narrow-waveform Sst+ interneurons; those two are not separable by
  width alone. `[established]`
- **Some principal cells are narrow.** The waveform-width distinction has a known error rate
  of roughly 5–10 % in rodent cortex, and it is worse in hippocampus, where dentate granule
  cells and mossy cells have atypical waveforms. The 32/13 split should be read as
  approximate. `[contested]`
- **Depth is not being used as evidence.** It could be — a narrow-waveform unit in the CA1
  pyramidal layer is a different prior from one in stratum oriens — but the landmark channels
  give only four boundaries and no layer assignment within cortex. Not attempted.

---

## 7 · Open items

1. **Get the raw `.dat` onto the analysis machine** if width-based classification is going to
   carry weight. Mean raw waveforms would remove the template-smoothing caveat entirely and
   cost only disk space.
2. **Confirm whether the continuous LFP exists.** It determines whether "gamma-modulated" can
   be measured at all, and it is already the highest-ranked open question in the reference
   note.
3. **The threshold has not been checked against the other 14 sessions.** If the 0.38 ms dip
   is reproducible across sessions, that is a much better argument for the threshold than the
   literature default. One line in the batch script.

---

## Sources

- Barthó, P. et al. (2004). Characterization of neocortical principal cells and interneurons
  by network interactions and extracellular features. *J Neurophysiol* **92**:600–608.
- Royer, S. et al. (2012). Control of timing, rate and bursts of hippocampal place cells by
  dendritic and somatic inhibition. *Nat Neurosci* **15**:769–775. — burst index, theta index.
- Sirota, A. et al. (2008). Entrainment of neocortical neurons and gamma oscillations by the
  hippocampal theta rhythm. *Neuron* **60**:683–697.
- Stark, E. et al. (2013). Inhibition-induced theta resonance in cortical circuits.
  *Neuron* **80**:1263–1276.
- Schneider, M. et al. (2023). *Cell Reports* **42**:112492. — optotagged PV+/Sst+ entrainment
  by gamma-rhythmic flicker.
