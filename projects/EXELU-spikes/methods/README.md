# Methods

Full derivations for the analyses that will be scrutinised in detail — the reasoning behind
each metric, the alternatives that were rejected, the failure modes, and what the result
cannot support.

These are the opposite of `../log/`. The log is skeletal and dated; these are long, undated
and rewritten in place whenever the method changes.

Written to be read line by line before the analysis is re-implemented in Julia. Nothing is
asserted without the reason it is true, and claims are tagged `[established]` (standard,
cited), `[contested]` (genuine method-dependence) or `[inference]` (my reasoning from this
dataset, needs checking).

| document | the question it answers |
|---|---|
| [`responder-metric-calculation.md`](responder-metric-calculation.md) | which units changed their firing rate |
| [`modulation-depth-calculation.md`](modulation-depth-calculation.md) | how strongly a unit follows each stimulus cycle, and exactly how peak and trough are detected |
| [`cell-type-classification.md`](cell-type-classification.md) | principal cell vs interneuron |

Related, kept with the code because it is a contract rather than a derivation:
[`../../../modules/FORMAT.md`](../../../modules/FORMAT.md) — the session
file format both languages implement.
