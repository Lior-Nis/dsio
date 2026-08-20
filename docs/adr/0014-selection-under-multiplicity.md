# 14. A ranking is an ordering; selection is what makes it a claim

Status: accepted (2026-08-19)
Implements: the plan's Phase 7.
Superseded (2026-08-20): `multiplicity.py` and `select.py` are deleted. Only the
effective-sample-size maths survives, in `eval/ess.py`; the selection-under-multiplicity
machinery this ADR argues for is gone.

## Context

ADR 0012 gave dsio a sweep engine, and the sweep engine manufactures its own bias. Pick the
best of N configurations and that score is systematically too high, because it is the
*maximum* of N noisy measurements rather than the value of the best configuration. Searching
harder produces a more impressive and less trustworthy winner.

The magnitude is not marginal. Simulated at the fold spread actually observed in a dsio
sweep — 0.006 — with 200 configurations that are **all genuinely identical**:

| trials | winner reports | inflation | vs fold sd |
|---|---|---|---|
| 5 | 0.9970 | +0.0070 | 1.2× |
| 20 | 1.0012 | +0.0112 | 1.9× |
| 200 | 1.0065 | +0.0165 | 2.7× |
| 1000 | 1.0095 | +0.0195 | 3.2× |

Nothing in that table is real. The observed gap between the top four runs of the sweep that
prompted this was 0.0003.

`dsio eval rank` already said a ranking is an ordering rather than a claim. This is what
makes the difference.

## Decisions

**Three gates, and the verdict names which one failed** rather than returning one opaque
boolean:

1. **Deflation** — does the winner beat what luck alone reaches across this many trials?
2. **Transfer (PBO)** — would picking the in-sample best have picked well out of sample?
3. **Evidence** — a dependence-aware interval, and an effective sample size that does not
   pretend overlapping examples are independent.

**The most valuable output is the negative one.** *"A sweep of 200 configurations found
nothing that survives the correction"* is a real answer, it is common, and a system that
cannot produce it will always hand back a winner. Verified end to end: a 12-cell sweep
varying only the random seed — so every cell is the same model — is correctly refused with
`deflated -0.0001, p=0.578, PBO 100%`.

**Trials are counted as *effectively independent*, not raw.** Configurations in a sweep share
structure and are scored on the same folds, so treating 200 correlated trials as 200
independent ones overstates the correction. Estimated from the mean pairwise correlation of
the per-block score vectors.

**The provenance is named, the finance is not imported.** Deflation, PBO/CSCV and the block
bootstrap come from the quantitative-finance literature on backtest overfitting, where this
failure is expensive and therefore well studied. Nothing here is finance-specific: the inputs
are a score matrix and a spread. What is deliberately left behind is the Deflated Sharpe
Ratio's skew and kurtosis machinery, which is about return distributions rather than about
selection.

`dsio.eval` remains a leaf: this needs numpy and `dsio.contracts` and nothing else. The
inverse normal CDF is hand-rolled for that reason and pinned against scipy in the dev
environment.

## Three things the tests caught, all real

**`survives` was a coin flip dressed as a test.** The first version gated on
`deflated > 0` — is the winner above the luck baseline. But `luck_baseline` is the *mean* of
the maximum's distribution, so a pure-noise sweep exceeds it about half the time **by
construction**. Measured: 60 pure-noise sweeps, ~half above the baseline. That is worse than
no correction, because it carries authority. The gate is now the p-value;
`test_the_luck_baseline_alone_would_be_a_coin_flip` pins why.

**The Gumbel approximation was not accurate enough.** The standard closed form for `E[max]`
is off by ~0.05 sd at n=2 and ~0.03 at n=10 — and small sweeps are exactly where a wrong
baseline flips a verdict. Replaced with numerical integration of
`E[max] = n ∫ x φ(x) Φ(x)^(n-1) dx`, accurate to ~0.002 against simulation, still fully
deterministic so two reports of one sweep agree exactly.

**PBO on a small matrix is itself very noisy.** Under the null, a 12×12 matrix returns PBO
anywhere from 0.08 to 0.92 depending on the draw, because every split reuses the same fixed
scores. It centres on 0.5 only across independent sweeps. The first test asserted ≈0.5 on a
single matrix and failed — correctly. `PBOReport.reliable` now reports whether the matrix is
large enough (≥10 configs and ≥10 blocks), and the summary says so, because a PBO figure
quoted without that caveat invites exactly the overconfidence the statistic exists to
prevent.

All three would have shipped as confident, wrong numbers. Statistical machinery has to be
tested against simulated ground truth in **both** directions — a correction that rejects
everything is as useless as one that accepts everything — and every gate here has a test for
each.

## Consequences

`dsio eval select` is the new command. It reads per-fold scores from the shared artifact
contract, which every runner already writes, so it works on tabular, torch, SSL and agent
runs without any of them knowing about it.

When not every run reports per-fold scores the transfer test is skipped and the verdict is
**provisional**, stated as such. Reporting a weaker check as if it were the full one is the
specific failure this avoids.

What is deliberately not built: no false-discovery-rate control across whole experiment
families, no sequential testing for a sweep that is still running, no automatic block-size
selection beyond the dependence-scaled default. Each is a real refinement; none is needed
before the basic correction exists.

`effective_sample_size` has an immediate application that is not yet wired: `sampling_noise`
still takes n at face value, so an evaluation over overlapping windows currently reports a
floor that is too optimistic. That is a known gap, not an oversight.
