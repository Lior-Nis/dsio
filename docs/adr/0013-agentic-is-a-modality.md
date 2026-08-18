# 13. Agentic is a modality, and its noise floor has two parts

Status: accepted (2026-08-19)
Supersedes the plan's "NLP scope" open question.

## Context

dsio's scope is machine, deep **and agentic** learning experimentation. The first two were
built; the third was missing entirely — no prompt-as-config, no response caching, no
trajectory artifact, no agent-task evaluation.

It is also the modality where the reproducibility problem is currently worst, and worst in a
way the rest of dsio's machinery does not address. Everywhere else, running the same config
on the same data twice gives the same number. Here it does not.

## Decisions

**An agent evaluation is a cross-validated experiment like any other.** It goes through the
same fold loop, writes the same artifact contract, and lands in the same ledger, so
`dsio eval verdict` compares a prompt change exactly the way it compares a learning rate.
Inventing a parallel evaluation path for agents is how agent results end up incomparable to
everything else in the same project.

**Tasks are `KeyedExamples`, so grouping works as it does everywhere.** This matters more
than it looks: benchmark suites routinely contain families of near-identical tasks generated
from one template, and splitting those across train and test is the same leak as splitting
one subject's recordings across folds. The generic `Examples` protocol from ADR 0012 made
this free.

**The artifact is the trajectory, not the score.** A success rate says a run solved 62% of
tasks. It cannot say whether the failures were bad plans, bad tool calls, or a tool that was
down — and those have completely different fixes. Same reasoning as keeping out-of-fold
predictions: the metric is a lossy summary chosen before you knew what you would need to ask.

**dsio ships no API clients.** `Provider` is a three-line protocol and a project brings its
own, which keeps vendor SDKs, retry policies and rate limits out of the spine. `ScriptedProvider`
is not a mock — it satisfies the same protocol and goes through the same loop, so a test
using it exercises the real machinery and removes only the network.

**Tool failures are observations, not exceptions.** A tool that raises is recorded on the
call and fed back to the model, because that is what happens in production and an agent
never shown a tool error is being evaluated on a world it will not meet. Exhausting the step
budget is likewise a recorded outcome. The loop raises only when it is *configured* wrongly.

## The response cache, and the trap in it

Caching model responses is obviously worth doing — the calls are the expensive part. But a
cache keyed only on the request **turns a stochastic system into an apparently deterministic
one**. Ask for five repeats at temperature 1.0, get the same cached reply five times,
measure zero variance, and report a confidence the run has not earned. That is worse than no
cache, because the number looks better.

So the key includes the **repeat index**. Repeat 0 and repeat 3 of the same request are
different entries, which makes a cached evaluation reproduce the whole distribution rather
than one draw from it. At temperature 0 the repeats coincide and nothing is lost.

A cache hit reports **zero cost**, not the replayed cost of the original call. A cached run
did not spend the money, and a cost that includes spending which did not happen is the wrong
number for both questions cost gets used for: what this experiment cost, and what running it
at scale would cost.

## The second noise floor

Everywhere else in dsio, uncertainty comes from which examples landed in which fold. Here
there is a second, independent source, and it is usually the larger one.

Ignoring it is the characteristic error of agent benchmarking. A prompt change that moves
success from 62% to 65% looks like a win; on a hundred tasks at temperature 1.0 the
run-to-run spread is frequently wider than that. Reporting that delta without the spread is
the same mistake as reporting a fold delta without the fold spread, which dsio already
refuses to make.

The estimator is the simplest one that answers the question people actually ask: take repeat
*j* of every task — that is one complete evaluation pass — and take the standard deviation
across the *k* passes. The result is literally "if I ran this whole evaluation again, how
much would the headline move?"

`combined_floor` adds fold spread and run spread in quadrature, because they are independent.
Using only the fold spread — which is what happens when an agent evaluation is dropped into
ordinary cross-validation machinery — understates the bar, usually by a lot.

Three consequences, all enforced:

- a configuration with `temperature > 0` and `repeats < 1` is **rejected at config time**: a
  point estimate of a distribution with no idea how far it moves;
- a single repeat reports `usable=False` and a verdict of **unknown**, not neutral and
  certainly not a win — the spread was never measured;
- ragged repeats are rejected rather than padded, because a task that failed to produce a
  repeat is missing evidence, and a zero would report a failure the model never had.

`agreement` is reported alongside the spread because it localises the instability. A low
spread with low agreement means individual tasks are flipping and cancelling out, which is a
different problem from a uniformly noisy system.

**Resources are reported beside the score.** Cost, tokens, latency, mean steps, tool error
rate, budget exhaustion. A configuration two points better at four times the price is not
obviously better, and a report that omits the price cannot say so.

## What building this changed elsewhere

`Fold` required a non-empty train part, which is a supervised assumption. Evaluating
something that was not trained here — a benchmark pass, a shipped model, an agent behind an
API — is a real shape, and my first attempt faked it with a one-element training set that
both lied and violated the disjointness `Fold` exists to enforce. `evaluation_only=True` now
makes it expressible, and the default still refuses an accidental empty train part with a
message that says which one you meant.

## Consequences

`dsio.agents` depends on `dsio.eval`, `dsio.data` and `dsio.config`, and on no third-party
package at all — no torch, no HTTP client. It is the cheapest modality in the repo to
install and the only one with no optional extra.

What is deliberately not built: no planner or memory abstractions, no multi-agent
orchestration, no prompt-optimisation loop. The loop here exists so there is something to
instrument; a project with its own agent framework can skip it and hand dsio trajectories
directly, which is the seam that matters.
