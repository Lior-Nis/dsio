"""Agent invariants: the cache trap, the second noise floor, and the trajectory artifact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dsio.agents import (
    CachedProvider,
    Message,
    Request,
    ResponseCache,
    ScriptedProvider,
    Trajectory,
    combined_floor,
    contains,
    exact_match,
    repeat_report,
    run_agent,
    stability_verdict,
)
from dsio.agents.loop import AgentError
from dsio.agents.repeats import RepeatError


def ask(text: str) -> Request:
    return Request(model="m", messages=(Message(role="user", content=text),))


# --- the cache trap -------------------------------------------------------------------


def test_the_cache_keys_on_the_repeat_index(tmp_path: Path) -> None:
    """The single most important property in this module.

    A cache keyed only on the request turns a stochastic system into an apparently
    deterministic one: five repeats at temperature 1.0 all return the same cached reply, the
    measured variance is zero, and the run reports a confidence it has not earned.
    """
    cache = ResponseCache(tmp_path)
    request = ask("hello")
    assert cache.key(request, repeat=0) != cache.key(request, repeat=1)


def test_repeats_are_cached_independently(tmp_path: Path) -> None:
    """Each draw is stored under its own key, so a cached rerun replays the whole
    distribution rather than one point from it."""
    inner = ScriptedProvider(default="answer", jitter=0.5)
    provider = CachedProvider(inner, ResponseCache(tmp_path))
    request = ask("q")

    first = [provider.complete(request, repeat=i).message.content for i in range(6)]
    calls_after_cold = inner.calls
    second = [provider.complete(request, repeat=i).message.content for i in range(6)]

    assert calls_after_cold == 6, "a cold cache must actually call the provider"
    assert inner.calls == 6, "a warm cache must call it zero more times"
    assert first == second, "the cached rerun must reproduce every draw"
    assert len(set(first)) > 1, "the fixture must actually vary, or this proves nothing"


def test_a_cached_response_is_marked_and_costs_nothing(tmp_path: Path) -> None:
    """A cached run did not spend the money, and a cost that includes spending which did
    not happen is the wrong number for both questions cost gets used for."""
    provider = CachedProvider(ScriptedProvider(default="x"), ResponseCache(tmp_path))
    request = ask("q")
    live = provider.complete(request)
    cached = provider.complete(request)

    assert live.cached is False and cached.cached is True
    assert cached.usage.cost_usd == 0.0
    assert cached.usage.output_tokens == 0
    assert cached.usage.cached_tokens == live.usage.total_tokens


def test_temperature_is_part_of_the_request_identity() -> None:
    """A reply generated at temperature 1.0 is not a substitute for one at 0.0."""
    assert ask("q").digest != ask("q").model_copy(update={"temperature": 1.0}).digest


def test_cache_reports_its_hit_rate(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    provider = CachedProvider(ScriptedProvider(default="x"), cache)
    for _ in range(4):
        provider.complete(ask("same"))
    assert cache.stats["hits"] == 3
    assert cache.stats["hit_rate"] == pytest.approx(0.75)


# --- the second noise floor -----------------------------------------------------------


def test_run_std_is_the_spread_across_complete_passes() -> None:
    """Literally "if I ran this whole evaluation again, how much would the headline move"."""
    outcomes = {
        "a": [1.0, 1.0, 1.0],
        "b": [1.0, 0.0, 1.0],
        "c": [0.0, 0.0, 1.0],
        "d": [1.0, 1.0, 0.0],
    }
    report = repeat_report(outcomes)
    assert report.n_examples == 4 and report.n_repeats == 3
    assert report.run_values == pytest.approx((0.75, 0.5, 0.75))
    assert report.run_std == pytest.approx(float(np.std([0.75, 0.5, 0.75], ddof=1)))


def test_a_single_repeat_reports_no_spread_and_says_so() -> None:
    """Reporting 0.0 as if it were measured is the mistake; `usable` is how it is flagged."""
    report = repeat_report({"a": [1.0], "b": [0.0]})
    assert report.usable is False
    assert report.run_std == 0.0
    assert "not measured" in " ".join(report.summary_lines())


def test_agreement_localises_the_instability() -> None:
    """A low spread with low agreement means individual tasks flip but cancel out, which is
    a different problem from a uniformly noisy system."""
    report = repeat_report({"steady": [1.0, 1.0], "flips": [1.0, 0.0], "other": [0.0, 1.0]})
    assert report.agreement == pytest.approx(1 / 3)
    assert set(report.unstable_examples) == {"flips", "other"}
    assert report.run_std == 0.0, "the flips cancel, so the pass-level spread is zero"


def test_ragged_repeats_are_rejected_not_padded() -> None:
    """A task that failed to produce a repeat is missing evidence; filling the hole with a
    zero would report a failure the model never had."""
    with pytest.raises(RepeatError, match="missing evidence"):
        repeat_report({"a": [1.0, 1.0], "b": [1.0]})


def test_floors_combine_in_quadrature() -> None:
    """Fold spread and sampling spread are independent, so the bar is the root of the sum
    of squares — using either alone understates it."""
    assert combined_floor(0.03, 0.04) == pytest.approx(0.05)
    assert combined_floor(0.0, 0.04) == pytest.approx(0.04)


def test_an_improvement_inside_the_sampling_noise_is_neutral() -> None:
    """The characteristic error of agent benchmarking, refused.

    62% to 65% looks like a win, and on a hundred tasks at temperature 1.0 the run-to-run
    spread is frequently wider than that.
    """
    candidate = repeat_report({str(i): [1.0, 0.0, 1.0, 1.0] for i in range(20)})
    baseline = repeat_report({str(i): [1.0, 0.0, 0.0, 1.0] for i in range(20)})
    verdict = stability_verdict(candidate, baseline)
    assert verdict["measured"] is True
    assert verdict["outcome"] == "neutral"


def test_a_consistent_improvement_clears_the_floor() -> None:
    candidate = repeat_report({str(i): [1.0, 1.0] for i in range(20)})
    baseline = repeat_report({str(i): [0.0, 0.0] for i in range(20)})
    assert stability_verdict(candidate, baseline)["outcome"] == "win"


def test_a_single_repeat_makes_the_verdict_unknown() -> None:
    """Not neutral, and certainly not a win: the spread was never measured."""
    verdict = stability_verdict(
        repeat_report({"a": [1.0]}), repeat_report({"a": [0.0]})
    )
    assert verdict["outcome"] == "unknown"
    assert verdict["measured"] is False
    assert "never" in str(verdict["reason"])


# --- the loop -------------------------------------------------------------------------


def test_a_simple_task_runs_and_records_its_trajectory() -> None:
    provider = ScriptedProvider({"2+2?": "4"})
    trajectory = run_agent(provider, example="t1", prompt="2+2?", expected="4")
    assert trajectory.outcome == "4"
    assert exact_match(trajectory) is True
    assert trajectory.n_steps == 1
    assert trajectory.error is None


def test_a_tool_error_is_recorded_and_fed_back_not_raised() -> None:
    """An agent that never sees a tool error is being evaluated on a world it will not meet."""

    def broken() -> str:
        raise RuntimeError("service is down")

    provider = ScriptedProvider(
        ["", "recovered"],
        tool_calls={"go": [{"name": "fetch", "arguments": {}}]},
    )
    trajectory = run_agent(
        provider, example="t", prompt="go", tools={"fetch": broken}, max_steps=3
    )
    assert trajectory.failed_tool_calls == 1
    assert "service is down" in (trajectory.tool_calls[0].error or "")
    assert trajectory.outcome == "recovered"


def test_an_unknown_tool_is_an_observation_not_a_crash() -> None:
    provider = ScriptedProvider(
        ["", "done"], tool_calls={"go": [{"name": "nope", "arguments": {}}]}
    )
    trajectory = run_agent(provider, example="t", prompt="go", tools={}, max_steps=3)
    assert "no such tool" in (trajectory.tool_calls[0].error or "")


def test_exhausting_the_step_budget_is_an_outcome() -> None:
    """A recorded outcome rather than an exception: a task that ran out of steps is a
    result, and losing the whole run over it would lose the evidence too."""
    provider = ScriptedProvider(
        [""] * 5, tool_calls={"*": [{"name": "loop", "arguments": {}}]}
    )
    trajectory = run_agent(
        provider, example="t", prompt="go", tools={"loop": lambda: "again"}, max_steps=2
    )
    assert trajectory.error is not None and "budget" in trajectory.error
    assert trajectory.n_steps == 2


def test_a_provider_failure_is_recorded_on_the_trajectory() -> None:
    class Broken:
        name = "broken"

        def complete(self, request, *, repeat: int = 0):  # type: ignore[no-untyped-def]
            raise RuntimeError("rate limited")

    trajectory = run_agent(Broken(), example="t", prompt="q")
    assert trajectory.error is not None and "rate limited" in trajectory.error
    assert trajectory.outcome is None


def test_a_zero_step_budget_is_a_configuration_error() -> None:
    """The loop raises only when it is configured wrongly; task failures are outcomes."""
    with pytest.raises(AgentError, match="at least 1"):
        run_agent(ScriptedProvider(), example="t", prompt="q", max_steps=0)


def test_usage_accumulates_across_steps() -> None:
    provider = ScriptedProvider(
        ["", "done"], tool_calls={"go": [{"name": "t", "arguments": {}}]}
    )
    trajectory = run_agent(
        provider, example="t", prompt="go", tools={"t": lambda: "ok"}, max_steps=3
    )
    assert trajectory.n_steps == 2
    assert trajectory.usage.output_tokens > 0


def test_graders_are_replaceable() -> None:
    trajectory = Trajectory(example="t", outcome="The answer is 4.", expected="4")
    assert exact_match(trajectory) is False
    assert contains(trajectory) is True


def test_a_trajectory_summarises_to_a_row() -> None:
    """Trajectories are the artifact; the row form is what makes a thousand of them
    readable without parsing every step."""
    provider = ScriptedProvider({"q": "a"})
    trajectory = run_agent(provider, example="t", prompt="q", expected="a")
    row = trajectory.summary()
    assert row["example"] == "t" and row["steps"] == 1
    assert set(row) >= {"cost_usd", "tool_calls", "failed_tool_calls", "output_tokens"}
    json.dumps(row)  # must be serialisable as-is
