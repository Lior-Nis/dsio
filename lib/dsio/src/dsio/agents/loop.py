"""The agent loop: request, tools, repeat, until done.

Deliberately small and unopinionated. dsio's job is to make an agent experiment
*reproducible and comparable*, not to be a framework for writing agents — the loop exists
so there is something to instrument, and a project that has its own loop can skip it
entirely and hand dsio the trajectories.

The parts that are not negotiable are the ones that make a run comparable to another run:
every request is recorded, every tool call is recorded with its error, the step budget is
explicit, and hitting it is a recorded outcome rather than an exception.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from dsio.agents.provider import Provider
from dsio.agents.types import Message, Request, Step, ToolCall, Trajectory

Tool = Callable[..., str]


class AgentError(RuntimeError):
    """Raised when the loop is misconfigured. Task failures are outcomes, not exceptions."""


def run_agent(
    provider: Provider,
    *,
    example: str,
    prompt: str,
    system: str | None = None,
    tools: Mapping[str, Tool] | None = None,
    model: str = "unspecified",
    temperature: float = 0.0,
    max_steps: int = 8,
    repeat: int = 0,
    expected: str | None = None,
    stop_when: Callable[[str], bool] | None = None,
) -> Trajectory:
    """Run one task to completion, a step budget, or an error, recording everything.

    A tool that raises is recorded on the :class:`~dsio.agents.types.ToolCall` and fed back
    to the model as an observation, because that is what happens in production and an agent
    that never sees a tool error is being evaluated on a world it will not meet. The loop
    itself only raises when it is *configured* wrongly.
    """
    if max_steps < 1:
        raise AgentError(f"max_steps must be at least 1, got {max_steps}")
    tools = dict(tools or {})

    history: list[Message] = []
    if system:
        history.append(Message(role="system", content=system))
    history.append(Message(role="user", content=prompt))

    steps: list[Step] = []
    outcome: str | None = None
    error: str | None = None

    for index in range(max_steps):
        request = Request(
            model=model,
            messages=tuple(history),
            tools=tuple(sorted(tools)),
            temperature=temperature,
        )
        try:
            response = provider.complete(request, repeat=repeat)
        except Exception as failure:  # noqa: BLE001 - recorded as the trajectory's outcome
            error = f"{type(failure).__name__}: {failure}"
            break

        executed = _run_tools(response.message.tool_calls, tools)
        response = response.model_copy(
            update={"message": response.message.model_copy(update={"tool_calls": executed})}
        )
        steps.append(Step(index=index, request=request, response=response))
        history.append(response.message)

        if executed:
            for call in executed:
                history.append(
                    Message(
                        role="tool",
                        name=call.name,
                        content=call.error or call.result or "",
                    )
                )
            continue

        outcome = response.message.content
        if stop_when is None or stop_when(outcome):
            break
        # The model answered but the caller does not consider it finished; keep going until
        # the budget runs out, and record that if it does.
        history.append(Message(role="user", content="continue"))
    else:
        error = error or f"step budget of {max_steps} exhausted without a final answer"

    return Trajectory(
        example=example,
        repeat=repeat,
        steps=tuple(steps),
        outcome=outcome,
        expected=expected,
        error=error,
    )


def _run_tools(calls: tuple[ToolCall, ...], tools: Mapping[str, Tool]) -> tuple[ToolCall, ...]:
    """Execute each requested tool, recording failures instead of raising them."""
    executed: list[ToolCall] = []
    for call in calls:
        started = time.perf_counter()
        if call.name not in tools:
            executed.append(
                call.model_copy(
                    update={
                        "error": (
                            f"no such tool {call.name!r}; available: "
                            f"{', '.join(sorted(tools)) or 'none'}"
                        ),
                        "seconds": time.perf_counter() - started,
                    }
                )
            )
            continue
        try:
            result = tools[call.name](**call.arguments)
            update: dict[str, Any] = {"result": str(result), "error": None}
        except Exception as failure:  # noqa: BLE001 - the model must see the error
            update = {"result": None, "error": f"{type(failure).__name__}: {failure}"}
        update["seconds"] = time.perf_counter() - started
        executed.append(call.model_copy(update=update))
    return tuple(executed)


def exact_match(trajectory: Trajectory) -> bool:
    """The default grader: the final answer equals what was expected, trimmed.

    Deliberately strict and deliberately replaceable. Most real tasks need a grader that
    knows something about the domain, and pretending a string comparison is general is how
    an agent benchmark ends up measuring formatting.
    """
    if trajectory.outcome is None or trajectory.expected is None:
        return False
    return trajectory.outcome.strip() == trajectory.expected.strip()


def contains(trajectory: Trajectory) -> bool:
    """Graded on whether the expected answer appears in the response."""
    if trajectory.outcome is None or trajectory.expected is None:
        return False
    return trajectory.expected.strip().lower() in trajectory.outcome.strip().lower()
