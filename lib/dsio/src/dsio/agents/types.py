"""What an agent run produces: requests, responses, and the trajectory between them.

The artifact of an agent experiment is the **trajectory**, not the score. A success rate
says a run solved 62% of tasks; it cannot say whether the failures were bad plans, bad tool
calls, or a tool that was down — and those have completely different fixes. Trajectories are
kept for the same reason out-of-fold predictions are kept: the metric is a lossy summary
chosen before you knew what you would need to ask.

Everything here is plain data with a canonical serialisation, because the request is what
gets hashed for the response cache and a hash that depends on dict ordering is a cache that
never hits.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from dsio.contracts import DsioModel, sha256_of

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(DsioModel):
    """One tool invocation and what came back."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None
    error: str | None = None
    seconds: float = 0.0

    @property
    def failed(self) -> bool:
        return self.error is not None


class Message(DsioModel):
    """One turn. ``tool_calls`` is empty for everything but an assistant turn that used one."""

    role: Role
    content: str = ""
    name: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


class Usage(DsioModel):
    """What a request cost, in the three currencies that matter.

    Cost is recorded per call rather than derived at the end from a price table, because
    prices change and a run's record has to keep meaning what it meant when it was made.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            seconds=self.seconds + other.seconds,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Request(DsioModel):
    """Everything that determines a response, and therefore everything in its cache key.

    ``temperature`` and ``seed`` are part of the identity rather than metadata. A response
    generated at temperature 1.0 is not a substitute for one generated at 0.0, and treating
    them as interchangeable is how a cache turns a stochastic system into an apparently
    deterministic one — which silently destroys the variance an agent evaluation exists to
    measure.
    """

    model: str
    messages: tuple[Message, ...]
    tools: tuple[str, ...] = ()
    temperature: float = 0.0
    max_tokens: int | None = None
    stop: tuple[str, ...] = ()
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def digest(self) -> str:
        return sha256_of(self.model_dump(mode="json"))


class Response(DsioModel):
    """One model reply, with what it cost and where it came from."""

    message: Message
    usage: Usage = Usage()
    finish_reason: str = "stop"
    provider: str = ""
    cached: bool = False


class Step(DsioModel):
    """One iteration of the agent loop: a request, a reply, and any tools it ran."""

    index: int
    request: Request
    response: Response

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return self.response.message.tool_calls


class Trajectory(DsioModel):
    """The full record of one attempt at one task.

    ``repeat`` distinguishes attempts at the *same* task by the same configuration. It is a
    first-class field rather than a tag because repeats are how the model's own sampling
    noise gets measured, and a trajectory that cannot say which attempt it was is useless
    for that.
    """

    example: str
    repeat: int = 0
    steps: tuple[Step, ...] = ()
    outcome: str | None = None
    expected: str | None = None
    success: bool | None = None
    error: str | None = None

    @property
    def usage(self) -> Usage:
        total = Usage()
        for step in self.steps:
            total = total + step.response.usage
        return total

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [call for step in self.steps for call in step.tool_calls]

    @property
    def failed_tool_calls(self) -> int:
        return sum(1 for call in self.tool_calls if call.failed)

    def summary(self) -> dict[str, Any]:
        """The row form, for a table of trajectories."""
        usage = self.usage
        return {
            "example": self.example,
            "repeat": self.repeat,
            "success": self.success,
            "steps": self.n_steps,
            "tool_calls": len(self.tool_calls),
            "failed_tool_calls": self.failed_tool_calls,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": usage.cost_usd,
            "seconds": usage.seconds,
            "error": self.error,
        }
