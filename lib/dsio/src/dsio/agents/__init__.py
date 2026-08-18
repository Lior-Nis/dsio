"""Agentic experimentation: trajectories as the artifact, sampling noise as a floor.

The reproducibility problem here is different in kind from a supervised one. The same
configuration on the same tasks returns a different number every time, so a result is a
distribution rather than a point — and the machinery has to measure the spread rather than
pretend it away. Two things follow, and both are structural:

- **repeats are first-class**, and the response cache keys on the repeat index, so caching
  cannot silently collapse a distribution to one draw;
- **the noise floor has two independent parts**, fold spread and run-to-run spread, and a
  claimed improvement is judged against both.

dsio ships no API clients. :class:`~dsio.agents.provider.Provider` is three lines and a
project brings its own, which keeps vendor SDKs, retries and rate limits out of the spine.
"""

from dsio.agents.loop import AgentError, Tool, contains, exact_match, run_agent
from dsio.agents.provider import (
    CachedProvider,
    Provider,
    ProviderError,
    ResponseCache,
    ScriptedProvider,
    cache_root,
)
from dsio.agents.repeats import (
    RepeatError,
    RepeatReport,
    combined_floor,
    repeat_report,
    stability_verdict,
)
from dsio.agents.types import Message, Request, Response, Step, ToolCall, Trajectory, Usage

__all__ = [
    "AgentError",
    "CachedProvider",
    "Message",
    "Provider",
    "ProviderError",
    "RepeatError",
    "RepeatReport",
    "Request",
    "Response",
    "ResponseCache",
    "ScriptedProvider",
    "Step",
    "Tool",
    "ToolCall",
    "Trajectory",
    "Usage",
    "cache_root",
    "combined_floor",
    "contains",
    "exact_match",
    "repeat_report",
    "run_agent",
    "stability_verdict",
]
