"""Providers, and the response cache — including the trap it must not fall into.

A provider is anything that turns a :class:`~dsio.agents.types.Request` into a
:class:`~dsio.agents.types.Response`. dsio ships no API clients: the protocol is three lines
and a project brings its own, which keeps vendor SDKs, retry policies and rate limits out of
the spine entirely.

**The trap.** Caching model responses is obviously worth doing — the calls are the expensive
part, and re-running an evaluation to change one metric should not re-buy every completion.
But a cache keyed only on the request turns a *stochastic* system into an apparently
deterministic one. Ask for five repeats at temperature 1.0 and all five get the same cached
reply, the measured variance is zero, and the run reports a confidence it has not earned.
That is worse than no cache, because the number looks better.

So the cache key includes the **repeat index**. Repeat 0 and repeat 3 of the same request
are different cache entries, which is what makes a cached evaluation reproduce the *whole*
distribution rather than one draw from it. At temperature 0 the repeats coincide anyway and
nothing is lost.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from dsio.agents.types import Message, Request, Response, Usage
from dsio.contracts import atomic_write, sha256_of

CACHE_DIRNAME = "responses"


class ProviderError(RuntimeError):
    """Raised when a provider cannot answer a request."""


@runtime_checkable
class Provider(Protocol):
    """Anything that answers a request. dsio ships no API clients on purpose."""

    @property
    def name(self) -> str: ...

    def complete(self, request: Request, *, repeat: int = 0) -> Response:
        """Answer ``request``. ``repeat`` distinguishes independent draws of the same one."""
        ...


class ResponseCache:
    """Content-addressed store of model replies, keyed by request *and* repeat.

    One JSON file per entry, named by digest. Deliberately not the stage cache: that keys on
    code and config to decide whether a whole stage must re-run, while this keys on a single
    request and is written thousands of times per run.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.hits = 0
        self.misses = 0

    def key(self, request: Request, repeat: int = 0) -> str:
        """Digest of the request together with which draw of it this is.

        Including ``repeat`` is what stops the cache from collapsing a distribution to a
        point. Excluding it would make every repeat identical and every variance estimate
        zero — a silent, confident lie about how stable the result is.
        """
        return sha256_of({"request": request.model_dump(mode="json"), "repeat": repeat})

    def path_for(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, request: Request, repeat: int = 0) -> Response | None:
        path = self.path_for(self.key(request, repeat))
        if not path.is_file():
            self.misses += 1
            return None
        self.hits += 1
        payload = json.loads(path.read_text(encoding="utf-8"))
        response = Response.model_validate(payload)
        return response.model_copy(update={"cached": True})

    def put(self, request: Request, response: Response, repeat: int = 0) -> None:
        path = self.path_for(self.key(request, repeat))
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(
            path,
            json.dumps(
                response.model_copy(update={"cached": False}).model_dump(mode="json"),
                indent=2,
                sort_keys=True,
            ).encode(),
        )

    @property
    def stats(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
        }


class CachedProvider:
    """Wrap a provider so identical (request, repeat) pairs are answered from disk.

    Costs are reported as **zero** for a cache hit rather than replayed from the original
    call. A cached run did not spend the money, and a cost figure that includes spending
    that did not happen is the wrong number for the only two questions cost gets used for:
    what this experiment cost, and what running it at scale would cost.
    """

    def __init__(self, inner: Provider, cache: ResponseCache) -> None:
        self.inner = inner
        self.cache = cache

    @property
    def name(self) -> str:
        return f"cached({self.inner.name})"

    def complete(self, request: Request, *, repeat: int = 0) -> Response:
        found = self.cache.get(request, repeat)
        if found is not None:
            return found.model_copy(update={"usage": Usage(cached_tokens=found.usage.total_tokens)})
        response = self.inner.complete(request, repeat=repeat)
        self.cache.put(request, response, repeat)
        return response


class ScriptedProvider:
    """A provider that replays a fixed script. For tests, and for offline reruns.

    Not a mock: it satisfies the same protocol and goes through the same loop, so a test
    using it exercises the real agent machinery. What it removes is the network, not the
    code path.
    """

    def __init__(
        self,
        replies: dict[str, str] | list[str] | None = None,
        *,
        name: str = "scripted",
        default: str = "",
        tool_calls: dict[str, list[dict[str, Any]]] | None = None,
        jitter: float = 0.0,
    ) -> None:
        self.replies = replies or {}
        self._name = name
        self.default = default
        self.tool_calls = tool_calls or {}
        self.jitter = jitter
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def complete(self, request: Request, *, repeat: int = 0) -> Response:
        self.calls += 1
        started = time.perf_counter()
        prompt = request.messages[-1].content if request.messages else ""

        if isinstance(self.replies, list):
            content = self.replies[(self.calls - 1) % len(self.replies)]
        else:
            content = self.replies.get(prompt, self.default)

        if self.jitter:
            # A deterministic stand-in for sampling noise: the same prompt answers
            # differently on different repeats, which is what makes the repeat-variance
            # machinery testable without a network.
            import hashlib

            digest = hashlib.sha256(f"{prompt}:{repeat}".encode()).digest()[0]
            if digest / 255.0 < self.jitter:
                content = f"{content}!"

        # "*" matches any prompt, which is how a scripted provider can keep calling tools
        # indefinitely — the only way to exercise the step budget.
        specs = self.tool_calls.get(prompt, self.tool_calls.get("*", []))
        calls = tuple(_tool_call(spec) for spec in specs)
        return Response(
            message=Message(role="assistant", content=content, tool_calls=calls),
            usage=Usage(
                input_tokens=sum(len(m.content) for m in request.messages) // 4,
                output_tokens=max(1, len(content) // 4),
                seconds=time.perf_counter() - started,
            ),
            provider=self._name,
        )


def _tool_call(spec: dict[str, Any]) -> Any:
    from dsio.agents.types import ToolCall

    return ToolCall(**spec)


def cache_root(root: Path | str | None = None) -> Path:
    return Path(root or "cache") / CACHE_DIRNAME
