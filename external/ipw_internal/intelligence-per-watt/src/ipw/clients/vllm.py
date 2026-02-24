"""Offline vLLM client backed by AsyncLLM."""

from __future__ import annotations

import asyncio
import atexit
import inspect
import json
import threading
import time
import uuid
from collections.abc import Iterable, Mapping
from typing import Any, Iterator, Sequence, Tuple

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

from ipw.core.registry import ClientRegistry
from ipw.core.types import ChatUsage, Response
from .base import InferenceClient

DEFAULT_WARMUP_COUNT = 10
DEFAULT_WARMUP_MAX_TOKENS = 8
_WARMUP_PROMPTS = (
    "This is a warmup prompt.",
    "Hello from the vLLM warmup.",
    "Intelligence Per Watt warmup query.",
)

_FALLBACK_ENGINE_KEYS = frozenset({
    "dtype", "quantization", "max_model_len", "gpu_memory_utilization",
    "tensor_parallel_size", "enforce_eager", "trust_remote_code",
    "max_num_seqs", "enable_prefix_caching",
})


def _resolve_engine_keys() -> frozenset[str]:
    """Resolve supported engine kwargs from AsyncEngineArgs when possible."""
    try:
        signature = inspect.signature(AsyncEngineArgs.__init__)
    except (TypeError, ValueError):
        return _FALLBACK_ENGINE_KEYS

    valid_kinds = {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }
    resolved = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self" and parameter.kind in valid_kinds
    }

    if not resolved:
        return _FALLBACK_ENGINE_KEYS

    # Mock/stub signatures often expose only varargs; ignore those.
    if resolved.issubset({"args", "kwargs", "kw"}):
        return _FALLBACK_ENGINE_KEYS

    return frozenset(resolved | _FALLBACK_ENGINE_KEYS)


_ENGINE_KEYS = _resolve_engine_keys()


def _coerce_value(value: Any) -> Any:
    """Convert CLI string values to native Python types."""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return value


class _AsyncLoopRunner:
    """Run an asyncio event loop in a background thread."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="ipw-vllm", daemon=True
        )
        self._thread.start()

    def run(self, coro) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def shutdown(self) -> None:
        if not self._loop.is_closed():

            async def _drain():
                current = asyncio.current_task()
                tasks = [
                    task
                    for task in asyncio.all_tasks()
                    if task is not current and not task.done()
                ]
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            try:
                future = asyncio.run_coroutine_threadsafe(_drain(), self._loop)
                future.result(timeout=5.0)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2.0)
            self._loop.close()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()


@ClientRegistry.register("vllm")
class VLLMClient(InferenceClient):
    """Offline AsyncLLM client."""

    client_id = "vllm"
    client_name = "vLLM Offline"
    DEFAULT_BASE_URL = "offline://vllm"

    def __init__(self, base_url: str | None = None, **config: Any) -> None:
        super().__init__(base_url or self.DEFAULT_BASE_URL, **config)
        self._engine_kwargs: dict[str, Any] = {
            key: _coerce_value(config[key])
            for key in _ENGINE_KEYS
            if key in config
        }
        self._sampling_defaults: dict[str, Any] = {
            "max_tokens": 4096,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
        }
        self._warmup_count = DEFAULT_WARMUP_COUNT
        self._warmup_max_tokens = DEFAULT_WARMUP_MAX_TOKENS
        self._warmup_done = False
        self._engine = None
        self._engine_args = None
        self._model_name = None
        self._loop_runner: _AsyncLoopRunner | None = _AsyncLoopRunner()
        self._closed = False
        atexit.register(self.close)

    def prepare(self, model: str) -> None:
        if self._closed:
            raise RuntimeError("vLLM client has been closed")
        self._ensure_engine(model)
        self._warmup_if_needed()

    def run_concurrent(
        self,
        model: str,
        prompt_iter: Iterable[Tuple[int, str]],
        max_in_flight: int,
        **params: Any,
    ) -> Iterator[Tuple[int, Response]]:
        if self._closed:
            raise RuntimeError("vLLM client has been closed")
        self._ensure_engine(model)
        self._warmup_if_needed()

        runner = self._loop_runner
        if runner is None:
            raise RuntimeError("vLLM client is shut down")

        sampling_params = self._build_sampling_params(params)

        async def _run_all():
            results: list[Tuple[int, Response]] = []
            pending: dict[str, Tuple[int, float]] = {}
            iterator = iter(prompt_iter)
            exhausted = False

            while True:
                # Fill up to max_in_flight
                while not exhausted and len(pending) < max_in_flight:
                    try:
                        index, prompt = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    request_id = f"ipw-{index}-{uuid.uuid4()}"
                    wall_start = time.time()
                    pending[request_id] = (index, wall_start)
                    # Start generation (non-blocking)
                    asyncio.create_task(
                        self._run_single_request(
                            request_id, prompt, sampling_params, pending, results
                        )
                    )

                if not pending:
                    break

                # Wait for at least one to complete
                await asyncio.sleep(0.001)

            return results

        all_results = runner.run(_run_all())
        for item in all_results:
            yield item

    async def _run_single_request(
        self,
        request_id: str,
        prompt: str,
        sampling_params: Any,
        pending: dict[str, Tuple[int, float]],
        results: list[Tuple[int, Response]],
    ) -> None:
        index, wall_start = pending[request_id]
        try:
            response = await self._stream_response(
                prompt=prompt, request_id=request_id, sampling_params=sampling_params
            )
            wall_end = time.time()
            response.request_start_time = wall_start
            response.request_end_time = wall_end
            results.append((index, response))
        finally:
            pending.pop(request_id, None)

    def stream_chat_completion(
        self, model: str, prompt: str, **params: Any
    ) -> Response:
        if self._closed:
            raise RuntimeError("vLLM client has been closed")
        self._ensure_engine(model)
        self._warmup_if_needed()

        sampling_params = self._build_sampling_params(params)
        request_id = str(params.get("request_id", uuid.uuid4()))
        runner = self._loop_runner
        if runner is None:
            raise RuntimeError("vLLM client is shut down")
        return runner.run(
            self._stream_response(
                prompt=prompt, request_id=request_id, sampling_params=sampling_params
            )
        )

    def list_models(self) -> Sequence[str]:
        return [self._model_name] if self._model_name else []

    def health(self) -> bool:
        return not self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._engine is not None:
                self._engine.shutdown()
        except Exception:  # pragma: no cover - shutdown best-effort
            pass
        finally:
            self._engine = None
            if self._loop_runner is not None:
                self._loop_runner.shutdown()
                self._loop_runner = None

    def _ensure_engine(self, model: str) -> None:
        if not model:
            raise ValueError("model name is required")
        if self._engine is not None:
            if model != self._model_name:
                raise RuntimeError(
                    f"vLLM client already loaded model '{self._model_name}', cannot switch to '{model}'"
                )
            return

        kwargs = dict(self._engine_kwargs)
        kwargs["model"] = model
        try:
            self._engine_args = AsyncEngineArgs(**kwargs)
            self._engine = AsyncLLM.from_engine_args(self._engine_args)
        except Exception as exc:  # pragma: no cover - forwarded to caller
            raise RuntimeError(f"Failed to initialize vLLM engine: {exc}") from exc
        self._model_name = model

    def _warmup_if_needed(self) -> None:
        if self._warmup_done or self._warmup_count <= 0:
            return
        runner = self._loop_runner
        if runner is None:
            raise RuntimeError("vLLM client is shut down")

        prompts = _WARMUP_PROMPTS or ("Warmup prompt",)
        sampling = SamplingParams(
            max_tokens=self._warmup_max_tokens,
            temperature=0.0,
            top_p=1.0,
            output_kind=RequestOutputKind.DELTA,
        )

        for idx in range(self._warmup_count):
            prompt = prompts[idx % len(prompts)]
            request_id = f"warmup-{idx}-{uuid.uuid4()}"
            runner.run(
                self._stream_response(
                    prompt=prompt, request_id=request_id, sampling_params=sampling
                )
            )

        self._warmup_done = True

    def _build_sampling_params(self, params: Mapping[str, Any]):
        recognized = {
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "max_tokens",
            "stop",
            "seed",
            "best_of",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "length_penalty",
        }

        overrides: dict[str, Any] = {}
        for key, value in params.items():
            if key.startswith("sampling_"):
                overrides[key.split("_", 1)[1]] = _coerce_value(value)
            elif key in recognized:
                overrides[key] = _coerce_value(value)

        sampling = {**self._sampling_defaults, **overrides}
        if "stop" in sampling:
            stop_value = sampling["stop"]
            if isinstance(stop_value, str):
                sampling["stop"] = [stop_value]
            elif isinstance(stop_value, (list, tuple)):
                sampling["stop"] = list(stop_value)
        sampling["output_kind"] = RequestOutputKind.DELTA
        return SamplingParams(**sampling)

    async def _stream_response(
        self, *, prompt: str, request_id: str, sampling_params: Any
    ) -> Response:
        if self._engine is None:
            raise RuntimeError("vLLM engine is not initialized")

        start_time = time.perf_counter()
        prompt_tokens: int | None = None
        completion_tokens = 0
        ttft_ms: float | None = None
        first_token_time: float | None = None
        content_parts: list[str] = []

        try:
            async for chunk in self._engine.generate(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
            ):
                outputs = getattr(chunk, "outputs", []) or []
                if prompt_tokens is None:
                    prompt_ids = getattr(chunk, "prompt_token_ids", None) or []
                    prompt_tokens = len(prompt_ids)

                stop_requested = False

                for completion in outputs:
                    delta_text = getattr(completion, "text", "") or ""
                    if delta_text:
                        content_parts.append(delta_text)
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - start_time) * 1000.0
                            first_token_time = time.time()

                    delta_token_ids = getattr(completion, "delta_token_ids", None)
                    if delta_token_ids is None:
                        delta_token_ids = getattr(completion, "token_ids_delta", None)
                    if delta_token_ids is not None:
                        completion_tokens += len(delta_token_ids)
                    else:
                        token_ids = getattr(completion, "token_ids", None)
                        if token_ids:
                            completion_tokens += len(token_ids)
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - start_time) * 1000.0
                                first_token_time = time.time()

                    finished_reason = getattr(completion, "finished_reason", None)
                    if finished_reason is not None:
                        if str(finished_reason).lower() in {
                            "stop",
                            "stopped",
                            "eos",
                            "eos_token",
                        }:
                            stop_requested = True

                if stop_requested:
                    break

                if getattr(chunk, "finished", False):
                    break
        except (
            Exception
        ) as exc:  # pragma: no cover - actual streaming exercised in integration
            raise RuntimeError(f"vLLM offline generation failed: {exc}") from exc

        usage = ChatUsage(
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens,
            total_tokens=(prompt_tokens or 0) + completion_tokens,
        )
        content = "".join(content_parts)
        return Response(
            content=content,
            usage=usage,
            time_to_first_token_ms=ttft_ms or 0.0,
            first_token_time=first_token_time,
        )
