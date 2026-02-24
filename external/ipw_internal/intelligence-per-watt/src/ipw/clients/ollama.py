from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Iterable, Iterator, Mapping, Tuple

from ollama import Client, ResponseError

from ipw.core.registry import ClientRegistry
from ipw.core.types import ChatUsage, Response
from .base import InferenceClient


def _normalize_base_url(base_url: str) -> str:
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"
    return base_url.rstrip("/")


@ClientRegistry.register("ollama")
class OllamaClient(InferenceClient):
    client_id, client_name = "ollama", "Ollama"
    DEFAULT_BASE_URL = "http://127.0.0.1:11434"

    def __init__(self, base_url: str | None = None, **config: Any) -> None:
        host = _normalize_base_url(base_url or self.DEFAULT_BASE_URL)
        super().__init__(host, **config)
        self._client = Client(host=host)

    def run_concurrent(
        self,
        model: str,
        prompt_iter: Iterable[Tuple[int, str]],
        max_in_flight: int,
        **params: Any,
    ) -> Iterator[Tuple[int, Response]]:
        iterator = iter(prompt_iter)
        max_workers = max(1, int(max_in_flight))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            in_flight: dict[Any, int] = {}
            exhausted = False
            while True:
                while not exhausted and len(in_flight) < max_workers:
                    try:
                        index, prompt = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    future = executor.submit(
                        self._run_single_prompt, model, prompt, params
                    )
                    in_flight[future] = index

                if not in_flight:
                    break

                done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    index = in_flight.pop(future)
                    response = future.result()
                    yield index, response

    def list_models(self) -> list[str]:
        try:
            response = self._client.list()
        except ResponseError as exc:
            raise RuntimeError(f"Ollama error: {exc}") from exc
        return [
            str(model.model)
            for model in response.models
            if getattr(model, "model", None)
        ]

    def health(self) -> bool:
        try:
            self._client.list()
            return True
        except Exception:
            return False

    def _build_payload(
        self, model: str, prompt: str, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        payload = dict(params)
        payload["model"] = model
        payload["prompt"] = prompt
        payload["stream"] = True
        return payload

    def _run_single_prompt(
        self,
        model: str,
        prompt: str,
        params: Mapping[str, Any],
    ) -> Response:
        payload = self._build_payload(model, prompt, params)
        wall_start = time.time()
        start = time.perf_counter()
        try:
            stream = self._client.generate(**payload)
        except ResponseError as exc:
            raise RuntimeError(f"Ollama error: {exc}") from exc

        content: list[str] = []
        prompt_tokens = completion_tokens = 0
        ttft_ms: float | None = None

        for chunk in stream:
            text = getattr(chunk, "response", None)
            if text:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start) * 1000
                content.append(text)
            if getattr(chunk, "done", False):
                prompt_tokens = int(chunk.prompt_eval_count or prompt_tokens)
                completion_tokens = int(chunk.eval_count or completion_tokens)

        wall_end = time.time()
        return Response(
            content="".join(content),
            usage=ChatUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            time_to_first_token_ms=ttft_ms or 0.0,
            request_start_time=wall_start,
            request_end_time=wall_end,
        )
