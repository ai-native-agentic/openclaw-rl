from __future__ import annotations

import argparse
import asyncio
import logging
import os
from hashlib import sha1
from typing import Any

import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .request_utils import json_payload

logger = logging.getLogger("terminal.env.router")
app = FastAPI()


def _format_error(exc: BaseException) -> str:
    detail = str(exc).strip()
    if detail:
        return f"{type(exc).__name__}: {detail}"
    return type(exc).__name__


def _status_from_payload(payload: dict[str, Any], default: int) -> int:
    raw = payload.get("status_code")
    if isinstance(raw, int):
        return raw
    return default


class Router:
    def __init__(
        self,
        worker_urls: list[str],
        forward_timeout: float = 600.0,
        forward_retries: int = 1,
        forward_retry_backoff: float = 0.2,
    ):
        if not worker_urls:
            raise ValueError("At least one worker URL is required")
        self.workers = [u.rstrip("/") for u in worker_urls]
        self.forward_timeout = float(forward_timeout)
        self.forward_retries = max(0, int(forward_retries))
        self.forward_retry_backoff = max(0.0, float(forward_retry_backoff))
        self._session: aiohttp.ClientSession | None = None

    @property
    def num_workers(self) -> int:
        return len(self.workers)

    async def startup(self) -> None:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.forward_timeout)
            connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)

    async def shutdown(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    def select_worker(self, task_key: str) -> tuple[int, str]:
        digest = sha1(task_key.encode("utf-8")).digest()
        idx = (
            int.from_bytes(digest[:8], byteorder="big", signed=False) % self.num_workers
        )
        return idx, self.workers[idx]

    @staticmethod
    def encode_lease(worker_idx: int, worker_lease: str) -> str:
        return f"{worker_idx}:{worker_lease}"

    @staticmethod
    def decode_lease(global_lease: str) -> tuple[int, str]:
        sep = global_lease.index(":")
        return int(global_lease[:sep]), global_lease[sep + 1 :]

    def worker_url(self, worker_idx: int) -> str:
        return self.workers[worker_idx]

    def iter_worker_candidates(self, start_idx: int) -> list[tuple[int, str]]:
        return [
            (
                (start_idx + offset) % self.num_workers,
                self.workers[(start_idx + offset) % self.num_workers],
            )
            for offset in range(self.num_workers)
        ]

    async def _request(
        self,
        method: str,
        worker_url: str,
        path: str,
        payload: dict[str, Any] | None,
        timeout: float | None,
    ) -> tuple[dict[str, Any], int]:
        if self._session is None:
            raise RuntimeError("Router HTTP session is not initialized")

        kwargs: dict[str, Any] = {}
        if payload is not None:
            kwargs["json"] = payload
        if timeout is not None:
            kwargs["timeout"] = aiohttp.ClientTimeout(total=float(timeout))

        max_attempts = self.forward_retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                async with self._session.request(
                    method, f"{worker_url}{path}", **kwargs
                ) as resp:
                    status = resp.status
                    try:
                        body = await resp.json(content_type=None)
                    except Exception:
                        raw_text = await resp.text()
                        body = {
                            "ok": False,
                            "error": "Worker returned non-JSON response",
                            "raw_text": raw_text,
                            "status_code": status,
                        }
                    return body, status
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt >= max_attempts:
                    raise
                logger.warning(
                    "Upstream request failed (%s %s) worker=%s attempt=%d/%d err=%s",
                    method,
                    path,
                    worker_url,
                    attempt,
                    max_attempts,
                    _format_error(exc),
                )
                backoff = self.forward_retry_backoff * attempt
                if backoff > 0:
                    await asyncio.sleep(backoff)

    async def forward(
        self,
        worker_url: str,
        path: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> tuple[dict[str, Any], int]:
        return await self._request("POST", worker_url, path, payload, timeout)

    async def forward_by_lease(
        self,
        global_lease: str,
        path: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> tuple[dict[str, Any], int]:
        worker_idx, worker_lease = self.decode_lease(global_lease)
        url = self.worker_url(worker_idx)
        forwarded_payload = dict(payload)
        forwarded_payload["lease_id"] = worker_lease
        return await self.forward(url, path, forwarded_payload, timeout)

    async def worker_status(
        self, worker_url: str, timeout: float = 10.0
    ) -> tuple[dict[str, Any], int]:
        return await self._request("GET", worker_url, "/status", None, timeout)


ROUTER: Router | None = None


def _worker_unreachable(
    *,
    worker_idx: int,
    worker_url: str,
    path: str,
    exc: BaseException,
    lease_id: str | None = None,
    task_key: str | None = None,
) -> JSONResponse:
    payload: dict[str, Any] = {
        "ok": False,
        "error": f"Worker unreachable: {_format_error(exc)}",
        "worker_idx": worker_idx,
        "worker_url": worker_url,
        "path": path,
    }
    if lease_id:
        payload["lease_id"] = lease_id
    if task_key:
        payload["task_key"] = task_key
    return JSONResponse(payload, status_code=502)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"ok": True}


@app.get("/status")
async def status() -> JSONResponse:
    if ROUTER is None:
        return JSONResponse(
            {"ok": False, "error": "Router is not initialized"}, status_code=500
        )

    async def _fetch(idx: int, url: str) -> dict[str, Any]:
        try:
            data, _ = await ROUTER.worker_status(url, timeout=10)
            return {"worker_idx": idx, "url": url, **data}
        except Exception as exc:
            return {
                "worker_idx": idx,
                "url": url,
                "ok": False,
                "error": _format_error(exc),
            }

    workers = await asyncio.gather(
        *[_fetch(idx, url) for idx, url in enumerate(ROUTER.workers)]
    )
    return JSONResponse(
        {"ok": True, "num_workers": ROUTER.num_workers, "workers": workers}
    )


@app.post("/allocate")
async def allocate(request: Request) -> JSONResponse:
    if ROUTER is None:
        return JSONResponse(
            {"ok": False, "error": "Router is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    task_key = data.get("task_key", "")
    request_id = data.get("request_id")

    if not task_key:
        return JSONResponse(
            {"ok": False, "error": "task_key is required"}, status_code=400
        )

    try:
        payload = {"task_key": task_key, "request_id": request_id}
        primary_idx, _ = ROUTER.select_worker(str(task_key))
        upstream_errors: list[dict[str, Any]] = []
        for worker_idx, worker_url in ROUTER.iter_worker_candidates(primary_idx):
            try:
                result, code = await ROUTER.forward(worker_url, "/allocate", payload)
                if worker_idx != primary_idx:
                    logger.warning(
                        "Primary worker unreachable for /allocate task_key=%s; fallback worker_idx=%d url=%s",
                        task_key,
                        worker_idx,
                        worker_url,
                    )
                if result.get("ok") and "lease_id" in result:
                    result["lease_id"] = Router.encode_lease(
                        worker_idx, str(result["lease_id"])
                    )
                    result["worker_idx"] = worker_idx

                return JSONResponse(
                    result, status_code=_status_from_payload(result, code)
                )
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning(
                    "Worker unreachable for /allocate task_key=%s worker_idx=%d url=%s err=%s",
                    task_key,
                    worker_idx,
                    worker_url,
                    _format_error(exc),
                )
                upstream_errors.append(
                    {
                        "worker_idx": worker_idx,
                        "worker_url": worker_url,
                        "detail": _format_error(exc),
                    }
                )

        return JSONResponse(
            {
                "ok": False,
                "error": "Worker unreachable: all candidates failed for /allocate",
                "task_key": task_key,
                "primary_worker_idx": primary_idx,
                "upstream_errors": upstream_errors,
            },
            status_code=502,
        )
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


async def _lease_proxy(path: str, request: Request) -> JSONResponse:
    if ROUTER is None:
        return JSONResponse(
            {"ok": False, "error": "Router is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    global_lease = data.get("lease_id", "")
    if not global_lease:
        return JSONResponse(
            {"ok": False, "error": "lease_id is required"}, status_code=400
        )

    try:
        worker_idx, worker_lease = ROUTER.decode_lease(str(global_lease))
        worker_url = ROUTER.worker_url(worker_idx)
    except (ValueError, IndexError) as exc:
        return JSONResponse(
            {"ok": False, "error": f"Invalid lease_id format: {exc}"}, status_code=400
        )

    payload = dict(data)
    payload["lease_id"] = worker_lease

    try:
        result, code = await ROUTER.forward(worker_url, path, payload)
        return JSONResponse(result, status_code=_status_from_payload(result, code))
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        return _worker_unreachable(
            worker_idx=worker_idx,
            worker_url=worker_url,
            path=path,
            exc=exc,
            lease_id=str(global_lease),
        )
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/heartbeat")
async def heartbeat(request: Request) -> JSONResponse:
    return await _lease_proxy("/heartbeat", request)


@app.post("/reset")
async def reset(request: Request) -> JSONResponse:
    return await _lease_proxy("/reset", request)


@app.post("/exec_tool")
async def exec_tool(request: Request) -> JSONResponse:
    return await _lease_proxy("/exec_tool", request)


@app.post("/evaluate")
async def evaluate(request: Request) -> JSONResponse:
    return await _lease_proxy("/evaluate", request)


@app.post("/close")
async def close(request: Request) -> JSONResponse:
    return await _lease_proxy("/close", request)


@app.on_event("startup")
async def _on_startup() -> None:
    if ROUTER is not None:
        await ROUTER.startup()


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    if ROUTER is not None:
        await ROUTER.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B-layer: terminal env router server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("ROUTER_PORT", "18080"))
    )
    parser.add_argument(
        "--workers",
        type=str,
        default=os.getenv("WORKER_URLS", ""),
        help="Comma-separated worker URLs, e.g. http://w0:18081,http://w1:18081",
    )
    parser.add_argument(
        "--forward-timeout",
        type=float,
        default=float(os.getenv("ROUTER_FORWARD_TIMEOUT", "600.0")),
        help="HTTP timeout (seconds) when forwarding to a worker",
    )
    parser.add_argument(
        "--forward-retries",
        type=int,
        default=int(os.getenv("ROUTER_FORWARD_RETRIES", "1")),
        help="Retries for transient worker connection errors",
    )
    parser.add_argument(
        "--forward-retry-backoff",
        type=float,
        default=float(os.getenv("ROUTER_FORWARD_RETRY_BACKOFF", "0.2")),
        help="Linear backoff (seconds) between worker retries",
    )
    return parser.parse_args()


def main() -> None:
    global ROUTER
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s %(levelname)s %(name)s] %(message)s"
    )

    worker_urls = [u.strip() for u in args.workers.split(",") if u.strip()]
    if not worker_urls:
        raise SystemExit(
            "ERROR: --workers (or WORKER_URLS env) must list at least one worker URL"
        )

    ROUTER = Router(
        worker_urls=worker_urls,
        forward_timeout=args.forward_timeout,
        forward_retries=args.forward_retries,
        forward_retry_backoff=args.forward_retry_backoff,
    )
    logger.info(
        "Starting router on %s:%s  workers=%s  forward_timeout=%s  forward_retries=%s  forward_retry_backoff=%s",
        args.host,
        args.port,
        worker_urls,
        args.forward_timeout,
        args.forward_retries,
        args.forward_retry_backoff,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
