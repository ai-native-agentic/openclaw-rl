from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..custom_types import RunContext, TaskSpec, TaskTimeouts
from ..request_utils import json_payload
from .terminal_env import TerminalEnv

logger = logging.getLogger("terminal.env.worker")
app = FastAPI()


def _parse_timeout_overrides(
    base: TaskTimeouts, payload: dict[str, Any] | None
) -> TaskTimeouts:
    if not isinstance(payload, dict):
        return base

    def _pick(key: str, default: float) -> float:
        raw = payload.get(key, default)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    return TaskTimeouts(
        ensure_image=_pick("ensure_image", base.ensure_image),
        reset_session=_pick("reset_session", base.reset_session),
        close_session=_pick("close_session", base.close_session),
        eval=_pick("eval", base.eval),
    )


def _build_task_spec(task_meta: dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        task_name=str(task_meta.get("task_name", "unknown")),
        task_path=str(task_meta.get("task_path", "")),
        instruction=str(task_meta.get("instruction", "")),
    )


def _build_run_ctx(
    run_ctx_payload: dict[str, Any] | None, default_log_dir: Path
) -> RunContext:
    payload = run_ctx_payload if isinstance(run_ctx_payload, dict) else {}
    uid = str(payload.get("uid") or uuid.uuid4().hex[:8])
    try:
        group_index = int(payload.get("group_index") or 0)
    except (TypeError, ValueError):
        group_index = 0
    try:
        sample_index = int(payload.get("sample_index") or 0)
    except (TypeError, ValueError):
        sample_index = 0

    log_dir_raw = payload.get("log_dir")
    if isinstance(log_dir_raw, str) and log_dir_raw:
        log_dir = Path(log_dir_raw).resolve()
    else:
        log_dir = default_log_dir.resolve()

    return RunContext(
        uid=uid,
        group_index=group_index,
        sample_index=sample_index,
        log_dir=log_dir,
    )


class CapacityError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass
class RunSlot:
    run_lease_id: str
    task_key: str
    env: TerminalEnv
    last_used_ts: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class TaskSlot:
    task_key: str
    runs: dict[str, RunSlot] = field(default_factory=dict)
    created_ts: float = field(default_factory=time.time)
    last_used_ts: float = field(default_factory=time.time)


class WorkerPool:
    def __init__(
        self,
        *,
        max_tasks: int,
        max_runs_per_task: int,
        run_idle_ttl: int,
        output_root: str,
        default_timeouts: TaskTimeouts,
        idempotency_ttl: int = 300,
        max_concurrent_closes: int = 8,
    ) -> None:
        self.max_tasks = max_tasks
        self.max_runs_per_task = max_runs_per_task
        self.run_idle_ttl = run_idle_ttl
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.default_timeouts = default_timeouts
        self.idempotency_ttl = idempotency_ttl

        self._tasks: dict[str, TaskSlot] = {}
        self._run_to_task: dict[str, str] = {}
        self._idempotency: dict[tuple[str, str], tuple[str, float]] = {}
        self._lock = asyncio.Lock()

        self._close_sem = asyncio.Semaphore(max_concurrent_closes)
        self._closing_tasks: set[asyncio.Task] = set()

    def _new_env(self) -> TerminalEnv:
        return TerminalEnv()

    async def _close_run_slot(
        self, task_key: str, run_lease_id: str, run_slot: RunSlot, *, reason: str
    ) -> None:
        logger.warning("%s %s (task=%s)", reason, run_lease_id, task_key)
        async with self._close_sem:
            async with run_slot.lock:
                try:
                    await run_slot.env.close()
                except Exception:
                    logger.exception("Failed to close run session %s", run_lease_id)

    def _schedule_close(
        self, task_key: str, run_lease_id: str, run_slot: RunSlot, *, reason: str
    ) -> None:
        task = asyncio.create_task(
            self._close_run_slot(task_key, run_lease_id, run_slot, reason=reason)
        )
        self._closing_tasks.add(task)
        task.add_done_callback(self._closing_tasks.discard)

    def _reap_idle_locked(self) -> list[tuple[str, str, RunSlot]]:
        now = time.time()
        expired_slots: list[tuple[str, str, RunSlot]] = []

        expired_idem = [
            k
            for k, (_, ts) in self._idempotency.items()
            if now - ts > self.idempotency_ttl
        ]
        for k in expired_idem:
            self._idempotency.pop(k, None)

        for task_key, task_slot in list(self._tasks.items()):
            expired_runs: list[str] = []
            for rid, rslot in task_slot.runs.items():
                if now - rslot.last_used_ts > self.run_idle_ttl:
                    expired_runs.append(rid)

            for rid in expired_runs:
                rslot = task_slot.runs.pop(rid, None)
                self._run_to_task.pop(rid, None)
                if rslot is not None:
                    expired_slots.append((task_key, rid, rslot))

            if task_slot.runs:
                task_slot.last_used_ts = max(
                    r.last_used_ts for r in task_slot.runs.values()
                )
            else:
                logger.info("Reaping empty task slot: %s", task_key)
                self._tasks.pop(task_key, None)

        return expired_slots

    def _get_run_slot(self, run_lease_id: str) -> RunSlot:
        task_key = self._run_to_task.get(run_lease_id)
        if task_key is None:
            raise KeyError(f"Unknown run_lease_id: {run_lease_id}")
        task_slot = self._tasks.get(task_key)
        if task_slot is None:
            raise KeyError(f"Run {run_lease_id} points to missing task slot")
        run_slot = task_slot.runs.get(run_lease_id)
        if run_slot is None:
            raise KeyError(f"Run {run_lease_id} not found in task slot")
        return run_slot

    async def allocate(
        self, task_key: str, request_id: str | None = None
    ) -> dict[str, Any]:
        async with self._lock:
            expired_slots = self._reap_idle_locked()

            if request_id:
                idem_key = (task_key, request_id)
                cached = self._idempotency.get(idem_key)
                if cached is not None:
                    run_lease_id, _ = cached
                    if run_lease_id in self._run_to_task:
                        return {"lease_id": run_lease_id, "reused": True}

            task_slot = self._tasks.get(task_key)
            if task_slot is None:
                if len(self._tasks) >= self.max_tasks:
                    raise CapacityError(
                        "TASK_SLOTS_EXHAUSTED",
                        f"Worker at task capacity: {len(self._tasks)}/{self.max_tasks}",
                    )
                task_slot = TaskSlot(task_key=task_key)
                self._tasks[task_key] = task_slot

            if len(task_slot.runs) >= self.max_runs_per_task:
                raise CapacityError(
                    "RUN_SLOTS_EXHAUSTED",
                    f"Task {task_key} at run capacity: {len(task_slot.runs)}/{self.max_runs_per_task}",
                )

            env = self._new_env()
            run_lease_id = f"run-{uuid.uuid4().hex[:16]}"
            run_slot = RunSlot(run_lease_id=run_lease_id, task_key=task_key, env=env)
            task_slot.runs[run_lease_id] = run_slot
            task_slot.last_used_ts = time.time()
            self._run_to_task[run_lease_id] = task_key

            if request_id:
                self._idempotency[(task_key, request_id)] = (run_lease_id, time.time())

        for tk, rid, rslot in expired_slots:
            self._schedule_close(tk, rid, rslot, reason="Reaping idle run slot")

        return {"lease_id": run_lease_id, "reused": False}

    async def heartbeat(self, run_lease_id: str) -> None:
        async with self._lock:
            run_slot = self._get_run_slot(run_lease_id)
        async with run_slot.lock:
            run_slot.last_used_ts = time.time()

    async def reset(
        self,
        run_lease_id: str,
        task_meta: dict[str, Any],
        run_ctx_payload: dict[str, Any] | None = None,
        task_timeouts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(task_meta, dict):
            raise ValueError("task_meta must be a dict")

        async with self._lock:
            run_slot = self._get_run_slot(run_lease_id)

        run_ctx = _build_run_ctx(
            run_ctx_payload, default_log_dir=self.output_root / "AgentRunner_Output"
        )
        timeouts = _parse_timeout_overrides(self.default_timeouts, task_timeouts)
        task_spec = _build_task_spec(task_meta)

        async with run_slot.lock:
            user_msg, tool_schemas = await run_slot.env.reset(
                task_meta=task_meta,
                task_spec=task_spec,
                run_ctx=run_ctx,
                timeouts=timeouts,
            )
            run_slot.last_used_ts = time.time()
            return {"user_msg": user_msg, "tool_schemas": tool_schemas}

    async def exec_tool(
        self, run_lease_id: str, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> str:
        async with self._lock:
            run_slot = self._get_run_slot(run_lease_id)
        async with run_slot.lock:
            observation = await run_slot.env.exec_tool(tool_name, arguments or {})
            run_slot.last_used_ts = time.time()
            return str(observation)

    async def evaluate(self, run_lease_id: str) -> float:
        async with self._lock:
            run_slot = self._get_run_slot(run_lease_id)
        async with run_slot.lock:
            score = await run_slot.env.evaluate()
            run_slot.last_used_ts = time.time()
            return float(score)

    async def close_run(self, run_lease_id: str) -> bool:
        async with self._lock:
            task_key = self._run_to_task.pop(run_lease_id, None)
            if task_key is None:
                logger.debug(
                    "close_run: lease %s already gone, nothing to do.", run_lease_id
                )
                return False
            task_slot = self._tasks.get(task_key)
            run_slot = task_slot.runs.pop(run_lease_id, None) if task_slot else None
            if task_slot is not None and not task_slot.runs:
                self._tasks.pop(task_key, None)
                logger.info("Removed empty task slot: %s", task_key)

        if run_slot is not None:
            self._schedule_close(
                task_key, run_lease_id, run_slot, reason="Closing run slot"
            )
        return True

    async def status(self) -> dict[str, Any]:
        async with self._lock:
            tasks_info: dict[str, Any] = {}
            total_runs = 0
            for tk, ts in self._tasks.items():
                tasks_info[tk] = {"active_runs": len(ts.runs)}
                total_runs += len(ts.runs)

            return {
                "max_tasks": self.max_tasks,
                "active_tasks": len(self._tasks),
                "max_runs_per_task": self.max_runs_per_task,
                "total_active_runs": total_runs,
                "pending_closes": len(self._closing_tasks),
                "tasks": tasks_info,
            }

    async def periodic_reap(self, interval: float = 60.0) -> None:
        while True:
            await asyncio.sleep(interval)
            try:
                async with self._lock:
                    expired_slots = self._reap_idle_locked()
                for tk, rid, rslot in expired_slots:
                    self._schedule_close(
                        tk, rid, rslot, reason="Periodic reaper: idle run slot"
                    )
                if expired_slots:
                    logger.info(
                        "Periodic reaper cleaned up %d idle run slots",
                        len(expired_slots),
                    )
            except Exception:
                logger.exception("Periodic reaper error")

    async def shutdown(self) -> None:
        async with self._lock:
            slots_to_close: list[tuple[str, str, RunSlot]] = []
            for task_key, task_slot in self._tasks.items():
                for run_lease_id, run_slot in task_slot.runs.items():
                    slots_to_close.append((task_key, run_lease_id, run_slot))
            self._tasks.clear()
            self._run_to_task.clear()
            self._idempotency.clear()

        for task_key, run_lease_id, run_slot in slots_to_close:
            self._schedule_close(
                task_key,
                run_lease_id,
                run_slot,
                reason="Closing run slot during shutdown",
            )

        if self._closing_tasks:
            logger.info(
                "Shutdown: waiting for %d pending close tasks...",
                len(self._closing_tasks),
            )
            await asyncio.gather(*self._closing_tasks, return_exceptions=True)


POOL: WorkerPool | None = None


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"ok": True}


@app.get("/status")
async def status() -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )
    return JSONResponse({"ok": True, "pool": await POOL.status()})


@app.post("/allocate")
async def allocate(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    task_key = data.get("task_key", "")
    request_id = data.get("request_id")

    if not task_key:
        return JSONResponse(
            {"ok": False, "error": "task_key is required"}, status_code=400
        )

    try:
        result = await POOL.allocate(task_key=str(task_key), request_id=request_id)
        return JSONResponse({"ok": True, **result})
    except CapacityError as exc:
        return JSONResponse(
            {"ok": False, "error": exc.message, "code": exc.code}, status_code=429
        )
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/heartbeat")
async def heartbeat(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    lease_id = data.get("lease_id")
    if not lease_id:
        return JSONResponse(
            {"ok": False, "error": "lease_id is required"}, status_code=400
        )

    try:
        await POOL.heartbeat(str(lease_id))
        return JSONResponse({"ok": True})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/reset")
async def reset(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    lease_id = data.get("lease_id")
    task_meta = data.get("task_meta")
    run_ctx_payload = data.get("run_ctx")
    task_timeouts = data.get("task_timeouts")

    if not lease_id:
        return JSONResponse(
            {"ok": False, "error": "lease_id is required"}, status_code=400
        )
    if not isinstance(task_meta, dict):
        return JSONResponse(
            {"ok": False, "error": "task_meta dict is required"}, status_code=400
        )

    try:
        out = await POOL.reset(
            run_lease_id=str(lease_id),
            task_meta=task_meta,
            run_ctx_payload=run_ctx_payload,
            task_timeouts=task_timeouts,
        )
        return JSONResponse({"ok": True, **out})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/exec_tool")
async def exec_tool(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    lease_id = data.get("lease_id")
    tool_call = data.get("tool_call")

    if not lease_id:
        return JSONResponse(
            {"ok": False, "error": "lease_id is required"}, status_code=400
        )
    if not isinstance(tool_call, dict):
        return JSONResponse(
            {"ok": False, "error": "tool_call dict is required"}, status_code=400
        )

    tool_name = tool_call.get("name")
    arguments = tool_call.get("arguments")

    if not isinstance(tool_name, str) or not tool_name:
        return JSONResponse(
            {"ok": False, "error": "tool_call.name is required"}, status_code=400
        )
    if arguments is not None and not isinstance(arguments, dict):
        return JSONResponse(
            {"ok": False, "error": "tool_call.arguments must be a dict"},
            status_code=400,
        )

    try:
        observation = await POOL.exec_tool(
            str(lease_id), tool_name, arguments=arguments
        )
        return JSONResponse({"ok": True, "observation": observation})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/evaluate")
async def evaluate(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    lease_id = data.get("lease_id")

    if not lease_id:
        return JSONResponse(
            {"ok": False, "error": "lease_id is required"}, status_code=400
        )

    try:
        score = await POOL.evaluate(str(lease_id))
        return JSONResponse({"ok": True, "score": score})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/close")
async def close(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    lease_id = data.get("lease_id")
    if not lease_id:
        return JSONResponse(
            {"ok": False, "error": "lease_id is required"}, status_code=400
        )

    try:
        found = await POOL.close_run(str(lease_id))
        return JSONResponse({"ok": True, "found": found})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


_REAPER_TASK: asyncio.Task | None = None


@app.on_event("startup")
async def _on_startup() -> None:
    global _REAPER_TASK
    if POOL is not None:
        _REAPER_TASK = asyncio.create_task(POOL.periodic_reap(interval=60.0))


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    global POOL, _REAPER_TASK
    if _REAPER_TASK is not None:
        _REAPER_TASK.cancel()
        _REAPER_TASK = None
    if POOL is not None:
        await POOL.shutdown()
        POOL = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C-layer: terminal env worker server")

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("ENV_SERVER_PORT", "18081"))
    )

    parser.add_argument(
        "--max-tasks", type=int, default=int(os.getenv("WORKER_MAX_TASKS", "16"))
    )
    parser.add_argument(
        "--max-runs-per-task",
        type=int,
        default=int(os.getenv("WORKER_MAX_RUNS_PER_TASK", "8")),
    )
    parser.add_argument(
        "--run-idle-ttl",
        type=int,
        default=int(os.getenv("WORKER_RUN_IDLE_TTL", "600")),
        help="Seconds before an idle RunSlot is reaped",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=os.getenv("TBENCH_OUTPUT_ROOT", "build_outputs"),
    )

    parser.add_argument(
        "--ensure-image-timeout",
        type=float,
        default=float(os.getenv("ENSURE_IMAGE_TIMEOUT", "300.0")),
    )
    parser.add_argument(
        "--reset-session-timeout",
        type=float,
        default=float(os.getenv("RESET_SESSION_TIMEOUT", "300.0")),
    )
    parser.add_argument(
        "--close-session-timeout",
        type=float,
        default=float(os.getenv("CLOSE_SESSION_TIMEOUT", "60.0")),
    )
    parser.add_argument(
        "--eval-timeout", type=float, default=float(os.getenv("EVAL_TIMEOUT", "600.0"))
    )
    parser.add_argument(
        "--max-concurrent-closes",
        type=int,
        default=int(os.getenv("WORKER_MAX_CONCURRENT_CLOSES", "10")),
        help="Max concurrent Docker stop operations",
    )

    return parser.parse_args()


def main() -> None:
    global POOL
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s %(levelname)s %(name)s] %(message)s"
    )

    POOL = WorkerPool(
        max_tasks=args.max_tasks,
        max_runs_per_task=args.max_runs_per_task,
        run_idle_ttl=args.run_idle_ttl,
        output_root=args.output_root,
        default_timeouts=TaskTimeouts(
            ensure_image=float(args.ensure_image_timeout),
            reset_session=float(args.reset_session_timeout),
            close_session=float(args.close_session_timeout),
            eval=float(args.eval_timeout),
        ),
        max_concurrent_closes=args.max_concurrent_closes,
    )

    logger.info(
        "Starting worker server on %s:%s  max_tasks=%s  max_runs_per_task=%s",
        args.host,
        args.port,
        args.max_tasks,
        args.max_runs_per_task,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
