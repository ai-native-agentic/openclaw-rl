from __future__ import annotations

import logging
import os
from typing import Any

from slime.utils.http_utils import post

logger = logging.getLogger(__name__)


class TerminalEnvClient:

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.default_max_retries = int(os.getenv("ENV_HTTP_MAX_RETRIES", "10"))
        self.allocate_max_retries = int(os.getenv("ENV_ALLOCATE_MAX_RETRIES", "100"))
        self.evaluate_max_retries = int(os.getenv("ENV_EVALUATE_MAX_RETRIES", "1"))
        self.close_max_retries = int(os.getenv("ENV_CLOSE_MAX_RETRIES", "3"))
        self.exec_tool_max_retries = int(os.getenv("ENV_EXEC_TOOL_MAX_RETRIES", "3"))

    async def allocate(
        self,
        task_key: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        out = await post(
            f"{self.base_url}/allocate",
            {"task_key": task_key, "request_id": request_id},
            max_retries=self.allocate_max_retries,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"allocate failed: {out}")
        return out

    async def heartbeat(self, lease_id: str) -> None:
        out = await post(
            f"{self.base_url}/heartbeat",
            {"lease_id": lease_id},
            max_retries=self.default_max_retries,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"heartbeat failed: {out}")

    async def reset(
        self,
        lease_id: str,
        task_meta: dict[str, Any],
        run_ctx: dict[str, Any],
        task_timeouts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        out = await post(
            f"{self.base_url}/reset",
            {
                "lease_id": lease_id,
                "task_meta": task_meta,
                "run_ctx": run_ctx,
                "task_timeouts": task_timeouts,
            },
            max_retries=self.default_max_retries,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"reset failed: {out}")
        return out

    async def exec_tool(
        self, lease_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        out = await post(
            f"{self.base_url}/exec_tool",
            {
                "lease_id": lease_id,
                "tool_call": {"name": tool_name, "arguments": arguments},
            },
            max_retries=self.exec_tool_max_retries,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"exec_tool failed: {out}")
        return str(out.get("observation", ""))

    async def evaluate(self, lease_id: str) -> float:
        out = await post(
            f"{self.base_url}/evaluate",
            {"lease_id": lease_id},
            max_retries=self.evaluate_max_retries,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"evaluate failed: {out}")
        return float(out.get("score", 0.0))

    async def close(self, lease_id: str) -> None:
        try:
            out = await post(
                f"{self.base_url}/close",
                {"lease_id": lease_id},
                max_retries=self.close_max_retries,
            )
        except Exception as exc:
            error_str = str(exc)
            resp_text = ""
            if hasattr(exc, "response"):
                try:
                    resp_text = exc.response.text
                except Exception:
                    pass
            combined = f"{error_str} {resp_text}"
            if "Unknown run_lease_id" in combined or "Unknown lease" in combined:
                logger.debug("close(%s): lease already gone, nothing to do.", lease_id)
                return
            raise
        if not out.get("ok", False):
            error_msg = str(out.get("error", ""))
            if "Unknown" in error_msg and "lease" in error_msg.lower():
                logger.debug("close(%s): lease already gone, nothing to do.", lease_id)
                return
            raise RuntimeError(f"close failed: {out}")
