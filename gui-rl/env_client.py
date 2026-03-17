from __future__ import annotations

import base64
import os
from typing import Any

from slime.utils.http_utils import post


def _decode_obs(obs: dict[str, Any]) -> dict[str, Any]:
    out = dict(obs or {})
    screenshot_b64 = out.pop("screenshot_b64", None)
    out["screenshot"] = base64.b64decode(screenshot_b64) if screenshot_b64 else b""
    return out


class GuiEnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        # Keep GUI env-control retries small to avoid long hangs on deterministic failures.
        self.default_max_retries = int(os.getenv("GUI_ENV_HTTP_MAX_RETRIES", "10"))
        self.evaluate_max_retries = int(os.getenv("GUI_EVALUATE_MAX_RETRIES", "6"))

    async def allocate(self, episode_id: str) -> dict[str, Any]:
        out = await post(f"{self.base_url}/allocate", {"episode_id": episode_id}, max_retries=self.default_max_retries)
        if not out.get("ok", False):
            raise RuntimeError(f"allocate failed: {out}")
        return out

    async def heartbeat(self, lease_id: str) -> None:
        out = await post(f"{self.base_url}/heartbeat", {"lease_id": lease_id}, max_retries=self.default_max_retries)
        if not out.get("ok", False):
            raise RuntimeError(f"heartbeat failed: {out}")

    async def reset(self, lease_id: str, task_config: dict[str, Any] | None) -> dict[str, Any]:
        out = await post(
            f"{self.base_url}/reset",
            {"lease_id": lease_id, "task_config": task_config},
            max_retries=self.default_max_retries,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"reset failed: {out}")
        return _decode_obs(out["observation"])

    async def get_obs(self, lease_id: str) -> dict[str, Any]:
        out = await post(
            f"{self.base_url}/get_obs",
            {"lease_id": lease_id},
            max_retries=self.default_max_retries,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"get_obs failed: {out}")
        return _decode_obs(out["observation"])

    async def step(self, lease_id: str, action: Any, sleep_after_execution: float) -> tuple[dict[str, Any], float, bool, dict]:
        out = await post(
            f"{self.base_url}/step",
            {
                "lease_id": lease_id,
                "action": action,
                "sleep_after_execution": sleep_after_execution,
            },
            max_retries=self.default_max_retries,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"step failed: {out}")
        obs = _decode_obs(out["observation"])
        return obs, float(out.get("reward", 0.0)), bool(out.get("done", False)), out.get("info", {})

    async def evaluate(self, lease_id: str) -> float:
        out = await post(
            f"{self.base_url}/evaluate",
            {"lease_id": lease_id},
            max_retries=self.evaluate_max_retries,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"evaluate failed: {out}")
        return float(out["score"])

    async def start_recording(self, lease_id: str) -> None:
        out = await post(
            f"{self.base_url}/start_recording",
            {"lease_id": lease_id},
            max_retries=self.default_max_retries,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"start_recording failed: {out}")

    async def end_recording(self, lease_id: str, out_path: str) -> None:
        out = await post(
            f"{self.base_url}/end_recording",
            {"lease_id": lease_id, "out_path": out_path},
            max_retries=self.default_max_retries,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"end_recording failed: {out}")

    async def close(self, lease_id: str) -> None:
        out = await post(f"{self.base_url}/close", {"lease_id": lease_id}, max_retries=self.default_max_retries)
        if not out.get("ok", False):
            raise RuntimeError(f"close failed: {out}")
