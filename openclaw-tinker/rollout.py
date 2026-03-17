"""Unified async rollout worker for all three training methods.

Bridges the API server (RL / OPD / Combine) and the Tinker trainer.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import Optional

from api_server import OpenClawRLServer, OpenClawOPDServer, OpenClawCombineServer
from config import TinkerConfig
from data_formatter import TrainingSample


class RolloutWorker:
    """Manages the FastAPI proxy server and collects training samples.

    Works with any of the three server types (RL, OPD, Combine).
    The server class is selected based on config.method during construction.
    """

    def __init__(
        self,
        config: TinkerConfig,
        sampling_client=None,
        scorer=None,
    ):
        self.config = config
        self.running = True
        self.output_queue: queue.Queue = queue.Queue(maxsize=100_000)
        self.worker_thread: Optional[threading.Thread] = None
        self._submission_enabled = threading.Event()

        # Select server class based on method
        method = config.method.lower()
        if method == "rl":
            self._server = OpenClawRLServer(
                config=config, output_queue=self.output_queue,
                submission_enabled=self._submission_enabled,
                sampling_client=sampling_client, prm_scorer=scorer,
            )
        elif method == "opd":
            self._server = OpenClawOPDServer(
                config=config, output_queue=self.output_queue,
                submission_enabled=self._submission_enabled,
                sampling_client=sampling_client, opd_scorer=scorer,
            )
        elif method == "combine":
            self._server = OpenClawCombineServer(
                config=config, output_queue=self.output_queue,
                submission_enabled=self._submission_enabled,
                sampling_client=sampling_client, scorer=scorer,
            )
        else:
            raise ValueError(f"Unknown method: {method!r} (expected 'rl', 'opd', or 'combine')")

    def start(self):
        self._server.start()
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(
                target=lambda: asyncio.run(self._keepalive()), daemon=True
            )
            self.worker_thread.start()

    async def _keepalive(self):
        while self.running:
            await asyncio.sleep(1.0)

    def stop(self):
        self.running = False
        self._submission_enabled.clear()
        self._server.stop()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def pause_submission(self):
        if self._submission_enabled.is_set():
            self._submission_enabled.clear()
            self._server.purge_record_files()

    def resume_submission(self):
        if not self._submission_enabled.is_set():
            self._submission_enabled.set()

    def update_sampling_client(self, client):
        self._server.update_sampling_client(client)

    def drain_eval_scores(self) -> list[float]:
        return self._server.drain_eval_scores()

    def reset_eval_scores(self):
        self._server.reset_eval_scores()


async def drain_output_queue(
    batch_size: int, worker: RolloutWorker
) -> list[list[TrainingSample]]:
    """Block until batch_size sample groups are collected."""
    data: list[list[TrainingSample]] = []
    pending: dict[int, list[TrainingSample]] = {}
    start = time.time()
    last_log = start

    while len(data) < batch_size:
        while True:
            try:
                group_id, group = worker.output_queue.get_nowait()
                pending[group_id] = group
            except queue.Empty:
                break

        for gid in sorted(list(pending.keys())):
            if len(data) >= batch_size:
                break
            data.append(pending.pop(gid))

        if time.time() - last_log > 30:
            print(
                f"[Rollout] waiting for samples: {len(data)}/{batch_size} "
                f"queue={worker.output_queue.qsize()}",
                flush=True,
            )
            last_log = time.time()

        if len(data) < batch_size:
            await asyncio.sleep(0.05)

    print(f"[Rollout] drained {len(data)} groups in {time.time() - start:.2f}s", flush=True)
    return data
