from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import torch

from slime.rollout.data_source import DataSource
from slime.utils.types import Sample


def _pop_first(buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples


class GuiMetaDataSource(DataSource):
    """
    Data source for GUI tasks from evaluation_examples meta files.

    This bypasses jsonl prompt-data generation and directly loads:
    - GUI_TEST_CONFIG_BASE_DIR
    - GUI_TRAIN_META_PATH
    """

    def __init__(self, args):
        self.args = args
        self.buffer: list[list[Sample]] = []
        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0

        base_dir = Path(
            os.getenv("GUI_TEST_CONFIG_BASE_DIR", str(Path(__file__).resolve().parent / "evaluation_examples"))
        )
        meta_path = Path(os.getenv("GUI_TRAIN_META_PATH", str(base_dir / "train_nochrome.json")))

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        tasks: list[dict] = []
        for domain, example_ids in meta.items():
            for example_id in example_ids:
                cfg_path = base_dir / "examples" / str(domain) / f"{example_id}.json"
                if not cfg_path.exists():
                    continue
                with open(cfg_path, "r", encoding="utf-8") as cf:
                    task_cfg = json.load(cf)
                tasks.append(
                    {
                        "domain": str(domain),
                        "example_id": str(example_id),
                        "instruction": task_cfg.get("instruction", ""),
                        "task_config": task_cfg,
                    }
                )

        if not tasks:
            raise RuntimeError(f"No tasks loaded from {meta_path}")

        self.tasks = tasks

    def _make_prompt_samples(self, num_samples: int) -> list[Sample]:
        out: list[Sample] = []
        for _ in range(num_samples):
            task = self.tasks[self.sample_offset % len(self.tasks)]
            self.sample_offset += 1
            if self.sample_offset % len(self.tasks) == 0:
                self.epoch_id += 1

            sample = Sample(
                prompt=task["instruction"],
                label="",
                metadata={
                    "domain": task["domain"],
                    "example_id": task["example_id"],
                    "instruction": task["instruction"],
                    "task_config": task["task_config"],
                },
            )
            out.append(sample)
        return out

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        samples = _pop_first(self.buffer, num_samples)
        num_samples -= len(samples)
        if num_samples <= 0:
            return samples

        prompt_samples = self._make_prompt_samples(num_samples)
        groups: list[list[Sample]] = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                s = copy.deepcopy(prompt_sample)
                s.group_index = self.sample_group_index
                s.index = self.sample_index
                self.sample_index += 1
                group.append(s)
            self.sample_group_index += 1
            groups.append(group)
        return samples + groups

    def add_samples(self, samples: list[list[Sample]]):
        if samples:
            self.buffer.extend(samples)

    def save(self, rollout_id):
        state = {
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "sample_offset": self.sample_offset,
        }
        path = os.path.join(self.args.save, f"rollout/gui_meta_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)

    def load(self, rollout_id=None):
        if self.args.load is None:
            return
        path = os.path.join(self.args.load, f"rollout/gui_meta_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            return
        state = torch.load(path)
        self.epoch_id = state.get("epoch_id", 0)
        self.sample_group_index = state.get("sample_group_index", 0)
        self.sample_index = state.get("sample_index", 0)
        self.sample_offset = state.get("sample_offset", 0)
