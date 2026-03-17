from __future__ import annotations

import asyncio
import copy
import faulthandler
import json
import logging
import os
import re
import shutil
import sys
import traceback
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from env_client import GuiEnvClient
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.misc import load_function
from slime.utils.types import Sample

logger = logging.getLogger(__name__)
faulthandler.enable(file=sys.stderr, all_threads=True)

_ANSI_RESET = "\033[0m"
_ANSI_COLORS = {
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "none": "",
}


def _gui_log(message: str, *args: Any) -> None:
    color_name = os.getenv("GUI_LOG_COLOR", "yellow").strip().lower()
    color_code = _ANSI_COLORS.get(color_name, _ANSI_COLORS["yellow"])
    prefix = "[GUI]"
    if color_code:
        prefix = f"{color_code}{prefix}{_ANSI_RESET}"
    logger.info(f"{prefix} {message}", *args)


@lru_cache(maxsize=1)
def _get_gui_trajectory_semaphore() -> asyncio.Semaphore:
    max_envs = max(1, int(os.getenv("GUI_POOL_MAX_ENVS", "4")))
    concurrency = max(1, int(os.getenv("GUI_TRAJECTORY_CONCURRENCY", str(max_envs))))
    return asyncio.Semaphore(concurrency)


@lru_cache(maxsize=1)
def _get_task_clear_lock() -> asyncio.Lock:
    return asyncio.Lock()


@lru_cache(maxsize=1)
def _get_cleared_task_keys() -> set[str]:
    return set()


@lru_cache(maxsize=16)
def _load_meta_pairs(meta_path: str) -> list[tuple[str, str]]:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    pairs: list[tuple[str, str]] = []
    for domain, examples in meta.items():
        for example_id in examples:
            pairs.append((str(domain), str(example_id)))

    shuffle_seed_str = os.getenv("GUI_TASK_SHUFFLE_SEED", "42")
    if shuffle_seed_str.strip().lower() not in {"", "none", "-1"}:
        import random
        rng = random.Random(int(shuffle_seed_str))
        rng.shuffle(pairs)
        _gui_log("Shuffled %d task pairs with seed=%s", len(pairs), shuffle_seed_str)

    return pairs


def _load_task_config(base_dir: str, domain: str, example_id: str) -> dict[str, Any]:
    cfg_path = Path(base_dir) / "examples" / domain / f"{example_id}.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_image(image_bytes: bytes, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(image_bytes)


def _sample_task_info(sample: Sample, evaluation: bool = False) -> tuple[str, dict[str, Any] | None, str, str]:
    metadata = sample.metadata or {}
    instruction = metadata.get("instruction")
    task_config = metadata.get("task_config")
    domain = metadata.get("domain", "default")
    example_id = metadata.get("example_id", f"sample_{sample.index if sample.index is not None else uuid.uuid4().hex[:8]}")

    if not instruction and isinstance(sample.prompt, str):
        prompt = sample.prompt.strip()
        if prompt.startswith("{") and prompt.endswith("}"):
            try:
                obj = json.loads(prompt)
                instruction = obj.get("instruction") or instruction
                task_config = obj.get("task_config") or task_config
                domain = obj.get("domain", domain)
                example_id = obj.get("example_id", example_id)
            except Exception:
                instruction = prompt
        else:
            instruction = prompt

    if task_config is None:
        base_dir = os.getenv(
            "GUI_TEST_CONFIG_BASE_DIR",
            str(Path(__file__).resolve().parent / "evaluation_examples"),
        )
        default_meta = "test_nogdrive.json" if evaluation else "train_nochrome.json"
        env_meta_key = "GUI_EVAL_META_PATH" if evaluation else "GUI_TRAIN_META_PATH"
        meta_path = os.getenv(env_meta_key, str(Path(base_dir) / default_meta))

        if isinstance(sample.prompt, str) and "/" in sample.prompt:
            maybe_domain, maybe_example_id = sample.prompt.split("/", 1)
            domain = maybe_domain
            example_id = maybe_example_id
        elif isinstance(sample.prompt, str) and re.fullmatch(r"[0-9a-fA-F-]{36}", sample.prompt):
            example_id = sample.prompt
        else:
            try:
                pairs = _load_meta_pairs(meta_path)
                sample_key = sample.group_index if sample.group_index is not None else sample.index or 0
                if pairs:
                    domain, example_id = pairs[int(sample_key) % len(pairs)]
            except Exception:
                logger.exception("Failed to resolve task pair from meta file: %s", meta_path)

        try:
            task_config = _load_task_config(base_dir, domain, example_id)
            if not instruction:
                instruction = str(task_config.get("instruction", ""))
        except Exception:
            logger.exception("Failed to load task config for %s/%s", domain, example_id)

    return instruction or "", task_config, str(domain), str(example_id)


def _build_result_dir(args: Any, domain: str, example_id: str, sample: Sample) -> Path:
    base = Path(os.getenv("GUI_RESULT_DIR", "./results"))
    action_space = os.getenv("GUI_ACTION_SPACE", "pyautogui")
    observation_type = os.getenv("GUI_OBSERVATION_TYPE", "screenshot")
    model_name = getattr(args, "hf_checkpoint", "gui-policy-model")
    model_tag = re.sub(r"[\\/]+", "_", str(model_name)).lstrip("_")
    # IMPORTANT: each sample must have an isolated output dir.
    # Using group_index alone causes all samples in the same prompt group to
    # write into one traj.jsonl/step_*.png and corrupt each other's traces.
    if sample.index is not None:
        run_idx = f"s{sample.index}"
    elif sample.group_index is not None:
        run_idx = f"g{sample.group_index}"
    else:
        run_idx = f"u{uuid.uuid4().hex[:8]}"
    out = base / action_space / observation_type / model_tag / domain / example_id / run_idx
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_task_dir(args: Any, domain: str, example_id: str) -> Path:
    base = Path(os.getenv("GUI_RESULT_DIR", "./results"))
    action_space = os.getenv("GUI_ACTION_SPACE", "pyautogui")
    observation_type = os.getenv("GUI_OBSERVATION_TYPE", "screenshot")
    model_name = getattr(args, "hf_checkpoint", "gui-policy-model")
    model_tag = re.sub(r"[\\/]+", "_", str(model_name)).lstrip("_")
    task_dir = base / action_space / observation_type / model_tag / domain / example_id
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


async def _maybe_clear_task_result_dir(args: Any, domain: str, example_id: str, sample: Sample) -> None:
    if os.getenv("GUI_CLEAR_TASK_RESULT_ON_START", "1").strip().lower() in {"0", "false", "no"}:
        return

    task_dir = _build_task_dir(args, domain, example_id)
    group_part = (
        f"group_{sample.group_index}" if sample.group_index is not None else f"sample_{sample.index if sample.index is not None else 'na'}"
    )
    clear_key = f"{domain}/{example_id}/{group_part}"
    lock = _get_task_clear_lock()
    cleared_keys = _get_cleared_task_keys()

    async with lock:
        if clear_key in cleared_keys:
            return
        if task_dir.exists():
            shutil.rmtree(task_dir)
        task_dir.mkdir(parents=True, exist_ok=True)
        cleared_keys.add(clear_key)
        _gui_log("cleared task result dir for %s", clear_key)


def _create_gui_agent(args: Any, *, max_steps: int, max_image_history_length: int, result_dir: Path):
    agent_cls_path = getattr(args, "gui_agent_class_path", None) or os.getenv("GUI_AGENT_CLASS_PATH")
    if not agent_cls_path:
        raise RuntimeError("GUI_AGENT_CLASS_PATH is required for GUI rollout framework.")
    agent_cls = load_function(agent_cls_path)
    return agent_cls(
        model=getattr(args, "hf_checkpoint", "gui-policy-model"),
        max_steps=max_steps,
        max_image_history_length=max_image_history_length,
        action_space=os.getenv("GUI_ACTION_SPACE", "pyautogui"),
        observation_type=os.getenv("GUI_OBSERVATION_TYPE", "screenshot"),
        coordinate_type=os.getenv("GUI_COORDINATE_TYPE", "relative"),
        example_result_dir=str(result_dir),
    )


def _build_dynamic_history_samples(
    args: Any,
    state: GenerateState,
    agent: Any,
    base_sample: Sample,
    step_snapshots: list[dict[str, Any]],
    outcome_reward: float,
    prm_score_by_step: dict[int, float] | None = None,
) -> list[Sample]:
    reward_key = getattr(args, "reward_key", None) or "score"
    dynamic_samples: list[Sample] = []

    for snapshot in step_snapshots:
        step_idx = int(snapshot["step_idx"])
        messages = snapshot["train_messages"]
        response_text = snapshot["response_text"]
        tool_spec = snapshot.get("tool_spec")
        token_ids, loss_mask, mm_train = agent.build_train_data(
            args=args,
            state=state,
            train_messages=messages,
            tool_spec=tool_spec,
        )

        # Each snapshot is one step context + current response.
        # Convert full-sequence mask to response-suffix mask expected by training.
        active_positions = [i for i in range(len(loss_mask)) if i < len(token_ids) and int(loss_mask[i]) == 1]
        if not active_positions:
            continue
        response_start = active_positions[0]
        response_length = len(token_ids) - response_start
        if response_length <= 0:
            continue
        child_loss_mask = [
            int(loss_mask[i]) if i < len(loss_mask) else 0 for i in range(response_start, len(token_ids))
        ]

        child_reward = {"score": float(outcome_reward)}
        child_reward[reward_key] = float(outcome_reward)
        child_reward_for_sample = None if getattr(args, "prm_enable", False) else child_reward

        child_metadata = copy.deepcopy(base_sample.metadata or {})
        # For dynamic-history GRPO, each child sample is exactly one step's
        # prompt+response training unit. Keep metadata minimal and explicit.
        child_metadata["dynamic_step_index"] = step_idx
        child_metadata["dynamic_outcome_reward"] = float(outcome_reward)
        if prm_score_by_step is not None:
            child_metadata["step_wise"] = {
                # For dynamic_history, rollout will infer span from loss_mask.
                "step_scores": [float(prm_score_by_step.get(step_idx, 0.0))],
                "step_indices": [int(step_idx)],
            }

        child = Sample(
            group_index=base_sample.group_index,
            index=base_sample.index,
            prompt=base_sample.prompt,
            tokens=token_ids,
            multimodal_inputs=base_sample.multimodal_inputs,
            multimodal_train_inputs=mm_train,
            response=response_text,
            response_length=response_length,
            label=base_sample.label,
            reward=child_reward_for_sample,
            loss_mask=child_loss_mask,
            weight_versions=list(base_sample.weight_versions),
            rollout_log_probs=None,
            rollout_routed_experts=None,
            remove_sample=base_sample.remove_sample,
            status=base_sample.status,
            metadata=child_metadata,
            generate_function_path=base_sample.generate_function_path,
            train_metadata=base_sample.train_metadata,
            non_generation_time=base_sample.non_generation_time,
            spec_info=base_sample.spec_info,
            prefix_cache_info=base_sample.prefix_cache_info,
        )
        dynamic_samples.append(child)

    return dynamic_samples


async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample | list[Sample]:
    assert not args.partial_rollout, "Partial rollout is not supported for GUI rollout."

    instruction, task_config, domain, example_id = _sample_task_info(sample, evaluation=evaluation)
    await _maybe_clear_task_result_dir(args, domain, example_id, sample)
    result_dir = _build_result_dir(args, domain, example_id, sample)
    traj_path = result_dir / "traj.jsonl"

    env_url = os.getenv("GUI_ENV_SERVER_URL", "http://127.0.0.1:18080")
    env_client = GuiEnvClient(env_url)
    state = GenerateState(args)

    rollout_max_steps = getattr(args, "gui_max_steps", None)
    if rollout_max_steps is None:
        rollout_max_steps = int(os.getenv("GUI_MAX_STEPS", "15"))

    eval_max_steps = getattr(args, "gui_eval_max_steps", None)
    max_steps = int(eval_max_steps if evaluation and eval_max_steps is not None else rollout_max_steps)

    rollout_sleep = getattr(args, "gui_sleep_after_execution", None)
    if rollout_sleep is None:
        rollout_sleep = float(os.getenv("GUI_SLEEP_AFTER_EXECUTION", "0.0"))

    eval_sleep = getattr(args, "gui_eval_sleep_after_execution", None)
    sleep_after_execution = float(eval_sleep if evaluation and eval_sleep is not None else rollout_sleep)

    rollout_wait_reset = getattr(args, "gui_wait_after_reset", None)
    if rollout_wait_reset is None:
        rollout_wait_reset = float(os.getenv("GUI_WAIT_AFTER_RESET", "0.0"))
    eval_wait_reset = getattr(args, "gui_eval_wait_after_reset", None)
    wait_after_reset = float(eval_wait_reset if evaluation and eval_wait_reset is not None else rollout_wait_reset)

    max_image_history_cfg = getattr(args, "gui_max_image_history_length", None)
    if max_image_history_cfg is None:
        max_image_history_cfg = int(os.getenv("GUI_MAX_IMAGE_HISTORY_LENGTH", str(max_steps)))
    max_image_history_length = int(max_image_history_cfg)

    sampling_params = dict(sampling_params)
    if evaluation and getattr(args, "eval_temperature", None) is not None:
        sampling_params["temperature"] = float(args.eval_temperature)
    elif (not evaluation) and getattr(args, "rollout_temperature", None) is not None:
        sampling_params["temperature"] = float(args.rollout_temperature)

    parser = _create_gui_agent(
        args,
        max_steps=max_steps,
        max_image_history_length=max_image_history_length,
        result_dir=result_dir,
    )
    parser.reset(logging.getLogger("desktopenv.gui_agent.rollout"))

    assistant_responses: list[str] = []
    step_snapshots: list[dict[str, Any]] = []
    fallback_train_messages: list[dict[str, Any]] = [parser.build_train_system_message()]
    fallback_tool_spec: dict[str, Any] | None = None
    last_step_train_messages_for_loss: list[dict[str, Any]] | None = None
    last_step_tool_spec_for_loss: dict[str, Any] | None = None
    trace_records: list[dict[str, Any]] = []
    prm_agent: Any | None = None
    prm_pending_tasks: list[tuple[int, asyncio.Task]] = []
    prm_step_scores: list[float] = []
    prm_step_details: list[dict[str, Any]] = []
    final_status = Sample.Status.COMPLETED
    eval_score = 0.0

    trajectory_semaphore = _get_gui_trajectory_semaphore()
    await trajectory_semaphore.acquire()
    lease_id: str | None = None
    try:
        _gui_log(
            "GUI rollout start sample=%s group=%s domain=%s example=%s eval=%s",
            sample.index,
            sample.group_index,
            domain,
            example_id,
            evaluation,
        )
        # GUI env capacity is usually the bottleneck. Retry allocation so
        # temporary pool exhaustion does not hard-crash rollout workers.
        allocate_retries = int(os.getenv("GUI_ALLOCATE_RETRIES", "10"))
        allocate_backoff_seconds = float(os.getenv("GUI_ALLOCATE_BACKOFF_SECONDS", "2.0"))
        allocate_error: Exception | None = None
        for attempt in range(allocate_retries):
            try:
                lease = await env_client.allocate(episode_id=f"{domain}:{example_id}:{uuid.uuid4().hex[:8]}")
                lease_id = lease["lease_id"]
                allocate_error = None
                break
            except Exception as e:  # pragma: no cover - external service call
                allocate_error = e
                if attempt < allocate_retries - 1:
                    logger.warning(
                        "GUI allocate failed (%d/%d), will retry in %.1fs: %s",
                        attempt + 1,
                        allocate_retries,
                        allocate_backoff_seconds,
                        e,
                    )
                    await asyncio.sleep(allocate_backoff_seconds)
        if lease_id is None:
            raise RuntimeError(
                f"GUI env allocate failed after {allocate_retries} retries from {env_url}: {allocate_error}"
            )

        obs = await env_client.reset(lease_id=lease_id, task_config=task_config)
        await env_client.start_recording(lease_id)
        if wait_after_reset > 0:
            _gui_log("wait_after_reset sample=%s seconds=%.1f", sample.index, wait_after_reset)
            await asyncio.sleep(wait_after_reset)
            obs = await env_client.get_obs(lease_id)
        _save_image(obs["screenshot"], result_dir / "step_0.png")
        _gui_log("lease ready sample=%s lease_id=%s", sample.index, lease_id)
        if getattr(args, "prm_enable", False):
            prm_max_hist = getattr(args, "gui_max_reward_image_history_length", None)
            if prm_max_hist is None:
                prm_max_hist = int(os.getenv("GUI_MAX_REWARD_IMAGE_HISTORY_LENGTH", "2"))
            policy_model_path = str(getattr(args, "hf_checkpoint", "") or "")
            prm_model_path = str(getattr(args, "prm_model_path", "") or policy_model_path)
            share_formatter = policy_model_path == prm_model_path
            if not share_formatter:
                _gui_log(
                    "PRM formatter mismatch detected; use prm tokenizer/processor. policy_model=%s prm_model=%s",
                    policy_model_path,
                    prm_model_path,
                )
            reward_cls_path = getattr(args, "gui_reward_agent_class_path", None) or os.getenv("GUI_REWARD_AGENT_CLASS_PATH")
            if not reward_cls_path:
                raise RuntimeError("GUI_REWARD_AGENT_CLASS_PATH is required when prm_enable=True.")
            reward_cls = load_function(reward_cls_path)
            prm_agent = reward_cls(
                max_reward_image_history_length=int(prm_max_hist),
                example_result_dir=str(result_dir),
                tokenizer=(state.tokenizer if share_formatter else None),
                processor=(state.processor if share_formatter else None),
            )

        for step_idx in range(max_steps):
            await env_client.heartbeat(lease_id)
            parse_ctx = parser.build_policy_messages(instruction=instruction, obs=obs)
            policy_messages = parse_ctx["messages"]
            tool_spec = parse_ctx.get("tool_spec")
            fallback_train_messages = policy_messages
            fallback_tool_spec = tool_spec

            response, finish_type = await parser.generate_with_sglang(
                args=args,
                state=state,
                messages=policy_messages,
                sampling_params=sampling_params,
                sampling_seed=((int(sample.index or 0) + 1) * 1000003 + step_idx * 9973),
                tool_spec=tool_spec,
            )
            preview_chars = int(os.getenv("GUI_LOG_RESPONSE_PREVIEW_CHARS", "0"))
            rendered_response = response
            if preview_chars > 0 and len(response) > preview_chars:
                rendered_response = response[:preview_chars] + "...(truncated)"
            _gui_log(
                "step sample=%s step=%s seed=%s finish=%s response=%s",
                sample.index,
                step_idx,
                ((int(sample.index or 0) + 1) * 1000003 + step_idx * 9973),
                finish_type,
                rendered_response,
            )
            if finish_type == "abort":
                final_status = Sample.Status.ABORTED
                break
            step_train_messages = copy.deepcopy(policy_messages)
            # Dynamic-history training should only optimize the current step response.
            # Explicitly disable loss on historical assistant turns.
            for msg in step_train_messages:
                if msg.get("role") == "assistant":
                    msg["step_loss_mask"] = 0
            step_train_messages.append({"role": "assistant", "content": response, "step_loss_mask": 1})
            last_step_train_messages_for_loss = step_train_messages
            last_step_tool_spec_for_loss = tool_spec
            assistant_responses.append(response)
            step_snapshots.append(
                {
                    "step_idx": step_idx,
                    "train_messages": step_train_messages,
                    "response_text": response,
                    "tool_spec": tool_spec,
                }
            )

            original_width = int(parse_ctx["original_width"])
            original_height = int(parse_ctx["original_height"])
            processed_width = int(parse_ctx["processed_width"])
            processed_height = int(parse_ctx["processed_height"])

            natural_action, actions, info_dict = parser.parse_response(
                response=response,
                original_width=original_width,
                original_height=original_height,
                processed_width=processed_width,
                processed_height=processed_height,
            )
            parser.record_policy_turn(
                action_text=natural_action or "Execute action",
                response=response,
                screenshot_bytes=obs["screenshot"],
            )

            if not actions or actions[0] == "":
                final_status = Sample.Status.FAILED
                break

            if str(actions[0]).upper() in {"FAIL"}:
                final_status = Sample.Status.FAILED
                break

            step_executed = False
            done = False
            for action in actions:
                obs, reward, done, info = await env_client.step(
                    lease_id=lease_id, action=action, sleep_after_execution=sleep_after_execution
                )
                _gui_log(
                    "action sample=%s step=%s action=%s reward=%.4f done=%s",
                    sample.index,
                    step_idx,
                    action,
                    float(reward),
                    done,
                )
                step_executed = True
                step_image_path = result_dir / f"step_{step_idx + 1}.png"
                _save_image(obs["screenshot"], step_image_path)
                with open(traj_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "step_num": step_idx + 1,
                                "action": action,
                                "natural_language_action": info_dict.get("action"),
                                "response": response,
                                "reward": reward,
                                "done": done,
                                "info": info,
                                "screenshot_file": step_image_path.name,
                            },
                            ensure_ascii=False,
                        )
                    )
                    f.write("\n")
                if done:
                    final_status = Sample.Status.COMPLETED
                    break

            trace_records.append(
                {
                    "messages": policy_messages,
                    "response": response,
                    "actions": actions,
                    "step_idx": step_idx,
                    "step_executed": step_executed,
                }
            )

            # PRM enqueue point:
            # Immediately after policy action is executed and next observation is ready,
            # dispatch reward-model judging task asynchronously (do not wait here).
            if prm_agent is not None and step_executed:
                prm_pending_tasks.append(
                    (
                        step_idx,
                        prm_agent.submit_step_judge(
                            args,
                            instruction=instruction,
                            actions_history=list(parser.actions),
                            policy_response=response,
                            step_index=step_idx,
                        ),
                    )
                )

            if done:
                break
        else:
            final_status = Sample.Status.TRUNCATED

        if prm_agent is not None and prm_pending_tasks:
            prm_step_scores, prm_step_details = await prm_agent.collect_step_results(prm_pending_tasks)

        if final_status != Sample.Status.COMPLETED:
            try:
                await env_client.step(lease_id=lease_id, action="FAIL", sleep_after_execution=0)
            except Exception:
                logger.debug("Failed to send terminal FAIL action", exc_info=True)

        eval_score = await env_client.evaluate(lease_id=lease_id)
        _gui_log("rollout end sample=%s status=%s score=%.4f", sample.index, final_status.value, eval_score)
        with open(result_dir / "result.txt", "w", encoding="utf-8") as f:
            f.write(f"{eval_score}\n")
        with open(result_dir / "trajectory.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "meta": {"result": eval_score},
                    "trajectory": trace_records,
                    "reward_trajectory": (prm_agent.reward_trajectory if prm_agent is not None else []),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        try:
            await env_client.end_recording(lease_id, out_path=str(result_dir / "recording.mp4"))
        except Exception:
            logger.exception("end_recording failed for %s", result_dir)
    except Exception as e:
        final_status = Sample.Status.ABORTED
        tb = traceback.format_exc()
        with open(traj_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"Error": str(e), "Traceback": tb}, ensure_ascii=False) + "\n")
        print(
            f"[GUI_ROLLOUT_ERROR] sample_index={sample.index} domain={domain} example_id={example_id}\n{tb}",
            file=sys.stderr,
            flush=True,
        )
        logger.exception("GUI rollout failed for sample %s", sample.index)
    finally:
        if lease_id is not None:
            try:
                await env_client.close(lease_id=lease_id)
            except Exception:
                logger.exception("Failed to close env lease: %s", lease_id)
        trajectory_semaphore.release()

    train_messages_for_loss = (
        last_step_train_messages_for_loss if last_step_train_messages_for_loss else fallback_train_messages
    )
    tool_spec_for_loss = last_step_tool_spec_for_loss if last_step_tool_spec_for_loss else fallback_tool_spec
    input_ids, loss_mask, mm_train = parser.build_train_data(
        args=args,
        state=state,
        train_messages=train_messages_for_loss,
        tool_spec=tool_spec_for_loss,
    )
    response_start = None
    active_positions = [i for i in range(len(loss_mask)) if i < len(input_ids) and int(loss_mask[i]) == 1]
    if active_positions:
        response_start = active_positions[0]
        response_length = len(input_ids) - response_start
        loss_mask = [int(loss_mask[i]) if i < len(loss_mask) else 0 for i in range(response_start, len(input_ids))]
    else:
        response_length = 0
        loss_mask = []

    sample.tokens = input_ids
    sample.loss_mask = loss_mask
    sample.response = "\n".join(assistant_responses)
    sample.response_length = response_length
    sample.multimodal_train_inputs = mm_train
    sample.status = final_status
    sample.metadata = sample.metadata or {}
    sample.metadata["gui_result_dir"] = str(result_dir)
    sample.metadata["gui_score"] = eval_score
    if getattr(args, "prm_enable", False):
        sample.metadata["prm"] = {
            "enabled": True,
            "step_scores": prm_step_scores,
            "step_mean_score": (sum(prm_step_scores) / len(prm_step_scores)) if prm_step_scores else 0.0,
            "step_details": prm_step_details,
        }
        # Current GUI non-dynamic path trains the suffix of one step response;
        # align step_wise metadata to that suffix span.
        if response_start is not None and response_length > 0:
            last_step_idx = int(step_snapshots[-1]["step_idx"]) if step_snapshots else 0
            prm_score_by_step = {int(d.get("step_index", i)): float(d.get("mean_score", 0.0)) for i, d in enumerate(prm_step_details)}
            sample.metadata["step_wise"] = {
                "step_scores": [float(prm_score_by_step.get(last_step_idx, 0.0))],
                "step_indices": [int(last_step_idx)],
                "step_token_spans": [[0, int(response_length)]],
            }
    gui_reward = 1.0 if eval_score == 1 else -1.0
    # Keep training reward on `score` (GRPO expects this),
    # and expose raw task accuracy on `acc` for eval logging.
    # Important:
    # - PRM path must go through reward_func so step-wise PRM composition
    #   and prm_example_eval are visible in rollout logs.
    # - Non-PRM path keeps old behavior with prefilled reward.
    if getattr(args, "prm_enable", False):
        sample.reward = None
    else:
        sample.reward = {"score": gui_reward, "acc": float(eval_score)}
    if getattr(args, "dynamic_history", False) and not evaluation:
        prm_score_by_step = None
        if getattr(args, "prm_enable", False) and isinstance(sample.metadata.get("prm"), dict):
            prm_score_by_step = {}
            for item in sample.metadata["prm"].get("step_details", []):
                if isinstance(item, dict) and "step_index" in item:
                    prm_score_by_step[int(item["step_index"])] = float(item.get("mean_score", 0.0))
        dynamic_samples = _build_dynamic_history_samples(
            args=args,
            state=state,
            agent=parser,
            base_sample=sample,
            step_snapshots=step_snapshots,
            outcome_reward=gui_reward,
            prm_score_by_step=prm_score_by_step,
        )
        result = dynamic_samples if dynamic_samples else [sample]
        _mark_aborted_samples(result)
        return result
    _mark_aborted_samples([sample])
    return sample


def _mark_aborted_samples(samples: list[Sample]) -> None:
    """Give ABORTED samples a default reward and exclude them from training.

    The rollout pipeline skips reward_func for lists containing any ABORTED
    sample, which leaves reward=None and crashes downstream metrics.  Setting a
    default reward + remove_sample=True keeps the pipeline stable while
    ensuring these broken trajectories never contribute to the gradient.
    """
    for s in samples:
        if s.status == Sample.Status.ABORTED:
            if s.reward is None:
                s.reward = {"score": 0.0, "acc": 0.0}
            s.remove_sample = True


def _single_reward(sample: Sample) -> dict[str, float]:
    outcome_score = 0.0
    raw_acc = 0.0
    if isinstance(sample.reward, dict):
        outcome_score = float(sample.reward.get("score", 0.0))
        raw_acc = float(sample.reward.get("acc", 0.0))
    elif isinstance(sample.metadata, dict):
        raw_acc = float(sample.metadata.get("gui_score", 0.0))
        outcome_score = 1.0 if raw_acc == 1.0 else -1.0
    result = {"score": outcome_score, "acc": raw_acc}
    return result


async def reward_func(args, sample: Sample | list[Sample], **kwargs):
    def _compose_with_prm(s: Sample) -> dict[str, float]:
        result = _single_reward(s)
        # Keep non-PRM behavior unchanged.
        if not getattr(args, "prm_enable", False):
            return result

        prm_metadata = s.metadata.get("prm", {}) if isinstance(s.metadata, dict) else {}
        if not isinstance(prm_metadata, dict):
            prm_metadata = {}

        prm_step_mean = float(prm_metadata.get("step_mean_score", 0.0))
        outcome_reward = float(result.get("score", 0.0))
        final_score = outcome_reward + float(getattr(args, "prm_step_coef", 1.0)) * prm_step_mean
        result["base_score"] = outcome_reward
        result["prm_step_score"] = prm_step_mean
        result["score"] = final_score
        # Align with retool: expose one concrete PRM raw output sample for quick sanity-check.
        prm_example_eval = ""
        step_details = prm_metadata.get("step_details", [])
        if isinstance(step_details, list) and step_details:
            first_step = step_details[0] if isinstance(step_details[0], dict) else {}
            votes = first_step.get("votes", []) if isinstance(first_step, dict) else []
            if isinstance(votes, list) and votes:
                first_vote = votes[0] if isinstance(votes[0], dict) else {}
                raw_text = first_vote.get("raw_text", "") if isinstance(first_vote, dict) else ""
                if isinstance(raw_text, str):
                    prm_example_eval = raw_text
        result["prm_example_eval"] = prm_example_eval

        # Populate step_wise composed rewards for step_wise advantage.
        if isinstance(s.metadata, dict):
            step_wise_meta = s.metadata.get("step_wise", {})
            if not isinstance(step_wise_meta, dict):
                step_wise_meta = {}
            step_wise_meta["outcome_reward"] = outcome_reward
            raw_step_scores = step_wise_meta.get("step_scores", [])
            if isinstance(raw_step_scores, list):
                step_wise_meta["step_scores_with_outcome"] = [
                    float(step_score) + float(outcome_reward) for step_score in raw_step_scores
                ]
            else:
                step_wise_meta["step_scores_with_outcome"] = []
            s.metadata["step_wise"] = step_wise_meta
        return result

    if isinstance(sample, list):
        return [_compose_with_prm(s) for s in sample]
    return _compose_with_prm(sample)


def gui_generate_rollout(args, rollout_id: int, data_source, evaluation: bool = False):
    # Train path keeps the default rollout behavior.
    if not evaluation:
        from slime.rollout.sglang_rollout import generate_rollout

        return generate_rollout(args, rollout_id, data_source, evaluation=False)

    from slime.utils.async_utils import run

    output, _ = run(_gui_eval_rollout(args))
    return output


async def _gui_eval_rollout(args):
    from slime.rollout.base_types import RolloutFnEvalOutput
    from slime.rollout.sglang_rollout import generate_and_rm

    base_dir = os.getenv(
        "GUI_TEST_CONFIG_BASE_DIR",
        str(Path(__file__).resolve().parent / "evaluation_examples"),
    )
    meta_path = os.getenv("GUI_EVAL_META_PATH", str(Path(base_dir) / "test_nochrome.json"))
    pairs = _load_meta_pairs(meta_path)
    if not pairs:
        raise RuntimeError(f"No eval tasks loaded from {meta_path}")

    eval_max_response_len = getattr(args, "eval_max_response_len", None)
    if eval_max_response_len is None:
        eval_max_response_len = getattr(args, "rollout_max_response_len", 512)

    eval_top_p = getattr(args, "eval_top_p", None)
    if eval_top_p is None:
        eval_top_p = getattr(args, "rollout_top_p", 1.0)
    eval_top_k = getattr(args, "eval_top_k", None)
    if eval_top_k is None:
        eval_top_k = getattr(args, "rollout_top_k", -1)

    sampling_params = dict(
        temperature=float(getattr(args, "eval_temperature", 0.0) or 0.0),
        top_p=float(eval_top_p),
        top_k=int(eval_top_k),
        max_new_tokens=int(eval_max_response_len),
        stop=getattr(args, "rollout_stop", None),
        stop_token_ids=getattr(args, "rollout_stop_token_ids", None),
        skip_special_tokens=getattr(args, "rollout_skip_special_tokens", False),
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    n_samples = int(getattr(args, "n_samples_per_eval_prompt", 1) or 1)
    tasks = []
    sample_index = 0
    for domain, example_id in pairs:
        cfg_path = Path(base_dir) / "examples" / str(domain) / f"{example_id}.json"
        if not cfg_path.exists():
            continue
        task_config = _load_task_config(base_dir, domain, example_id)
        instruction = str(task_config.get("instruction", ""))
        group_index = sample_index // n_samples
        for _ in range(n_samples):
            sample = Sample(
                prompt=instruction,
                label="",
                metadata={
                    "domain": domain,
                    "example_id": example_id,
                    "instruction": instruction,
                    "task_config": task_config,
                },
            )
            sample.index = sample_index
            sample.group_index = group_index
            sample_index += 1
            tasks.append(
                asyncio.create_task(
                    generate_and_rm(args, sample, sampling_params=sampling_params, evaluation=True)
                )
            )

    if not tasks:
        raise RuntimeError(f"No valid eval tasks found from {meta_path}")

    data = []
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)

    data.sort(key=lambda s: s.index)
    reward_key = getattr(args, "eval_reward_key", None) or getattr(args, "reward_key", "score")
    rewards = []
    for sample in data:
        if isinstance(sample.reward, dict):
            rewards.append(float(sample.reward.get(reward_key, 0.0)))
        elif sample.reward is not None:
            rewards.append(float(sample.reward))
        else:
            rewards.append(0.0)

    return RolloutFnEvalOutput(
        data={
            "gui_eval": {
                "rewards": rewards,
                "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
                "samples": data,
            }
        }
    ), []
