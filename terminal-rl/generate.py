from __future__ import annotations

import logging
import os
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional
import asyncio

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from agent.prm_agent import TerminalPRMAgent
from custom_types import (
    Interaction,
    RunContext,
    TaskSpec,
    TaskTimeouts,
    TurnContext,
    TurnResult,
)
from inference_client import SGLangTurnClient
from agent_runner import create_agent_runner
from env_client import TerminalEnvClient

logger = logging.getLogger(__name__)


def _extract_task_meta(sample: Sample) -> Dict[str, Any]:
    if isinstance(sample.prompt, dict):
        return sample.prompt

    metadata = sample.metadata or {}
    task_meta = metadata.get("task_meta") if isinstance(metadata, dict) else None
    if isinstance(task_meta, dict):
        return task_meta

    if isinstance(metadata, dict):
        return metadata

    return {}


def _make_task_spec(meta: Dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        task_name=meta.get("task_name", "unknown"),
        task_path=meta.get("task_path", ""),
        instruction=meta.get("instruction", ""),
    )


def _build_samples(
    interactions: List[Interaction],
    base_sample: Sample,
    outcome: float,
    status: Sample.Status,
    prm_turn_scores: dict[int, float] | None = None,
    prm_coef: float = 1.0,
    discount: float = 1.0,
    encourage: bool = False,
) -> List[Sample]:
    """Create one Sample per interaction with discounted reward."""
    num_turns = len(interactions)
    samples: List[Sample] = []

    accuracy = float(outcome)
    raw_score = accuracy + (accuracy == 1.0) * int(encourage)
    base_outcome = 2.0 * accuracy - 1.0

    for interaction in interactions:
        turn_idx = interaction.turn_idx
        s = deepcopy(base_sample)
        s.tokens = interaction.input_ids + interaction.output_token_ids
        s.response_length = len(interaction.output_token_ids)
        s.loss_mask = [1] * s.response_length
        s.rollout_log_probs = list(interaction.output_token_logprobs)
        s.response = interaction.output_text
        s.status = status

        s.metadata.update(
            {
                "turn_idx": turn_idx,
                "num_turns": num_turns,
                "finish_reason": interaction.finish_reason,
                "latency_ms": interaction.latency_ms,
            }
        )

        steps_from_end = num_turns - 1 - turn_idx
        discounted_base = base_outcome * (discount**steps_from_end)

        if prm_turn_scores is not None:
            prm = prm_turn_scores.get(turn_idx, 0.0)
            final = discounted_base + prm_coef * prm
            s.metadata["step_wise"] = {
                "step_scores": [prm],
                "step_scores_with_outcome": [final],
                "step_indices": [turn_idx],
                "step_token_spans": [[0, s.response_length]],
            }
        else:
            final = discounted_base

        s.reward = {
            "accuracy": accuracy,
            "raw_score": raw_score,
            "base_score": discounted_base,
            "score": final,
        }

        if prm_turn_scores is not None:
            s.reward["prm_turn_score"] = prm
        samples.append(s)

    return samples


def _mark_non_trainable_samples(samples: List[Sample]) -> None:
    for sample in samples:
        if sample.status in {Sample.Status.ABORTED, Sample.Status.FAILED}:
            if sample.reward is None:
                sample.reward = {"score": 0.0}
            sample.remove_sample = True


def _infer_completion_budget(sampling_params: Dict[str, Any]) -> int:
    for key in ("max_new_tokens", "max_tokens", "max_completion_tokens"):
        raw_value = sampling_params.get(key)
        if raw_value is None:
            continue
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return 0


def _normalize_tool_schemas(raw_tools: List[Any]) -> List[Dict[str, Any]]:
    schemas: List[Dict[str, Any]] = []
    for tool in raw_tools:
        if hasattr(tool, "get_openai_tool_schema") and callable(
            tool.get_openai_tool_schema
        ):
            schemas.append(tool.get_openai_tool_schema())
        elif isinstance(tool, dict):
            schemas.append(tool)
        else:
            raise TypeError(f"Unsupported tool schema object type: {type(tool)!r}")
    return schemas


async def _create_env_client(
    task_spec: TaskSpec,
    run_ctx: RunContext,
) -> tuple[TerminalEnvClient, str]:
    env_server_url = os.getenv("ENV_SERVER_URL", "")
    if not env_server_url:
        raise RuntimeError("ENV_SERVER_URL is empty.")

    env_client = TerminalEnvClient(env_server_url)
    task_key = f"{task_spec.task_name}:{task_spec.task_path}"
    request_id = (
        f"{task_key}:{run_ctx.uid}:{run_ctx.group_index}:{run_ctx.sample_index}"
    )
    lease = await env_client.allocate(task_key=task_key, request_id=request_id)
    lease_id = str(lease["lease_id"])
    logger.info(
        "Using remote terminal env backend lease=%s server=%s", lease_id, env_server_url
    )
    return env_client, lease_id


def _create_sglang_client(
    args: Any,
    tokenizer: Any,
    sampling_params: Dict[str, Any],
    max_total_tokens: int,
    enable_sglang_non_think: bool,
    *,
    sglang_url: str | None = None,
    max_retries: int = 30,
) -> SGLangTurnClient:
    if not sglang_url:
        sglang_url = (
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        )
    client_template_kwargs = {
        "chat_template_type": getattr(args, "chat_template_type", "hf"),
        "chat_template_kwargs": getattr(args, "chat_template_kwargs", None),
        "messages_delimiter_start": getattr(
            args, "messages_delimiter_start", "<|im_start|>"
        ),
        "messages_delimiter_end": getattr(args, "messages_delimiter_end", "<|im_end|>"),
        "tool_call_parser": getattr(args, "tool_call_parser", "qwen25"),
    }
    if enable_sglang_non_think:
        raw_chat_template_kwargs = client_template_kwargs.get("chat_template_kwargs")
        if isinstance(raw_chat_template_kwargs, dict):
            merged_chat_template_kwargs = dict(raw_chat_template_kwargs)
        else:
            merged_chat_template_kwargs = {}
        merged_chat_template_kwargs["enable_thinking"] = False
        client_template_kwargs["chat_template_kwargs"] = merged_chat_template_kwargs

    completion_budget = _infer_completion_budget(sampling_params)
    effective_context_limit = max_total_tokens
    for maybe_cap in (
        getattr(args, "rollout_max_context_len", None),
        getattr(args, "sglang_max_context_len", None),
    ):
        try:
            parsed_cap = int(maybe_cap)
        except (TypeError, ValueError):
            continue
        if parsed_cap > 0:
            effective_context_limit = min(effective_context_limit, parsed_cap)
    max_input_tokens = max(1, effective_context_limit - completion_budget)
    logger.info(
        "SGLang client: url=%s context_limit=%d, completion_budget=%d, max_input_tokens=%d",
        sglang_url,
        effective_context_limit,
        completion_budget,
        max_input_tokens,
    )
    raw_request_timeout = getattr(args, "sglang_request_timeout", None)
    if raw_request_timeout in (None, "", 0, 0.0):
        raw_request_timeout = os.getenv("SGLANG_REQUEST_TIMEOUT")
    try:
        request_timeout = (
            float(raw_request_timeout) if raw_request_timeout is not None else None
        )
    except (TypeError, ValueError):
        request_timeout = None
    if request_timeout is not None and request_timeout <= 0:
        request_timeout = None

    return SGLangTurnClient(
        model_type=None,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        url=sglang_url,
        session_id=None,
        max_input_tokens=max_input_tokens,
        request_timeout=request_timeout,
        max_retries=max_retries,
        **client_template_kwargs,
    )


async def generate(
    args,
    sample: Sample,
    sampling_params: Dict[str, Any],
    evaluation: bool = False,
) -> List[Sample]:
    _ = evaluation
    state = GenerateState(args)

    task_meta = _extract_task_meta(sample)
    uid = (sample.metadata or {}).get("uid") or uuid.uuid4().hex[:8]
    group_index = int(sample.group_index) if sample.group_index is not None else -1
    sample_index = int(sample.index) if sample.index is not None else -1
    task_spec = _make_task_spec(task_meta)
    run_ctx = RunContext(
        uid=uid,
        group_index=group_index,
        sample_index=sample_index,
        log_dir=Path(getattr(args, "tbench_output_root", "build_outputs"))
        / "AgentRunner_Output",
    )
    run_ctx_payload = run_ctx.to_payload()
    timeouts = TaskTimeouts(
        ensure_image=getattr(args, "ensure_image_timeout", 300.0),
        reset_session=getattr(args, "reset_session_timeout", 300.0),
        close_session=getattr(args, "close_session_timeout", 60.0),
        eval=getattr(args, "eval_timeout", 600.0),
    )
    timeouts_payload = timeouts.to_payload()

    env_client: Optional[TerminalEnvClient] = None
    lease_id: Optional[str] = None

    prm_enable = bool(getattr(args, "prm_enable", False)) and (not evaluation)
    prm_coef = float(getattr(args, "prm_turn_coef", 1.0))
    prm_agent: TerminalPRMAgent | None = None
    prm_pending: list[tuple[int, asyncio.Task]] = []
    prm_turn_scores: dict[int, float] = {}
    prm_turn_details: list[dict[str, Any]] = []

    _log_tag = f"[task={task_spec.task_name} uid={run_ctx.uid} group_idx={run_ctx.group_index} sample_idx={run_ctx.sample_index}]"

    try:
        env_client, lease_id = await _create_env_client(task_spec, run_ctx)
        reset_payload = await env_client.reset(
            lease_id=lease_id,
            task_meta=task_meta,
            run_ctx=run_ctx_payload,
            task_timeouts=timeouts_payload,
        )
        user_msg = str(reset_payload.get("user_msg", ""))
        raw_tools = list(reset_payload.get("tool_schemas", []))
        logger.info("%s Start terminal rollout", _log_tag)

        tool_schemas = _normalize_tool_schemas(raw_tools)
        agent_type = str(getattr(args, "terminal_agent_type", "camel_agent"))
        model_type = str(getattr(args, "model_type", "slime-sglang"))
        non_think_mode = bool(getattr(args, "non_think_mode", True))
        non_think_mode_source = str(
            getattr(args, "non_think_mode_source", "prompt")
        ).lower()
        if non_think_mode_source not in {"prompt", "sglang", "both"}:
            non_think_mode_source = "prompt"
        enable_prompt_non_think = non_think_mode and non_think_mode_source in {
            "prompt",
            "both",
        }
        enable_sglang_non_think = non_think_mode and non_think_mode_source in {
            "sglang",
            "both",
        }

        terminal_max_iterations = max(1, int(getattr(args, "max_iteration", 10)))
        terminal_max_parse_errors = max(1, int(getattr(args, "max_parse_errors", 3)))
        max_total_tokens = int(getattr(args, "max_total_tokens", 32768))
        sglang_client = _create_sglang_client(
            args=args,
            tokenizer=state.tokenizer,
            sampling_params=sampling_params,
            max_total_tokens=max_total_tokens,
            enable_sglang_non_think=enable_sglang_non_think,
        )

        if prm_enable:
            prm_router_ip = getattr(args, "prm_router_ip", None)
            prm_router_port = getattr(args, "prm_router_port", None)
            if prm_router_ip and prm_router_port:
                prm_sglang_url = f"http://{prm_router_ip}:{prm_router_port}/generate"
            else:
                prm_sglang_url = getattr(args, "prm_sglang_url", None) or os.getenv(
                    "PRM_SGLANG_URL", ""
                )
            if not prm_sglang_url:
                raise RuntimeError(
                    "prm_enable=True but no PRM endpoint: set prm_router_ip/port, "
                    "prm_sglang_url, or PRM_SGLANG_URL env var."
                )
            prm_sampling_params = {
                "temperature": float(getattr(args, "prm_temperature", 0.0)),
                "max_new_tokens": int(getattr(args, "prm_max_new_tokens", 4096)),
            }
            prm_max_total_tokens = int(getattr(args, "prm_max_total_tokens", 16384))
            prm_sglang_client = _create_sglang_client(
                args=args,
                tokenizer=state.tokenizer,
                sampling_params=prm_sampling_params,
                max_total_tokens=prm_max_total_tokens,
                enable_sglang_non_think=True,
                sglang_url=prm_sglang_url,
                max_retries=10,
            )
            prm_agent = TerminalPRMAgent(
                sglang_client=prm_sglang_client,
                task_instruction=task_spec.instruction,
                history_mode=str(getattr(args, "prm_history_mode", "head_tail")),
            )
            logger.info(
                "%s PRM enabled: url=%s coef=%.3f", _log_tag, prm_sglang_url, prm_coef
            )

        agent_runner = create_agent_runner(
            agent_type=agent_type,
            sglang_client=sglang_client,
            model_type=model_type,
            tool_schemas=tool_schemas,
            non_think_mode=enable_prompt_non_think,
            max_total_tokens=max_total_tokens,
        )
        agent_runner.reset(user_msg)
        agent_runner.set_max_parse_errors(terminal_max_parse_errors)
        agent_runner.set_max_iterations(terminal_max_iterations)

        # Loop
        interactions: List[Interaction] = []
        final_model_response = None
        final_response = None
        reached_iteration_limit = False
        reached_parse_error_limit = False

        while True:
            context_result: TurnContext = await agent_runner.get_turn_context()
            if context_result.terminated_response is not None:
                logger.warning("%s Rollout pre-terminated before model turn.", _log_tag)
                final_response = context_result.terminated_response
                break
            if context_result.context_messages is None:
                logger.warning("%s Rollout context is empty; aborting loop.", _log_tag)
                break

            turn_state: TurnResult = await agent_runner.run_model_turn(
                context_result.context_messages
            )
            interaction = turn_state.interaction
            turn_idx = int(interaction.turn_idx)
            interactions.append(interaction)

            if prm_agent is not None:
                tool_calls_for_prm = [
                    {"tool_name": tc.tool_name, "args": tc.args}
                    for tc in (turn_state.tool_call_requests or [])
                ]
                prm_agent.record_model_turn(
                    turn_idx,
                    assistant_text=interaction.output_text or "",
                    tool_calls=tool_calls_for_prm or None,
                    parse_error_recorded=turn_state.parse_error_recorded,
                    finish_reason=interaction.finish_reason,
                )

            if turn_state.terminated_response is not None:
                logger.warning(
                    "%s Rollout terminated during model turn %d.", _log_tag, turn_idx
                )
                final_response = turn_state.terminated_response
                break
            if turn_state.model_response is None:
                logger.warning(
                    "%s Model turn %d returned empty model_response.",
                    _log_tag,
                    turn_idx,
                )
                break

            should_continue_loop = False
            if tool_call_requests := turn_state.tool_call_requests:
                logger.info(
                    "%s Turn %d: executing %d tool call(s).",
                    _log_tag,
                    turn_idx,
                    len(tool_call_requests),
                )
                for tool_call_request in tool_call_requests:
                    assert env_client is not None and lease_id is not None
                    await env_client.heartbeat(lease_id)
                    raw_result = await env_client.exec_tool(
                        lease_id,
                        tool_call_request.tool_name,
                        tool_call_request.args,
                    )
                    agent_runner.record_tool_result(tool_call_request, raw_result)
                    if prm_agent is not None:
                        prm_agent.record_tool_result(
                            turn_idx, tool_call_request, raw_result
                        )
                should_continue_loop = True

            if turn_state.parse_error_recorded:
                logger.warning(
                    "%s Turn %d: tool-call parse error.",
                    _log_tag,
                    turn_idx,
                )
                should_continue_loop = True

            if prm_agent is not None:
                task = asyncio.create_task(prm_agent.judge_turn(turn_idx))
                prm_pending.append((turn_idx, task))

            if should_continue_loop:
                if (
                    turn_state.parse_error_recorded
                    and agent_runner.reached_parse_error_limit()
                ):
                    logger.error(
                        "%s Max parse errors (%d) reached at turn %d.",
                        _log_tag,
                        agent_runner.max_parse_errors,
                        turn_idx,
                    )
                    reached_parse_error_limit = True
                    final_model_response = turn_state.model_response
                    break
                if agent_runner.reached_iteration_limit():
                    logger.warning(
                        "%s Max iterations (%d) reached.",
                        _log_tag,
                        agent_runner.max_iterations,
                    )
                    reached_iteration_limit = True
                    final_model_response = turn_state.model_response
                    break
                continue

            final_model_response = turn_state.model_response
            break

        if final_response is None and final_model_response is not None:
            final_response = agent_runner.finalize_response(final_model_response)

        if final_response is None:
            logger.error(
                "%s No final response produced; mark sample aborted.", _log_tag
            )
            sample.status = Sample.Status.ABORTED
            sample.remove_sample = True
            sample.reward = {"score": 0.0}
            return [sample]

        finish_reasons = final_response.info.get("termination_reasons", [])
        is_aborted = not final_response.msg

        if final_response.terminated and "max_tokens_exceeded" in finish_reasons:
            status = Sample.Status.TRUNCATED
        elif reached_iteration_limit:
            status = Sample.Status.TRUNCATED
        elif reached_parse_error_limit:
            status = Sample.Status.FAILED
        elif is_aborted:
            status = Sample.Status.ABORTED
        else:
            status = Sample.Status.COMPLETED
        logger.info(
            "%s Rollout finished: status=%s turns=%d parse_errors=%d",
            _log_tag,
            status,
            agent_runner.model_turn_count,
            agent_runner.parse_error_count,
        )

        # Evaluation
        reward = 0.0
        eval_error: str | None = None
        should_evaluate = (not is_aborted) and status != Sample.Status.FAILED
        if should_evaluate:
            try:
                assert env_client is not None and lease_id is not None
                await env_client.heartbeat(lease_id)
                raw_score = await env_client.evaluate(lease_id)
                reward = float(raw_score)
                logger.info("%s Evaluation reward=%.4f", _log_tag, reward)
            except Exception as exc:
                eval_error = f"{type(exc).__name__}: {exc}"
                status = Sample.Status.FAILED
                reward = 0.0
                logger.error(
                    "%s Evaluation failed, marking FAILED: %s",
                    _log_tag,
                    eval_error,
                )

        if not interactions:
            logger.warning("%s No interactions recorded; remove sample.", _log_tag)
            sample.status = status
            sample.remove_sample = True
            sample.reward = {"score": 0.0}
            return [sample]

        if prm_agent is not None and prm_pending:
            for turn_idx, prm_task in prm_pending:
                try:
                    output_text, score = await prm_task
                    prm_turn_scores[turn_idx] = float(score)
                    prm_turn_details.append(
                        {
                            "turn_idx": turn_idx,
                            "score": float(score),
                            "output_text": output_text,
                        }
                    )
                    logger.info(
                        "%s PRM judge turn %d score=%.4f, output_text=%s",
                        _log_tag,
                        turn_idx,
                        float(score),
                        output_text.replace("\n", ""),
                    )
                except Exception as exc:
                    logger.warning(
                        "%s PRM judge failed for turn %d (ignored): %s",
                        _log_tag,
                        turn_idx,
                        exc,
                    )
                    prm_turn_scores[turn_idx] = 0.0
                    prm_turn_details.append(
                        {"turn_idx": turn_idx, "score": 0.0, "error": str(exc)}
                    )

        if prm_agent is not None:
            sample.metadata["prm"] = {
                "enabled": True,
                "coef": prm_coef,
                "turn_scores": prm_turn_scores,
                "turn_details": prm_turn_details,
            }

        # Build training samples
        samples = _build_samples(
            interactions=interactions,
            base_sample=sample,
            outcome=reward,
            status=status,
            prm_turn_scores=(prm_turn_scores if prm_agent is not None else None),
            prm_coef=prm_coef,
            discount=1.0,
            encourage=False,
        )
        for s in samples:
            s.metadata["model_turn_count"] = agent_runner.model_turn_count
            s.metadata["parse_error_count"] = agent_runner.parse_error_count
            if eval_error is not None:
                s.metadata["evaluation_failed"] = True
                s.metadata["evaluation_error"] = eval_error
        _mark_non_trainable_samples(samples)
        return samples

    except Exception as exc:
        logger.error(
            "%s Generate failed (%s): %s",
            _log_tag,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        sample.status = Sample.Status.FAILED
        sample.remove_sample = True
        sample.reward = {"score": 0.0}

        eos = state.tokenizer.eos_token_id
        if eos is None:
            sample.tokens = []
            sample.response_length = 0
            sample.rollout_log_probs = []
            sample.loss_mask = []
        else:
            sample.tokens = [eos, eos]
            sample.response_length = 1
            sample.rollout_log_probs = [0.0]
            sample.loss_mask = [0]
        return [sample]

    finally:
        for _turn_idx, t in prm_pending:
            if not t.done():
                t.cancel()

        if env_client is not None and lease_id is not None:
            try:
                await env_client.close(lease_id)
            except Exception as exc:
                logger.debug(
                    "%s Best-effort remote close failed lease=%s: %s",
                    _log_tag,
                    lease_id,
                    exc,
                )
