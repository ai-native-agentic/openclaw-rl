from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, List


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    task_path: str
    instruction: str


@dataclass(frozen=True)
class RunContext:
    uid: str
    group_index: int
    sample_index: int
    log_dir: Path

    def to_payload(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "group_index": self.group_index,
            "sample_index": self.sample_index,
            "log_dir": str(self.log_dir),
        }


@dataclass
class TaskTimeouts:
    ensure_image: float = 300.0
    reset_session: float = 300.0
    close_session: float = 60.0
    eval: float = 600.0

    def to_payload(self) -> dict[str, float]:
        return {
            "ensure_image": float(self.ensure_image),
            "reset_session": float(self.reset_session),
            "close_session": float(self.close_session),
            "eval": float(self.eval),
        }


from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam as OpenAIMessage,
)


@dataclass
class Interaction:
    turn_idx: int = 0
    completion: ChatCompletion | None = None
    input_ids: list[int] = field(default_factory=list)
    output_token_ids: list[int] = field(default_factory=list)
    output_token_logprobs: list[float] = field(default_factory=list)
    output_text: str = ""
    finish_reason: str = ""
    messages: list[OpenAIMessage] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class TurnContext:
    context_messages: Optional[List[dict[str, Any]]]
    terminated_response: Optional[Any] = None


@dataclass
class TurnResult:
    interaction: Interaction
    model_response: Optional[Any]
    tool_call_requests: List[Any]
    parse_error_recorded: bool
    terminated_response: Optional[Any] = None
