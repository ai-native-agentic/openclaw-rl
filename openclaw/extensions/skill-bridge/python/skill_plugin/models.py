from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SkillInfo:
    name: str
    path: Path
    skill_md: Path
    category: str | None = None
    description: str | None = None
    frontmatter: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillOperationResult:
    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {"success": self.success, "message": self.message}
        payload.update(self.data)
        return payload
