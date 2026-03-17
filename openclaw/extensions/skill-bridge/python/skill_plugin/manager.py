from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .models import SkillInfo, SkillOperationResult

MAX_NAME_LENGTH = 64
VALID_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
ALLOWED_SUBDIRS = {"references", "templates", "scripts", "assets"}


class SkillManagerError(ValueError):
    pass


class SkillManager:
    def __init__(self, skills_dir: str | os.PathLike[str]):
        self.skills_dir = Path(skills_dir).expanduser()

    def create(self, name: str, content: str, category: str | None = None) -> SkillOperationResult:
        self._validate_name(name)
        self._validate_frontmatter(content)
        existing = self.find(name)
        if existing:
            raise SkillManagerError(f"A skill named '{name}' already exists at {existing.path}.")

        skill_dir = self._resolve_skill_dir(name, category)
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md = skill_dir / "SKILL.md"
        self._atomic_write_text(skill_md, content)

        result = {
            "name": name,
            "path": str(skill_dir),
            "skill_md": str(skill_md),
        }
        if category:
            result["category"] = category
        return SkillOperationResult(True, f"Skill '{name}' created.", result)

    def edit(self, name: str, content: str) -> SkillOperationResult:
        self._validate_frontmatter(content)
        skill = self._require_skill(name)
        self._atomic_write_text(skill.skill_md, content)
        return SkillOperationResult(True, f"Skill '{name}' updated.", {"path": str(skill.path)})

    def patch(
        self,
        name: str,
        old_string: str,
        new_string: str,
        file_path: str | None = None,
        replace_all: bool = False,
    ) -> SkillOperationResult:
        if not old_string:
            raise SkillManagerError("old_string is required for patch().")
        if new_string is None:
            raise SkillManagerError("new_string is required for patch(). Use an empty string to delete text.")

        skill = self._require_skill(name)
        target = self._resolve_target_file(skill, file_path)
        if not target.exists():
            raise SkillManagerError(f"File not found: {target.relative_to(skill.path)}")

        content = target.read_text(encoding="utf-8")
        count = content.count(old_string)
        if count == 0:
            preview = content[:500] + ("..." if len(content) > 500 else "")
            raise SkillManagerError(f"old_string not found in file. Preview:\n{preview}")
        if count > 1 and not replace_all:
            raise SkillManagerError(
                f"old_string matched {count} times. Provide more context or set replace_all=True."
            )

        new_content = (
            content.replace(old_string, new_string)
            if replace_all
            else content.replace(old_string, new_string, 1)
        )
        if file_path is None:
            self._validate_frontmatter(new_content)

        self._atomic_write_text(target, new_content)
        replacements = count if replace_all else 1
        target_label = file_path or "SKILL.md"
        return SkillOperationResult(
            True,
            f"Patched {target_label} in skill '{name}' ({replacements} replacement{'s' if replacements != 1 else ''}).",
        )

    def delete(self, name: str) -> SkillOperationResult:
        skill = self._require_skill(name)
        shutil.rmtree(skill.path)

        parent = skill.path.parent
        if parent != self.skills_dir and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()

        return SkillOperationResult(True, f"Skill '{name}' deleted.")

    def write_file(self, name: str, file_path: str, file_content: str) -> SkillOperationResult:
        self._validate_file_path(file_path)
        skill = self._require_skill(name)
        target = skill.path / file_path
        self._atomic_write_text(target, file_content)
        return SkillOperationResult(
            True,
            f"File '{file_path}' written to skill '{name}'.",
            {"path": str(target)},
        )

    def remove_file(self, name: str, file_path: str) -> SkillOperationResult:
        self._validate_file_path(file_path)
        skill = self._require_skill(name)
        target = skill.path / file_path
        if not target.exists():
            raise SkillManagerError(f"File '{file_path}' not found in skill '{name}'.")

        target.unlink()
        parent = target.parent
        if parent != skill.path and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()

        return SkillOperationResult(True, f"File '{file_path}' removed from skill '{name}'.")

    def list_skills(self) -> list[SkillInfo]:
        if not self.skills_dir.exists():
            return []

        skills: list[SkillInfo] = []
        for skill_md in sorted(self.skills_dir.rglob("SKILL.md")):
            skill_dir = skill_md.parent
            relative = skill_dir.relative_to(self.skills_dir)
            category = relative.parts[0] if len(relative.parts) > 1 else None
            content = skill_md.read_text(encoding="utf-8")
            frontmatter = self.parse_frontmatter(content)
            skills.append(
                SkillInfo(
                    name=skill_dir.name,
                    path=skill_dir,
                    skill_md=skill_md,
                    category=category,
                    description=str(frontmatter.get("description", "")) or None,
                    frontmatter=frontmatter,
                )
            )
        return skills

    def find(self, name: str) -> SkillInfo | None:
        for skill in self.list_skills():
            if skill.name == name:
                return skill
        return None

    def _require_skill(self, name: str) -> SkillInfo:
        skill = self.find(name)
        if not skill:
            raise SkillManagerError(f"Skill '{name}' not found.")
        return skill

    def _resolve_skill_dir(self, name: str, category: str | None = None) -> Path:
        return self.skills_dir / category / name if category else self.skills_dir / name

    def _resolve_target_file(self, skill: SkillInfo, file_path: str | None) -> Path:
        if file_path is None:
            return skill.skill_md
        self._validate_file_path(file_path)
        return skill.path / file_path

    @staticmethod
    def parse_frontmatter(content: str) -> dict[str, Any]:
        start, end = SkillManager._split_frontmatter(content)
        payload: dict[str, Any] = {}
        for line in start.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if value.lower() == "true":
                payload[key] = True
            elif value.lower() == "false":
                payload[key] = False
            elif value.isdigit():
                payload[key] = int(value)
            else:
                payload[key] = value
        SkillManager._validate_required_frontmatter(payload)
        return payload

    @staticmethod
    def _validate_name(name: str) -> None:
        if not name:
            raise SkillManagerError("Skill name is required.")
        if len(name) > MAX_NAME_LENGTH:
            raise SkillManagerError(f"Skill name exceeds {MAX_NAME_LENGTH} characters.")
        if not VALID_NAME_RE.match(name):
            raise SkillManagerError(
                f"Invalid skill name '{name}'. Use lowercase letters, numbers, hyphens, dots, and underscores."
            )

    @staticmethod
    def _validate_frontmatter(content: str) -> None:
        frontmatter, _ = SkillManager._split_frontmatter(content)
        parsed = SkillManager.parse_frontmatter(f"---\n{frontmatter}\n---\n")
        SkillManager._validate_required_frontmatter(parsed)

    @staticmethod
    def _validate_required_frontmatter(parsed: dict[str, Any]) -> None:
        name = parsed.get("name")
        description = parsed.get("description")
        if not isinstance(name, str) or not name.strip():
            raise SkillManagerError("SKILL.md frontmatter requires a non-empty string 'name'.")
        if not isinstance(description, str) or not description.strip():
            raise SkillManagerError("SKILL.md frontmatter requires a non-empty string 'description'.")

    @staticmethod
    def _split_frontmatter(content: str) -> tuple[str, str]:
        if not content.startswith("---\n"):
            raise SkillManagerError("SKILL.md must start with YAML frontmatter delimited by --- lines.")
        marker = "\n---\n"
        end_index = content.find(marker, 4)
        if end_index < 0:
            raise SkillManagerError("SKILL.md frontmatter must end with a closing --- line.")
        return content[4:end_index], content[end_index + len(marker) :]

    @staticmethod
    def _validate_file_path(file_path: str) -> None:
        normalized = Path(file_path)
        if normalized.is_absolute():
            raise SkillManagerError("file_path must be relative to the skill directory.")
        if any(part == ".." for part in normalized.parts):
            raise SkillManagerError("file_path cannot traverse outside the skill directory.")
        if not normalized.parts:
            raise SkillManagerError("file_path is required.")
        if normalized.parts[0] not in ALLOWED_SUBDIRS:
            raise SkillManagerError(
                f"file_path must start with one of {sorted(ALLOWED_SUBDIRS)}."
            )

    @staticmethod
    def _atomic_write_text(target: Path, content: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=target.parent) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        tmp_path.replace(target)
