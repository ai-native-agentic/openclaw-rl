from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .manager import SkillManager, SkillManagerError

OPENAI_SKILL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "skill_manage",
        "description": (
            "Manage reusable workflow skills stored on disk. Use create for a new "
            "SKILL.md, patch for targeted fixes, edit for full rewrites, delete to "
            "remove a skill, and write_file/remove_file for references, templates, "
            "scripts, and assets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "patch", "edit", "delete", "write_file", "remove_file"],
                },
                "name": {"type": "string"},
                "content": {"type": "string"},
                "category": {"type": "string"},
                "file_path": {"type": "string"},
                "file_content": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["action", "name"],
        },
    },
}


class SkillToolAdapter:
    def __init__(self, manager: SkillManager, tool_name: str = "skill_manage"):
        self.manager = manager
        self.tool_name = tool_name

    def schema(self) -> dict[str, Any]:
        schema = dict(OPENAI_SKILL_TOOL_SCHEMA)
        schema["function"] = dict(schema["function"])
        schema["function"]["name"] = self.tool_name
        return schema

    def dispatch(self, arguments: Mapping[str, Any]) -> str:
        action = arguments.get("action")
        name = arguments.get("name")

        try:
            if action == "create":
                result = self.manager.create(
                    name=name,
                    content=arguments.get("content") or "",
                    category=arguments.get("category"),
                )
            elif action == "edit":
                result = self.manager.edit(name=name, content=arguments.get("content") or "")
            elif action == "patch":
                result = self.manager.patch(
                    name=name,
                    old_string=arguments.get("old_string") or "",
                    new_string=arguments.get("new_string"),
                    file_path=arguments.get("file_path"),
                    replace_all=bool(arguments.get("replace_all", False)),
                )
            elif action == "delete":
                result = self.manager.delete(name=name)
            elif action == "write_file":
                result = self.manager.write_file(
                    name=name,
                    file_path=arguments.get("file_path") or "",
                    file_content=arguments.get("file_content", ""),
                )
            elif action == "remove_file":
                result = self.manager.remove_file(
                    name=name,
                    file_path=arguments.get("file_path") or "",
                )
            else:
                return json.dumps(
                    {"success": False, "error": f"Unknown action '{action}'."},
                    ensure_ascii=False,
                )
        except SkillManagerError as exc:
            return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)

        return json.dumps(result.as_dict(), ensure_ascii=False)
