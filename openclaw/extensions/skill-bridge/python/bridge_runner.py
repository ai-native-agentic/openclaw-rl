from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("Expected JSON arguments as argv[1].")

    python_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(python_root))

    from skill_plugin import SkillManager, SkillToolAdapter

    skills_dir = os.environ["SKILL_PLUGIN_SKILLS_DIR"]
    tool_name = os.environ.get("SKILL_PLUGIN_TOOL_NAME", "skill_manage")
    arguments = json.loads(sys.argv[1])
    adapter = SkillToolAdapter(SkillManager(skills_dir), tool_name=tool_name)
    print(adapter.dispatch(arguments))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
