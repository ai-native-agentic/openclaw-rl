# Skill Bridge

`skill-bridge` adds a `skill_manage` agent tool backed by a bundled Python skill manager.

The plugin is aimed at OpenClaw setups that want writable, file-backed skill authoring from the agent runtime without depending on a separate external package layout.

## Behavior

- Registers a `skill_manage` tool for creating, patching, editing, and deleting `SKILL.md`-based skills.
- Stores skills in `<workspace>/skills` by default.
- Adds lightweight prompt guidance and periodic nudges for memory and skill persistence.
- Enqueues a flush reminder before compaction/reset unless disabled.

## Config

Configure under `plugins.entries.skill-bridge.config`:

```json
{
  "pythonPath": "python3",
  "skillsDir": "/absolute/path/to/skills",
  "memoryNudgeInterval": 8,
  "skillNudgeInterval": 15,
  "enqueueFlushReminder": true
}
```

Notes:

- `pythonPath` defaults to `python3`.
- `skillsDir` is optional. If omitted, the plugin writes to the current workspace `skills/` directory.
- If both `skillsDir` and `workspaceDir` are unavailable, tool execution fails fast.

## Development

The Python implementation is bundled under [`python/`](./python) so the plugin does not depend on repository-root Python modules or custom `sys.path` setup.
