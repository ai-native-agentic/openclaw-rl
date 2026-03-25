# RL Training Headers

OpenClaw plugin that injects **`X-Session-Id`** and **`X-Turn-Type`** HTTP headers into every outgoing LLM API request, enabling downstream RL training pipelines to classify and segment training data.

## Headers

| Header | Value | Description |
|---|---|---|
| `X-Session-Id` | `<uuid>` | The active agent session identifier |
| `X-Turn-Type` | `main` | User-initiated conversation turn |
| `X-Turn-Type` | `side` | Housekeeping turn (heartbeat, memory flush, cron) |

## Install

1. Copy (or symlink) this folder into `extensions/rl-training-headers` inside your OpenClaw repo.

2. Enable the plugin:

```bash
corepack pnpm start -- plugins enable rl-training-headers
```

3. Restart the gateway:

```bash
corepack pnpm start -- gateway restart
```

That's it. Every LLM API request now carries the two extra headers.

## Configuration (optional)

You can customize the header names in `~/.openclaw/openclaw.json` under `plugins.entries`:

```json
{
  "plugins": {
    "entries": {
      "rl-training-headers": {
        "enabled": true,
        "config": {
          "sessionIdHeader": "X-Session-Id",
          "turnTypeHeader": "X-Turn-Type"
        }
      }
    }
  }
}
```

## How it works

The plugin hooks into the `before_prompt_build` lifecycle event to capture the current session ID and turn type (derived from the `trigger` field: `"user"` → `main`, `"heartbeat"` / `"memory"` / `"cron"` → `side`).

It then patches `globalThis.fetch` to inject these headers into all outgoing POST requests during an active agent run. Headers are cleared after the agent run completes (`agent_end`).

## Extracting training data

On the API proxy / logging side, use these headers to:

- **Group** requests by `X-Session-Id` to reconstruct full conversation sessions.
- **Filter** by `X-Turn-Type: main` to keep only user-facing dialogue for reward modelling, or include `side` turns for full trajectory analysis.
