import path from "node:path";
import { spawnSync } from "node:child_process";
import type { AnyAgentTool, OpenClawPluginApi } from "openclaw/plugin-sdk";
import { jsonResult, readStringParam } from "openclaw/plugin-sdk";

const TOOL_NAME = "skill_manage";

const SKILL_POLICY_PROMPT =
  "Skills are reusable procedural memory. Save a skill after a complex or iterative task, " +
  "after recovering from non-trivial errors, when a user corrects the approach and the corrected " +
  "workflow works, or when the user explicitly asks you to remember a procedure. Prefer patching " +
  "an existing skill when the workflow is mostly right but stale or missing a pitfall. Update a " +
  "skill during use when instructions are stale, steps are missing, pitfalls are discovered, or " +
  "the workflow fails on a specific OS or environment. Use patch for targeted fixes, edit for " +
  "major rewrites, and write_file for supporting references, scripts, templates, or assets. " +
  `When acting, use \`${TOOL_NAME}\` to create, patch, edit, or delete skill files. Confirm before creating or deleting a skill.`;

const SKILL_CREATOR_STYLE_PROMPT =
  "When creating or rewriting a skill, keep the skill concise, include clear YAML frontmatter " +
  "with name and description, keep the SKILL.md body focused on essential procedural guidance, " +
  "avoid extra documentation files, and move detailed material into references/, scripts/, or " +
  "assets/ only when needed. Prefer patch for small fixes and edit for major rewrites.";

const MEMORY_NUDGE_TEXT =
  "[System: You've had several exchanges in this session. Consider whether there's anything worth saving to your memories.]";

const SKILL_NUDGE_TEXT =
  "[System: The previous task involved many steps. If you discovered a reusable workflow, consider saving it as a skill.]";

const FLUSH_REMINDER_TEXT =
  "[System: Context may be compacted or reset soon. Before it is discarded, save important facts to memory and capture reusable workflows as skills if they will help future sessions.]";

type SessionCounters = {
  turnsSinceMemory: number;
  turnsSinceSkill: number;
};

type SkillBridgeConfig = {
  pythonPath?: string;
  skillsDir?: string;
  memoryNudgeInterval?: number;
  skillNudgeInterval?: number;
  enqueueFlushReminder?: boolean;
};

const sessionState = new Map<string, SessionCounters>();

function getSessionState(sessionId?: string, sessionKey?: string): SessionCounters {
  const key = sessionId || sessionKey || "global";
  const existing = sessionState.get(key);
  if (existing) {
    return existing;
  }
  const created = { turnsSinceMemory: 0, turnsSinceSkill: 0 };
  sessionState.set(key, created);
  return created;
}

function deleteSessionState(sessionId?: string, sessionKey?: string): void {
  const key = sessionId || sessionKey;
  if (key) {
    sessionState.delete(key);
  }
}

function toNonNegativeInt(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? Math.trunc(value)
    : fallback;
}

function resolveBridgeConfig(api: OpenClawPluginApi): Required<SkillBridgeConfig> {
  const raw = (api.pluginConfig ?? {}) as SkillBridgeConfig;
  return {
    pythonPath: raw.pythonPath?.trim() || "python3",
    skillsDir: raw.skillsDir?.trim() || "",
    memoryNudgeInterval: toNonNegativeInt(raw.memoryNudgeInterval, 8),
    skillNudgeInterval: toNonNegativeInt(raw.skillNudgeInterval, 15),
    enqueueFlushReminder: raw.enqueueFlushReminder !== false,
  };
}

function resolveSkillsDir(config: Required<SkillBridgeConfig>, workspaceDir?: string): string {
  if (config.skillsDir.trim()) {
    return path.resolve(config.skillsDir);
  }
  if (workspaceDir?.trim()) {
    return path.join(path.resolve(workspaceDir), "skills");
  }
  throw new Error("skill-bridge: workspaceDir is unavailable and skillsDir is not configured.");
}

function createSkillManageTool(api: OpenClawPluginApi, workspaceDir?: string): AnyAgentTool {
  const parameters = {
    type: "object",
    additionalProperties: false,
    properties: {
      action: {
        type: "string",
        enum: ["create", "patch", "edit", "delete", "write_file", "remove_file"],
      },
      name: { type: "string" },
      content: { type: "string" },
      category: { type: "string" },
      file_path: { type: "string" },
      file_content: { type: "string" },
      old_string: { type: "string" },
      new_string: { type: "string" },
      replace_all: { type: "boolean" },
    },
    required: ["action", "name"],
  };

  return {
    name: TOOL_NAME,
    label: "Skill Manage",
    description:
      "Manage reusable workflow skills stored on disk. Create or update SKILL.md files and supporting references, scripts, templates, or assets.",
    parameters,
    async execute(_toolCallId: string, rawParams: Record<string, unknown>) {
      const config = resolveBridgeConfig(api);
      const pluginRoot = api.resolvePath(".");
      const runnerPath = path.join(pluginRoot, "python", "bridge_runner.py");
      const skillsDir = resolveSkillsDir(config, workspaceDir);
      const action = readStringParam(rawParams, "action", { required: true });
      const name = readStringParam(rawParams, "name", { required: true });
      const payload: Record<string, unknown> = {
        ...rawParams,
        action,
        name,
      };
      const category = readStringParam(rawParams, "category");
      if (category && category.trim() && category.trim() !== "workspace") {
        payload.category = category.trim();
      } else {
        delete payload.category;
      }

      const result = spawnSync(config.pythonPath, [runnerPath, JSON.stringify(payload)], {
        cwd: pluginRoot,
        encoding: "utf8",
        timeout: 30_000,
        maxBuffer: 1024 * 1024,
        env: {
          ...process.env,
          SKILL_PLUGIN_SKILLS_DIR: skillsDir,
          SKILL_PLUGIN_TOOL_NAME: TOOL_NAME,
        },
      });

      if (result.error) {
        throw result.error;
      }
      if (result.status !== 0) {
        const stderr = result.stderr?.trim();
        const stdout = result.stdout?.trim();
        throw new Error(stderr || stdout || `skill-bridge python exit code ${result.status}`);
      }

      const stdout = result.stdout?.trim();
      if (!stdout) {
        throw new Error("skill-bridge returned empty output.");
      }

      let parsed: unknown;
      try {
        parsed = JSON.parse(stdout);
      } catch (error) {
        throw new Error(`skill-bridge returned invalid JSON: ${String(error)}`);
      }

      return jsonResult(parsed);
    },
  };
}

export default function register(api: OpenClawPluginApi) {
  api.registerTool((ctx) => createSkillManageTool(api, ctx.workspaceDir), { name: TOOL_NAME });

  api.on("before_prompt_build", async (_event, ctx) => {
    const config = resolveBridgeConfig(api);
    const state = getSessionState(ctx.sessionId, ctx.sessionKey);
    const parts = [SKILL_POLICY_PROMPT, SKILL_CREATOR_STYLE_PROMPT];

    if (config.memoryNudgeInterval > 0) {
      state.turnsSinceMemory += 1;
      if (state.turnsSinceMemory >= config.memoryNudgeInterval) {
        parts.push(MEMORY_NUDGE_TEXT);
        state.turnsSinceMemory = 0;
      }
    }

    if (config.skillNudgeInterval > 0) {
      state.turnsSinceSkill += 1;
      if (state.turnsSinceSkill >= config.skillNudgeInterval) {
        parts.push(SKILL_NUDGE_TEXT);
        state.turnsSinceSkill = 0;
      }
    }

    return { prependContext: parts.join("\n\n") };
  });

  api.on("after_tool_call", async (event, ctx) => {
    const state = getSessionState(ctx.sessionId, ctx.sessionKey);
    if (event.toolName === TOOL_NAME) {
      state.turnsSinceSkill = 0;
    }
    if (event.toolName === "memory" || event.toolName.startsWith("memory_")) {
      state.turnsSinceMemory = 0;
    }
  });

  api.on("before_compaction", async (_event, ctx) => {
    const config = resolveBridgeConfig(api);
    if (!config.enqueueFlushReminder || !ctx.sessionKey) {
      return;
    }
    api.runtime.system.enqueueSystemEvent(FLUSH_REMINDER_TEXT, { sessionKey: ctx.sessionKey });
  });

  api.on("before_reset", async (_event, ctx) => {
    const config = resolveBridgeConfig(api);
    if (!config.enqueueFlushReminder || !ctx.sessionKey) {
      return;
    }
    api.runtime.system.enqueueSystemEvent(FLUSH_REMINDER_TEXT, { sessionKey: ctx.sessionKey });
  });

  api.on("session_end", async (_event, ctx) => {
    deleteSessionState(ctx.sessionId, ctx.sessionKey);
  });
}
