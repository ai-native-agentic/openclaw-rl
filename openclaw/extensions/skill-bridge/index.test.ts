import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPluginRuntimeMock } from "../test-utils/plugin-runtime-mock.js";

const spawnSyncMock = vi.fn();

vi.mock("node:child_process", () => ({
  spawnSync: spawnSyncMock,
}));

import register from "./index.js";

describe("skill-bridge plugin", () => {
  const hooks: Record<string, Function> = {};
  const registerTool = vi.fn();
  const enqueueSystemEvent = vi.fn();
  const api = {
    id: "skill-bridge",
    name: "Skill Bridge",
    pluginConfig: {},
    config: {},
    logger: {
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
      debug: vi.fn(),
    },
    runtime: createPluginRuntimeMock({
      system: {
        enqueueSystemEvent: enqueueSystemEvent as any,
      },
    }),
    resolvePath: vi.fn((input: string) =>
      input === "." ? "/repo/openclaw/extensions/skill-bridge" : input,
    ),
    registerTool,
    on: vi.fn((name: string, handler: Function) => {
      hooks[name] = handler;
    }),
  };

  beforeEach(() => {
    vi.clearAllMocks();
    spawnSyncMock.mockReset();
    for (const key of Object.keys(hooks)) {
      delete hooks[key];
    }
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("registers tool and lifecycle hooks", () => {
    register(api as any);

    expect(registerTool).toHaveBeenCalledTimes(1);
    expect(registerTool).toHaveBeenCalledWith(expect.any(Function), { name: "skill_manage" });
    expect(api.on).toHaveBeenCalledWith("before_prompt_build", expect.any(Function));
    expect(api.on).toHaveBeenCalledWith("after_tool_call", expect.any(Function));
    expect(api.on).toHaveBeenCalledWith("before_compaction", expect.any(Function));
    expect(api.on).toHaveBeenCalledWith("before_reset", expect.any(Function));
    expect(api.on).toHaveBeenCalledWith("session_end", expect.any(Function));
  });

  it("executes the bundled python bridge with workspace skills by default", async () => {
    register(api as any);
    const toolFactory = registerTool.mock.calls[0]?.[0] as Function;
    const tool = toolFactory({ workspaceDir: "/tmp/workspace" });

    spawnSyncMock.mockReturnValue({
      status: 0,
      stdout: JSON.stringify({ success: true, message: "ok" }),
      stderr: "",
    });

    const result = await tool.execute("call-1", {
      action: "create",
      name: "demo-skill",
      category: "workspace",
      content: "---\nname: demo-skill\ndescription: demo\n---\nbody",
    });

    expect(spawnSyncMock).toHaveBeenCalledWith(
      "python3",
      [
        "/repo/openclaw/extensions/skill-bridge/python/bridge_runner.py",
        expect.any(String),
      ],
      expect.objectContaining({
        cwd: "/repo/openclaw/extensions/skill-bridge",
        env: expect.objectContaining({
          SKILL_PLUGIN_SKILLS_DIR: "/tmp/workspace/skills",
          SKILL_PLUGIN_TOOL_NAME: "skill_manage",
        }),
      }),
    );
    expect(JSON.parse(spawnSyncMock.mock.calls[0]?.[1]?.[1] as string)).toEqual({
      action: "create",
      name: "demo-skill",
      content: "---\nname: demo-skill\ndescription: demo\n---\nbody",
    });
    expect(result).toMatchObject({
      content: [
        {
          type: "text",
          text: expect.stringContaining('"success": true'),
        },
      ],
      details: {
        success: true,
        message: "ok",
      },
    });
  });

  it("adds flush reminders when enabled", async () => {
    register(api as any);

    await hooks.before_compaction({}, { sessionKey: "session-1" });
    await hooks.before_reset({}, { sessionKey: "session-1" });

    expect(enqueueSystemEvent).toHaveBeenCalledTimes(2);
    expect(enqueueSystemEvent).toHaveBeenCalledWith(
      expect.stringContaining("Context may be compacted or reset soon."),
      { sessionKey: "session-1" },
    );
  });
});
