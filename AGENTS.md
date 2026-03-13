# OpenClaw-RL — RL Training Stack for Personalized Agents

## OVERVIEW

OpenClaw-RL is a fully asynchronous reinforcement learning framework that trains personalized AI agents from natural conversation feedback. It wraps self-hosted models in OpenClaw as an OpenAI-compatible API, intercepts live multi-turn conversations, and continuously optimizes the policy in the background without interrupting usage.

The stack decouples agent serving, rollout collection, PRM judging, and policy training into independent async loops. Conversation data never leaves your infrastructure. No external API keys required.

**Key capabilities:**
- Binary RL (GRPO): Process Reward Model scores each turn as good/bad/neutral based on next-state feedback
- On-Policy Distillation (OPD): Extract hindsight hints from feedback and distill them into the policy at token level
- Production-ready: session-aware training, graceful weight updates, at-least-one guarantee, hint quality filtering

**For detailed OpenClaw gateway documentation**, see `openclaw/AGENTS.md` (23KB comprehensive guide).

## STRUCTURE

openclaw-rl/
├── Megatron-LM/
│   ├── docker/
│   ├── docs/
│   ├── examples/
│   ├── images/
│   ├── megatron/
│   ├── scripts/
│   ├── tasks/
│   ├── tests/
│   ├── tools/
│   ├── README.md
│   ├── pyproject.toml
├── assets/
├── instructions/
│   └── README.md
├── openclaw/
│   ├── Swabble/
│   ├── apps/
│   ├── assets/
│   ├── docs/
│   ├── extensions/
│   ├── git-hooks/
│   ├── packages/
│   ├── patches/
│   ├── scripts/
│   ├── skills/
│   ├── src/
│   ├── test/
│   ├── ui/
│   ├── AGENTS.md
│   ├── README.md
│   ├── package.json
│   ├── pyproject.toml
├── openclaw-combine/
│   └── README.md
├── openclaw-opd/
│   └── README.md
├── openclaw-rl/
│   └── README.md
├── openclaw-test/
│   ├── README.md
├── slime/
│   ├── docker/
│   ├── docs/
│   ├── examples/
│   ├── imgs/
│   ├── scripts/
│   ├── slime/
│   ├── slime_plugins/
│   ├── tests/
│   ├── tools/
│   └── README.md
│   ├── pyproject.toml
├── AGENTS.md
├── README.md

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Gateway setup and configuration | `openclaw/AGENTS.md` | 23KB comprehensive guide for OpenClaw gateway |
| Gateway source code | `openclaw/src/` | CLI, commands, channels, routing, media pipeline |
| Gateway extensions | `openclaw/extensions/` | Channel plugins (Matrix, Teams, Zalo, voice-call) |
| Binary RL algorithm | `openclaw-rl/README.md` | Scalar reward via PRM majority vote + GRPO |
| Binary RL server | `openclaw-rl/openclaw_api_server.py` | OpenAI-compatible API + async training loop |
| Binary RL rollout | `openclaw-rl/openclaw_rollout.py` | Conversation trajectory collection |
| On-Policy Distillation | `openclaw-opd/README.md` | Token-level directional distillation |
| Environment setup | `instructions/README.md` | CUDA 12.9, Python 3.12, Slime framework |
| Base RL framework | `slime/` | THUDM Slime (8-GPU default, configurable) |
| LLM training infra | `Megatron-LM/` | Large-scale model training |

## COMMANDS

**Gateway (TypeScript/Node 22+, pnpm):**
```bash
cd openclaw
pnpm install          # Install dependencies
pnpm dev              # Run gateway in dev mode
pnpm test             # Run test suite
pnpm build            # Build for production
pnpm check            # Format check + lint + type check
```

**RL Server (Python 3.12, CUDA 12.9):**
```bash
# Binary RL (scalar reward)
cd slime
bash ../openclaw-rl/run_qwen3_4b_openclaw_rl.sh

# On-Policy Distillation (token-level)
cd slime
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd.sh
```

**Environment variables (RL server):**
- `NUM_GPUS=8` (total GPUs)
- `ACTOR_GPUS=4` (training actor)
- `ROLLOUT_GPUS=2` (rollout generation)
- `PRM_GPUS=2` (Process Reward Model)
- `HF_CKPT` (base checkpoint path)
- `PRM_MODEL_PATH` (reward model path)
- `SAVE_CKPT` (output checkpoint path)
- `SGLANG_API_KEY` (API key for serving endpoint)

## CONVENTIONS

**Gateway (TypeScript):**
- Language: TypeScript (ESM), strict typing, no `any`
- Formatting: oxfmt, oxlint (run `pnpm check` before commits)
- Tests: Vitest, colocated `*.test.ts`, 70% coverage threshold
- Naming: `openclaw` for CLI/package/paths, `OpenClaw` for product/docs
- Commit format: `type(scope): description` (concise, action-oriented)
- Keep files under ~500 LOC when feasible (split/refactor as needed)

**RL Module (Python):**
- Python 3.12+, async-first design
- Logging: use `logging.getLogger(__name__)`, never `print()`
- Configuration: environment variables, no hardcoded secrets
- Session-aware: multi-turn conversations tracked per-session
- JSONL logging: all conversations and PRM evaluations recorded

**Shared:**
- AGENTS.md in project root (this file)
- Docker-first for service orchestration
- No committing model artifacts (`.onnx`, `.gguf`, `.bin`)
- No blocking I/O in async context

## ANTI-PATTERNS

| Forbidden | Why |
|-----------|-----|
| Hardcoded API keys / secrets | Use env vars or config files |
| `print()` in Python services | Use `logging.getLogger(__name__)` |
| Blocking I/O in async context | Async-first design required |
| `as any` / `@ts-ignore` in TypeScript | Type safety required |
| Committing model artifacts | Large binary churn, usually generated |
| Skipping gateway docs | `openclaw/AGENTS.md` is the canonical reference |
| Cross-module imports without boundaries | Use explicit integration layers |
| Unsafe shell patterns | Security and reproducibility risk |
| Feature creep in core modules | Keep concerns separated (gateway vs. RL) |

## NOTES

- **Not a monorepo**: Each project has independent git history
- **Gateway docs**: See `openclaw/AGENTS.md` for comprehensive 23KB guide
- **Hardware baseline**: 8× GPUs (configurable via env vars)
- **Architecture**: 4-component async design (Actor, Rollout, PRM, Critic) with graceful weight updates
- **Research**: Based on Slime framework, RLAnything, and Open-AgentRL
- **Production-ready**: Fully async 4-component architecture with graceful weight updates
