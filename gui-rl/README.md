# GUI-RL: Reinforcement Learning for Desktop GUI Agents

This directory trains and evaluates **vision-language models** (Qwen3-VL) to operate real desktop environments through screenshots and `pyautogui` actions. The agent observes the screen, reasons about the task, and emits keyboard/mouse actions — all inside cloud VMs managed by a pool server.

## Architecture

```
┌────────────────────┐        ┌────────────────────────────────────┐
│  Env Pool Server   │        │         Ray Cluster                │
│  (Flask, port      │  HTTP  │  ┌──────────┐  ┌───────────────┐  │
│   18080)           │◄──────►│  │  Rollout  │  │   Actor /     │  │
│                    │        │  │  (SGLang) │  │   Trainer     │  │
│  allocate / reset  │        │  └──────────┘  └───────────────┘  │
│  step / evaluate   │        │       ▲               ▲            │
│  close             │        │       │  screenshots  │  weights   │
└────────┬───────────┘        └───────┼───────────────┼────────────┘
         │                            │               │
         │  VM lifecycle              │               │
         ▼                            │               │
┌────────────────────┐        ┌───────┴───────────────┴───┐
│  Cloud VMs         │        │   generate_with_gui.py    │
│  (Volcengine /     │        │   (trajectory loop:       │
│   AWS / Aliyun)    │        │    screenshot → agent →   │
│  Ubuntu desktops   │        │    action → step → reward)│
└────────────────────┘        └───────────────────────────┘
```

**Env Pool Server** — A Flask service that manages a pool of cloud VM instances. It handles allocation, task reset (snapshot restore + setup scripts), action execution, evaluation, and teardown. VMs are pre-warmed at launch to eliminate cold-start latency during training.

**Agent** (`Qwen3VLAgentLocal`) — Converts screenshot observations into multi-turn VLM conversations (system prompt + image history + action output), and parses the model's response into executable `pyautogui` calls.

**Reward Agent** (`Qwen3VLRewardAgent`, PRM mode only) — A separate VLM that scores each agent step by comparing the action's intent against the next observation, producing +1/−1 per-step rewards for step-wise advantage estimation.

**Trajectory Loop** (`generate_with_gui.py`) — Orchestrates multi-step episodes: allocate a VM, reset to a task, loop (screenshot → model inference → parse action → execute → observe), compute final reward via OSWorld evaluators, and return the trajectory for training.

## Three Operating Modes

| Mode | Script | Description |
|---|---|---|
| **Eval-only** | `gui_qwen3vl_4b_eval.sh` | Load a checkpoint, run all eval tasks, report accuracy. No training. |
| **RL Training** | `gui_qwen3vl_8b_rl.sh` | GRPO-based RL with sequence-level binary rewards from OSWorld evaluators. |
| **PRM + RL Training** | `gui_qwen3vl_8b_prm_rl.sh` | RL with an additional Process Reward Model that provides per-step rewards via a separate VLM judge. |

## Prerequisites

- 8 GPUs (default; configurable via `NUM_GPUS`)
- A cloud provider account (Volcengine by default) with VM image and network configured
- Volcengine secrets (`VOLC_ACCESSKEY`, `VOLC_SECRETKEY`) exported in your shell
- Python environment with `requirements.txt` installed
- SLIME framework and Megatron-LM available in sibling directories

Install the GUI-specific dependencies:

```bash
pip install -r requirements.txt
playwright install chromium
```

## Step-by-Step Guide

### Mode 1: Evaluation Only

Run a checkpoint on the full eval task suite without any training:

```bash
# Optional: point to your checkpoint (default: Qwen3-VL-4B-Thinking)
export HF_CKPT="/path/to/checkpoint"

# Optional: W&B logging
export WANDB_KEY="your-wandb-key"

cd slime
bash ../gui-rl/gui_qwen3vl_4b_eval.sh
```

The script:
1. Starts the env pool server and pre-warms VMs.
2. Launches a Ray cluster with all GPUs dedicated to rollout (no actor).
3. Runs every task in `evaluation_examples/test_nochrome.json`.
4. Reports accuracy per domain.

### Mode 2: RL Training (GRPO)

Train with sequence-level binary rewards — the agent gets +1 if the OSWorld evaluator marks the task as successful, −1 otherwise:

```bash
export HF_CKPT="/path/to/Qwen3-VL-8B-Thinking"
export WANDB_KEY="your-wandb-key"

# Optional GPU split (default: 4 actor + 4 rollout)
export ACTOR_GPUS=4
export ROLLOUT_GPUS=4

cd slime
bash ../gui-rl/gui_qwen3vl_8b_rl.sh
```

The script partitions GPUs between the training actor and the SGLang rollout engine, then alternates between:
1. **Rollout**: generate trajectories across `n_samples_per_prompt` parallel attempts per task.
2. **Train**: update the policy using GRPO with KL regularization.
3. **Eval**: periodically run the eval task set and log accuracy to W&B.

### Mode 3: PRM-Augmented RL Training

Adds a **Process Reward Model** that scores each step, enabling step-wise advantage estimation instead of sequence-level:

```bash
export HF_CKPT="/path/to/Qwen3-VL-8B-Thinking"
export PRM_MODEL_PATH="/path/to/Qwen3-VL-4B-Thinking"
export WANDB_KEY="your-wandb-key"

# GPU partition: actor + rollout + PRM must equal NUM_GPUS
export ACTOR_GPUS=4
export ROLLOUT_GPUS=3
export PRM_GPUS=1

cd slime
bash ../gui-rl/gui_qwen3vl_8b_prm_rl.sh
```

The PRM agent takes the interaction history plus the next observation after each step, and assigns +1 (progress) or −1 (no effect / regression). These per-step rewards feed into a `step_wise` advantage estimator, providing denser gradient signal than the final-outcome-only GRPO.

## Key Environment Variables

### GPU & Training

| Variable | Default | Description |
|---|---|---|
| `NUM_GPUS` | `8` | Total GPUs on the node |
| `ACTOR_GPUS` | `4` | GPUs for the Megatron training actor |
| `ROLLOUT_GPUS` | `4` (`3` for PRM) | GPUs for the SGLang rollout engine |
| `PRM_GPUS` | `1` | GPUs for the PRM judge engine (PRM mode) |
| `HF_CKPT` | *(model-specific)* | HuggingFace checkpoint path |
| `PRM_MODEL_PATH` | — | Checkpoint for the PRM reward model |
| `ROLLOUT_BATCH_SIZE` | `8` | Tasks per rollout batch |
| `N_SAMPLES_PER_PROMPT` | `8` | Parallel attempts per task |

### GUI Environment

| Variable | Default | Description |
|---|---|---|
| `GUI_POOL_MAX_ENVS` | `64` | Max concurrent VM instances |
| `GUI_PREWARM_ENVS` | `64` | VMs to pre-warm before training starts |
| `GUI_PROVIDER_NAME` | `volcengine` | Cloud provider (`volcengine` / `aws` / `aliyun`) |
| `GUI_REGION` | `cn-beijing` | Cloud region for VM provisioning |
| `GUI_ACTION_SPACE` | `pyautogui` | Action space for the agent |
| `GUI_OBSERVATION_TYPE` | `screenshot` | Observation modality |
| `GUI_SCREEN_WIDTH` | `1920` | VM screen resolution width |
| `GUI_SCREEN_HEIGHT` | `1080` | VM screen resolution height |

### Logging

| Variable | Default | Description |
|---|---|---|
| `WANDB_KEY` | — | Weights & Biases API key |
| `WANDB_PROJECT` | `slime_gui` | W&B project name |
| `GUI_RESULT_DIR` | `./results` | Directory for trajectory logs and eval results |

### Cloud Provider (Volcengine)

Secrets must be exported before running:

```bash
export VOLC_ACCESSKEY="your-access-key"
export VOLC_SECRETKEY="your-secret-key"
```

Non-secret configs (`VOLCENGINE_IMAGE_ID`, `VOLCENGINE_SUBNET_ID`, etc.) have defaults in the shell scripts and can be overridden.

## File Structure

```text
gui-rl/
├── README.md
├── requirements.txt
│
├── gui_qwen3vl_4b_eval.sh            # Eval-only (4B, single node)
├── gui_qwen3vl_8b_rl.sh              # RL training (8B, GRPO)
├── gui_qwen3vl_8b_prm_rl.sh          # PRM + RL training (8B, step-wise)
│
├── generate_with_gui.py               # Trajectory loop: screenshot → action → step → reward
├── env_pool_server.py                 # Flask server managing VM pool lifecycle
├── env_client.py                      # Async HTTP client for the env pool server
├── gui_data_source.py                 # DataSource that loads tasks from meta JSON files
├── build_gui_prompt_dataset.py        # Utility to convert meta JSON → JSONL prompt dataset
│
├── agents/
│   ├── qwen3vl_agent.py               # Policy agent: screenshot → VLM → pyautogui action
│   ├── qwen3vl_reward_agent.py        # PRM reward agent: step scoring via separate VLM
│   └── utils/
│       └── qwen_vl_utils.py           # Image resizing utilities for Qwen VL
│
├── desktop_env/                        # VM environment abstraction (forked from OSWorld)
│   ├── desktop_env.py                  # DesktopEnv class: VM connection, screenshot, action execution
│   ├── actions.py                      # Action parsing and execution
│   ├── controllers/                    # Remote execution controllers (Python, setup scripts)
│   ├── evaluators/                     # OSWorld task evaluators (getters + metrics)
│   ├── providers/                      # Cloud VM providers (Volcengine, AWS, Aliyun)
│   └── server/                         # VM-side server components
│
├── evaluation_examples/                # Task definitions
│   ├── examples/                       # Per-domain task configs (chrome, os, vlc, word, ...)
│   ├── test_nochrome.json              # Eval task meta (excludes Chrome tasks)
│   ├── train_nochrome.json             # Train task meta
│   ├── test_all.json                   # Full eval set
│   └── settings/                       # App-specific config templates
│
└── results/                            # Runtime output (auto-created, not tracked in git)
```
