# Remote worker (pool server)

This directory runs on the **remote worker**: a pool server that manages Docker containers and executes terminal tasks.

## Prerequisites

Set up a machine that will act as a worker node: a cloud VM (e.g. AWS EC2, GCP, or any provider), a bare-metal server, or any host where:

- You can install **Docker** (and Docker Compose).
- You have network connectivity so that the **training cluster** (where the router runs) can reach this host on the port you use for the pool server (default **18081**).

### GPU (optional)

A GPU is not required to run the pool server, but may be required by some tasks.

---

## Instructions

### 1. Clone the repo

From a directory of your choice:

```bash
git clone https://github.com/Gen-Verse/OpenClaw-RL.git
cd OpenClaw-RL
```

### 2. Install dependencies

Install Docker, a Python 3.12 environment, and the Python packages required by the pool server. You can use the provided script (run from **repo root**):

```bash
bash terminal-rl/remote/setup.sh
```

This will:

- Install Docker and Docker Compose if missing.
- Install [uv](https://github.com/astral-sh/uv) and create a virtualenv at repo root (`.venv`).
- Install other required packages.

### 3. Download dataset

To download a dataset:

```bash
source .venv/bin/activate
export DATASET_DIR="terminal-rl/dataset"
python terminal-rl/data_utils/download.py seta_env
```

The `seta_env` dataset corresponds to the task dataset published in: [camel-ai/seta-env](https://github.com/camel-ai/seta-env/tree/main/Dataset).

### 4. Run the pool server

From the **repo root**:

```bash
bash terminal-rl/remote/run_pool_server.sh
```

This script:

- Activates the venv if `.venv` exists.
- Sets `DATASET_DIR` and `TBENCH_OUTPUT_ROOT` under `terminal-rl/` by default.
- Starts the pool server with `python -m terminal-rl.remote.pool_server` on `0.0.0.0:18081` (overridable via `ENV_SERVER_PORT`, `WORKER_MAX_TASKS`, `WORKER_MAX_RUNS_PER_TASK`).

Run in background / under a process manager as needed. Example (nohup`):

```bash
nohup bash terminal-rl/remote/run_pool_server.sh > pool_server.log 2>&1 &
```

### 5. Tell the training side the worker URL

On the training machine (router host), set `WORKER_URLS` to include this worker:

```bash
export WORKER_URLS="http://<this-machine-ip-or-hostname>:18081"
```

For multiple workers, use a comma-separated list:

```bash
export WORKER_URLS="http://worker1:18081,http://worker2:18081"
```

Then start the router and training as described in the main Terminal RL docs; the router forwards requests to these pool servers.

---

## Optional environment variables

When running the pool server (via `run_pool_server.sh` or `python -m terminal-rl.remote.pool_server`), the following variables are supported:


| Variable                     | Default                     | Description                                                                                                 |
| ---------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `DATASET_DIR`                | `terminal-rl/dataset`       | Path to the task dataset directory.                                                                         |
| `TBENCH_OUTPUT_ROOT`         | `terminal-rl/build_outputs` | Root directory for build/output artifacts.                                                                  |
| `ENV_SERVER_PORT`            | `18081`                     | Port the pool server listens on.                                                                            |
| `WORKER_MAX_TASKS`           | `16`                        | Max tasks allocate to per worker.                                                                           |
| `WORKER_MAX_RUNS_PER_TASK`   | `8`                         | Max concurrent runs per task.                                                                               |
| `TBENCH_DOCKER_IMAGE_SOURCE` | `build`                     | `build` or `pull` — build images locally or pull from a registry.                                           |
| `TBENCH_DOCKER_PULL_PREFIX`  | —                           | Image name prefix used in `pull` mode; the task name is appended (e.g., `task-1374` → `<prefix>task-1374`). |
| `COMPOSE_OVERRIDE_PATH`      | —                           | Optional Docker Compose override file.                                                                      |


Example with custom port and limits:

```bash
export ENV_SERVER_PORT=18082
export WORKER_MAX_TASKS=10
export WORKER_MAX_RUNS_PER_TASK=8
bash terminal-rl/remote/run_pool_server.sh
```

Example using pre-built images from a registry (pull mode). Set the image source and prefix; you can build and push your own:

```bash
export TBENCH_DOCKER_IMAGE_SOURCE=pull
export TBENCH_DOCKER_PULL_PREFIX="ghcr.io/<your-org>/<your-image>:task-"
export COMPOSE_OVERRIDE_PATH="terminal-rl/remote/compose_override.yaml"
bash terminal-rl/remote/run_pool_server.sh
```

