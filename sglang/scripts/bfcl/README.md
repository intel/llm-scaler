# BFCL Evaluation Kit on Intel XPU (sglang)

This README provides a step-by-step guide to building and running the Berkeley Function Calling Leaderboard (BFCL) from scratch inside a container. This kit is designed to evaluate models (e.g., Qwen3.6-35B-A3B) using sglang on Intel XPU.

## 1. Prerequisites
Ensure you are inside the GPU container with the Intel OneAPI environment and sglang installed. You will need a Python virtual environment (default is /opt/venv).

## 2. Clone the Repository and Set Environment Variables
First, create a root directory for the evaluation kit (e.g., /workspace/bfcl_kit), clone the target fork into a vendor subdirectory, and checkout the required commit.

# Step 1. Define the kit root directory

```bash
export KIT_ROOT="/workspace/bfcl_kit"
mkdir -p "$KIT_ROOT"
cd "$KIT_ROOT"
```
 
# Step 2. Clone the repository into the 'vendor' directory

```bash
git clone https://github.com/zhangYiIntel/gorilla.git vendor
```
 
# Step 3. Enter the directory and checkout the exact pinned commit

```bash
cd vendor
git checkout 6ee7b7718d9c7498b26e043635db6381a6583593
```

# Step 4. Configure Environment Variables
Before running the installation and evaluation scripts, you must export the necessary environment variables. These include paths to your local model weights, tokenizer, and configuration directories.

```bash
# Local Model Weights & Tokenizer Directory (TOK_DIR)
# This should point to the directory containing your tokenizer.json and model config files.
# If left empty, the prepare script will attempt to fetch the unsloth tokenizer.
export TOK_DIR="/models/unsloth/Qwen3.6-35B-A3B" 
 
 
# Server and XPU configurations
export PORT="9010"
export ZE_MASK="0"               # Which XPU tile to use
```

# Step 5. Install BFCL
Run the installation script to install the vendored BFCL package into your virtual environment. This script performs an offline-first editable install from the vendor directory.

```bash
bash 02_install_bfcl_fork.sh
```

Note: This script installs the package from $KIT_ROOT/vendor/berkeley-function-call-leaderboard and verifies that bfcl_eval resolves correctly without being shadowed by a stale site-packages version.

# Step 6. Prepare the Workspace
Generate the RUN_CONFIG.sh file and ensure the tokenizer is properly staged. The script uses the TOK_DIR environment variable to locate your local model weights/tokenizer directory.

```bash
bash 03_prepare_workspace.sh
```

Note: If TOK_DIR is not set or the tokenizer.json is missing, this script will automatically attempt to download the unsloth/Qwen3.6-35B-A3B tokenizer files.

6. Run the BFCL Evaluation
To launch a fresh server (cold radix) and run the BFCL multi-turn evaluation:

```bash
SKIP_START=1 MODEL_ID="Qwen/Qwen3.6-35B-A3B-FC" BFCL_NUM_THREADS=1 bash 04_run.sh multi_turn_base
```

Usage options for 04_run.sh:

```bash
bash 04_run.sh multi_turn_base — Run the full multi_turn_base (200) category.
bash 04_run.sh multi_turn_base 6 — Run a single entry base_6 (smoke test).
bash 04_run.sh multi_turn_base 0-29 — Run a subset base_0..29.
IDS="6,10,42" bash 04_run.sh — Run an explicit list of IDs.
BFCL_NUM_THREADS=16 bash 04_run.sh — Run on 16 BFCL workers.
```

Note: It is suggested that threads=1 as only batch-size 1 is guaranteed to be valid on XPU: Batch Invariant hasn't been achieved on vllm for XPU yet. 