# Site settings for the Slurm scripts. Copy to slurm/env.sh (not tracked by git) and edit.
# Every script in slurm/ sources slurm/env.sh if it exists. Keep secrets out of it:
# put `export HF_TOKEN=...` in ~/.bashrc on the cluster instead.

# 1) Software: whatever gives you Python 3.10+ with PyTorch 2.x + CUDA 12 on this site.
#    Example for a project-local virtualenv (see NAIRR.md, "Environment"):
# module purge; module load modtree/gpu          # Anvil
# source "$PROJECT/maxgpt-venv/bin/activate"

# 2) Where the big files live (scratch on Anvil; the node-local NVMe on Expanse AI).
export MAXGPT_DATA="${MAXGPT_DATA:-$PWD/data}"                 # shards_train/, shards_val_ultra/, shards_anneal/ live here
export MAXGPT_SOURCES_ROOT="${MAXGPT_SOURCES_ROOT:-$PWD/data/sources}"   # downloaded dataset files (about 500 GB)
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"                    # Hugging Face cache; keep it off the small home quota

# 3) Run settings (defaults shown).
export MAXGPT_RUN="${MAXGPT_RUN:-runs/pretrain}"
export MAXGPT_CONFIG="${MAXGPT_CONFIG:-configs/ultra_h100.yaml}"
export MAXGPT_GPUS="${MAXGPT_GPUS:-4}"
export PYTHONUNBUFFERED=1
