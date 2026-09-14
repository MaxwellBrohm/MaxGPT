#!/usr/bin/env bash
# MaxGPT-Ultra bootstrap for the school Lambda box: no sudo, everything lives in $HOME.
#
#   git clone https://github.com/MaxwellBrohm/MaxGPT.git ~/MaxGPT
#   bash ~/MaxGPT/maxgpt-ultra/scripts/lambda_setup.sh
#
# Re-runnable. Installs uv -> Python 3.12 -> ~/venv, a torch wheel matched to the
# installed driver, the project deps, then runs the same smoke tests as the WSL
# runbook (Phase B) and prints a hardware summary to paste back into the chat.

set -euo pipefail

REPO="$HOME/MaxGPT/maxgpt-ultra"
VENV="$HOME/venv"
PY="3.12"

step() { printf '\n==> %s\n' "$*"; }

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: is this the GPU box?"; exit 1; }
[ -d "$REPO" ] || { echo "repo not found at $REPO (clone it first, see the header)"; exit 1; }

# 1) uv: user-space Python installer, no sudo needed (Ubuntu 20.04's python3 is 3.8, too old for us)
step "uv"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version

# 2) Python + venv at the same path the WSL runbook uses, so its commands carry over unchanged
step "Python $PY -> $VENV"
uv python install "$PY"
[ -x "$VENV/bin/python" ] || uv venv --python "$PY" "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python --version

# 3) torch wheel matched to the driver (the driver decides which CUDA runtime it can host)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
MAJ=${DRV%%.*}
if   [ "$MAJ" -ge 570 ]; then CU=cu128     # same wheel as the 5070 box
elif [ "$MAJ" -ge 525 ]; then CU=cu126     # CUDA 12.x minor-version compatibility
else                          CU=cu118     # pre-CUDA-12 driver
fi
step "torch ($CU for driver $DRV)"
uv pip install --python "$VENV/bin/python" --index-url "https://download.pytorch.org/whl/$CU" torch

# 4) project deps (+ the two the 5070 box installs by hand)
step "project deps"
uv pip install --python "$VENV/bin/python" -r "$REPO/requirements.txt" datasets bitsandbytes

# 5) smoke tests on the emptiest card: cuda, bf16, compile, paged 8-bit optimizer
#    (tiny tensors, but a CUDA context still costs a few hundred MB, so stay off busy cards)
step "smoke tests"
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$FREE_GPU}"
echo "using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python - <<'PYEOF'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| built for CUDA", torch.version.cuda)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  visible gpu{i}: {p.name}  {p.total_memory / 2**30:.1f}GB  sm_{p.major}{p.minor}")
print("bf16 supported:", torch.cuda.is_bf16_supported())
m = torch.nn.Linear(8, 8).cuda()
print("compile:", tuple(torch.compile(m)(torch.randn(4, 8, device="cuda")).shape))
import bitsandbytes as bnb
opt = bnb.optim.PagedAdamW8bit(m.parameters(), lr=1e-3)
m(torch.randn(4, 8, device="cuda")).sum().backward(); opt.step()
print("paged 8-bit AdamW: OK (bitsandbytes", bnb.__version__ + ")")
PYEOF

# 6) summary to paste back
step "box summary (paste this into the chat)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
echo "driver $DRV | cpus $(nproc) | ram $(free -g | awk '/Mem/{print $2}')G | $(lsb_release -ds 2>/dev/null)"
df -h "$HOME" /tmp
df -h 2>/dev/null | awk 'NR > 1 && $4 ~ /T$/ {print "big mount:", $6, "free:", $4}'
curl -s -m 5 -o /dev/null -w "outbound internet: HTTP %{http_code}\n" https://huggingface.co || echo "outbound internet: none"
printf '\nnext: read %s/LAMBDA.md\n' "$REPO"
