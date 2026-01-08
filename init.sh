#!/bin/bash
# =============================================================================
# GiftLive Environment Initialization
# =============================================================================
# Usage: source init.sh
# =============================================================================

# Load .env if present (local defaults)
if [ -f ./.env ]; then
  set -a
  . ./.env
  set +a
  echo "✅ Loaded .env"
fi

# Conda setup
source /srv/local/tmp/swei20/miniconda3/etc/profile.d/conda.sh
conda activate viska-torch-3

# Set project root
export GIFTLIVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${GIFTLIVE_ROOT}:${PYTHONPATH}"

echo "=============================================="
echo "🎁 GiftLive Environment Initialized"
echo "=============================================="
echo "ROOT:       $GIFTLIVE_ROOT"
echo "PYTHONPATH: $PYTHONPATH"
echo "Conda Env:  $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Python:     $(which python)"
echo "=============================================="
