#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 WEATHER.epw SEED"
    exit 1
fi

WEATHER="$(realpath "$1")"
SEED="$2"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENERGYPLUS_DIR="${ENERGYPLUS_DIR:-/usr/local/EnergyPlus-22-1-0}"

case "$SEED" in
    42|123|2026) ;;
    *)
        echo "Seed must be 42, 123, or 2026."
        exit 1
        ;;
esac

cd "$ROOT/src"

python evaluate_ppo_joint_shade_categorical_dwell.py   --model "$ROOT/models/categorical/seed${SEED}/model.zip"   --vecnormalize "$ROOT/models/categorical/seed${SEED}/vecnormalize.pkl"   --idf "$ROOT/building/highrise_multivariable_binary_shade.idf"   --weather "$WEATHER"   --energyplus-dir "$ENERGYPLUS_DIR"   --shade-mode policy   --shade-min-dwell-steps 12   --output "$ROOT/outputs/categorical_dwell120_seed${SEED}"   --seed "$SEED"
