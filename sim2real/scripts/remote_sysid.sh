#!/usr/bin/env bash
# gradmotion remote bootstrap: install deps + run the full SPI sysid pipeline.
# startScript form:
#   gm-run X1_identify/sim2real/scripts/remote_sysid.py [--validate-only]
#
# Repo layout (self-contained; no F1/Humanoid_motion sibling checkout needed):
#   ./X1_identify/                      this repo
#     ├── data/raw/walk_diag_*.csv      real-robot data (DATA-01, 100 Hz)
#     ├── data/derived/                 step M1 regression evidence + clips npz
#     ├── X1_infer/module/.../mjcf/     MuJoCo model (nominal, from sim_module)
#     └── sim2real/                     SPI pipeline (vendored from F1 dev/sim2real-spi)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"          # .../X1_identify/sim2real/scripts
REPO="$(cd "$HERE/../.." && pwd)"              # repo root (any name on the platform)
cd "$REPO"

# mode: full pipeline (default) or --validate-only (dataset + unittests +
# validation + apply; skips the ~15 min CMA-ES identification, reuses
# committed params from sim2real/results/)
MODE="${1:-full}"
if [ "$MODE" = "--validate-only" ]; then
  echo "remote_sysid: VALIDATE-ONLY mode (skip identification)"
fi

# --- model sanity (MJCF lives in this repo: X1_infer) ----------------------
MJCF_PATH="X1_infer/module/sim_module/model/mjcf/xyber_x1_flat.xml"
if [ ! -f "$MJCF_PATH" ]; then
  echo "FATAL: MJCF not found at $MJCF_PATH" >&2
  exit 3
fi

# --- python deps (image ships torch; mujoco/optuna are pip-only) ----------
python -m pip install -q --no-input mujoco optuna cmaes pyyaml matplotlib 2>&1 | tail -1 || true
python - <<'PY'
import mujoco, optuna, cmaes, yaml, numpy
print("deps OK:", "mujoco", mujoco.__version__, "| optuna", optuna.__version__,
      "| cmaes", cmaes.__version__)
PY

# --- stage 0: unit tests (numpy-level; gate the pipeline) ------------------
python -m unittest discover -s sim2real/tests -v 2>&1 | tail -8

# --- stage 1: dataset ------------------------------------------------------
python sim2real/scripts/prepare_dataset.py \
  --config sim2real/configs/x1_spi.yaml \
  --out data/derived/x1_clips.npz

if [ "$MODE" != "--validate-only" ]; then
  # --- stage 2: SPI identification ----------------------------------------
  python sim2real/scripts/run_spi.py \
    --config sim2real/configs/x1_spi.yaml \
    --dataset data/derived/x1_clips.npz \
    --out-dir logs/spi_sysid
else
  echo "remote_sysid: [validate-only] params must come from a previous run"
  # fresh container has no logs/ — fall back to the repo-committed params file
  # (F1 v15 PASS result, same data/model; payload format identical)
  if [ ! -f logs/spi_sysid/gm_play/identified_params.json ]; then
    mkdir -p logs/spi_sysid/gm_play
    cp sim2real/results/identified_params.json \
       logs/spi_sysid/gm_play/identified_params.json
    echo "remote_sysid: [validate-only] restored params from sim2real/results/"
  fi
  ls -la logs/spi_sysid/gm_play/identified_params.json
fi

# --- stage 3: validation (completion criteria, 完成标准) --------------------
# exit code: 0=PASS 1=FAIL 2=error; run after artifacts so logs always exist
# (set -e would kill the pipeline on rc=1 before diagnostics run — capture it)
VALIDATE_RC=0
python sim2real/scripts/validate_spi.py \
  --config sim2real/configs/x1_spi.yaml \
  --dataset data/derived/x1_clips.npz \
  --params logs/spi_sysid/gm_play/identified_params.json \
  --out-dir logs/spi_sysid || VALIDATE_RC=$?
echo "remote_sysid: validation exit code = $VALIDATE_RC"

# --- diagnostics: mass landscape ------------------------------------------
python sim2real/scripts/mass_landscape.py \
  --config sim2real/configs/x1_spi.yaml \
  --dataset data/derived/x1_clips.npz \
  --out-dir logs/mass_landscape || true

# --- apply params (URDF/MJCF patch + DR config) ---------------------------
python sim2real/scripts/apply_params.py \
  --params logs/spi_sysid/gm_play/identified_params.json \
  --out-dir sim2real/export || true

echo "remote_sysid: ALL DONE (validation rc=$VALIDATE_RC)"
ls -R logs/ | head -40

# propagate the validation verdict as the task exit code: 0 PASS / 1 FAIL
exit "$VALIDATE_RC"
