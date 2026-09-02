#!/usr/bin/env bash
# gradmotion remote bootstrap: install deps + run the full SPI sysid pipeline.
# startScript form:
#   gm-run X1_identify/spi_identify/scripts/remote_sysid.py [--validate-only]
#
# Repo layout (self-contained; no F1/Humanoid_motion sibling checkout needed):
#   ./X1_identify/                      this repo
#     ├── data/raw/walk_diag_*.csv      real-robot data (DATA-01, 100 Hz)
#     ├── data/derived/                 step M1 regression evidence + clips npz
#     ├── X1_infer/module/.../mjcf/     MuJoCo model (nominal, from sim_module)
#     └── spi_identify/                     SPI pipeline (vendored from F1 dev/sim2real-spi)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"          # .../X1_identify/spi_identify/scripts
REPO="$(cd "$HERE/../.." && pwd)"              # repo root (any name on the platform)
cd "$REPO"

# mode: full pipeline (default) or --validate-only (dataset + unittests +
# validation + apply; skips the ~15 min CMA-ES identification, reuses
# committed params from spi_identify/results/)
# Args are passed through to run_spi.py (e.g. --seed N --n-trials N) or, in
# validate-only mode, --params-file=PATH selects the committed params file.
# NOTE: MODE matching is flag-aware ("--seed 3" must NOT be taken for the
# mode; historically MODE="${1:-full}" swallowed the first flag, so all three
# T2 seed tasks ran the identical config -- see methods_log 2026-09-02).
MODE="full"
PARAMS_FILE=""
PASSTHROUGH=()
for a in "$@"; do
  case "$a" in
    --validate-only) MODE="--validate-only" ;;
    --params-file=*) PARAMS_FILE="${a#*=}" ;;
    *) PASSTHROUGH+=("$a") ;;
  esac
done
if [ "$MODE" = "--validate-only" ]; then
  echo "remote_sysid: VALIDATE-ONLY mode (skip identification)"
fi

# --- model sanity (vendored SPI MJCF; meshes from X1_infer via meshdir) ----
MJCF_PATH="spi_identify/resources/mjcf/xyber_x1_flat.xml"
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
python -m unittest discover -s spi_identify/tests -v 2>&1 | tail -8

# --- stage 1: dataset ------------------------------------------------------
python spi_identify/scripts/prepare_dataset.py \
  --config spi_identify/configs/x1_spi.yaml \
  --out data/derived/x1_clips.npz \
  --out-cross data/derived/x1_cross_clips.npz

if [ "$MODE" != "--validate-only" ]; then
  # --- stage 2: SPI identification ----------------------------------------
  # remaining args pass through to run_spi.py (e.g. --seed N --n-trials N)
  python spi_identify/scripts/run_spi.py \
    --config spi_identify/configs/x1_spi.yaml \
    --dataset data/derived/x1_clips.npz \
    --out-dir logs/spi_sysid "${PASSTHROUGH[@]}"
else
  echo "remote_sysid: [validate-only] params must come from a previous run"
  # fresh container has no logs/ — fall back to a repo-committed params file
  # (default: F1 v15 PASS result; --params-file=PATH revalidates another set)
  if [ ! -f logs/spi_sysid/gm_play/identified_params.json ]; then
    mkdir -p logs/spi_sysid/gm_play
    cp "${PARAMS_FILE:-spi_identify/results/identified_params.json}" \
       logs/spi_sysid/gm_play/identified_params.json
    echo "remote_sysid: [validate-only] restored params from ${PARAMS_FILE:-spi_identify/results/identified_params.json}"
  fi
  ls -la logs/spi_sysid/gm_play/identified_params.json
fi

# --- stage 3: validation (completion criteria, 完成标准) --------------------
# exit code: 0=PASS 1=FAIL 2=error; run after artifacts so logs always exist
# (set -e would kill the pipeline on rc=1 before diagnostics run — capture it)
VALIDATE_RC=0
python spi_identify/scripts/validate_spi.py \
  --config spi_identify/configs/x1_spi.yaml \
  --dataset data/derived/x1_clips.npz \
  --cross-dataset data/derived/x1_cross_clips.npz \
  --params logs/spi_sysid/gm_play/identified_params.json \
  --out-dir logs/spi_sysid || VALIDATE_RC=$?
echo "remote_sysid: validation exit code = $VALIDATE_RC"

# --- diagnostics: mass landscape ------------------------------------------
python spi_identify/scripts/mass_landscape.py \
  --config spi_identify/configs/x1_spi.yaml \
  --dataset data/derived/x1_clips.npz \
  --out-dir logs/mass_landscape || true

# --- apply params (URDF/MJCF patch + DR config) ---------------------------
python spi_identify/scripts/apply_params.py \
  --params logs/spi_sysid/gm_play/identified_params.json \
  --out-dir spi_identify/export || true

echo "remote_sysid: ALL DONE (validation rc=$VALIDATE_RC)"
ls -R logs/ | head -40

# propagate the validation verdict as the task exit code: 0 PASS / 1 FAIL
exit "$VALIDATE_RC"
