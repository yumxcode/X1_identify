#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final combined remote validation (single-task, budget-efficient).

Runs, in order:
  1. joint_identify unit tests (numpy-only, seconds)
  2. joint-level identification on the 12 step CSVs (mujoco gravity LUT)
     -> GATE-J1..J5 verdict
  3. SPI dual-bucket dataset preparation (train + cross)
  4. SPI five-criterion validation of the committed R8-projected params
     (re-baselined floors) on the NEW multi-dataset data
     -> EFFECTIVENESS/PHYSICAL/ACCEL/ACTUATOR/CROSS-DATASET verdict

Optional: --params-file=PATH revalidates another committed params file
(default spi_identify/results/r8p_projected_params.json).

Exit code: 0 only if BOTH verdicts are PASS (bitwise-or of the two rc's,
keeping 1 = FAIL, 2 = error).

startScript:
  gm-run X1_identify/joint_identify/scripts/remote_finalize.py [--params-file=...]
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

PARAMS_FILE = "spi_identify/results/r8p_projected_params.json"
for a in sys.argv[1:]:
    if a.startswith("--params-file="):
        PARAMS_FILE = a.split("=", 1)[1]


def run(cmd, desc, allow_fail=True):
    print(f"\n=== [finalize] {desc}: {' '.join(str(c) for c in cmd)}", flush=True)
    rc = subprocess.call([str(c) for c in cmd], cwd=str(REPO))
    print(f"=== [finalize] {desc} rc={rc}", flush=True)
    return rc if allow_fail else (sys.exit(2) if rc else 0)


def main():
    py = sys.executable
    subprocess.call([py, "-m", "pip", "install", "-q", "--no-input",
                     "mujoco", "pyyaml"], cwd=str(REPO))

    # 1+2: joint-level identification (gate verdict = its exit code)
    rc_joint = run([py, "-m", "unittest", "discover", "-s",
                    "joint_identify/tests"], "joint unit tests", allow_fail=False)
    rc_joint |= run([py, "joint_identify/scripts/run_joint_identify.py"],
                    "joint identification (GATE-J1..J5)")

    # 3: dual-bucket SPI dataset
    rc_joint |= run([py, "spi_identify/scripts/prepare_dataset.py",
                     "--config", "spi_identify/configs/x1_spi.yaml",
                     "--out", "data/derived/x1_clips.npz",
                     "--out-cross", "data/derived/x1_cross_clips.npz"],
                    "SPI dataset (train + cross)")

    # 4: five-criterion SPI validation on the committed R8-projected params
    rc_spi = run([py, "spi_identify/scripts/validate_spi.py",
                  "--config", "spi_identify/configs/x1_spi.yaml",
                  "--dataset", "data/derived/x1_clips.npz",
                  "--cross-dataset", "data/derived/x1_cross_clips.npz",
                  "--params", PARAMS_FILE,
                  "--out-dir", "logs/spi_sysid"],
                 "SPI validation (5 criteria, R8-projected params)")

    print(f"\n[finalize] joint rc={rc_joint & 1} | spi rc={rc_spi & 1}")
    print("[finalize] FINAL:", "PASS" if (rc_joint | rc_spi) == 0 else "FAIL")
    return 1 if (rc_joint | rc_spi) != 0 else 0


if __name__ == "__main__":
    sys.exit(main())
