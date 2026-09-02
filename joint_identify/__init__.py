# -*- coding: utf-8 -*-
"""X1 joint-level system identification package.

Submodules
----------
regress         pure-numpy regression primitives (unit-testable, no mujoco)
gravity_torque  lazy-mujoco suspended gravity torque LUT g(q)

scripts/run_joint_identify.py  main entry (12 step CSVs -> joint_params)
scripts/validate_joint.py      GATE-J1..J5 validation (exit code = PASS/FAIL)

Reads data/raw/*_step_*.csv, data/derived/step_m1_regression_all.json and
spi_identify/resources/mjcf/xyber_x1_flat.xml; writes logs/joint_identify/.
"""
