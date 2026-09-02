#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Suspended-base gravity torque g(q) for X1 joints, via MuJoCo (lazy import).

Method (no integration; contacts disabled => robot hangs in the sling):
  1. load spi_identify/resources/mjcf/xyber_x1_flat.xml
  2. disable contacts: model.opt.disableflags |= mjDSBL_CONTACT
  3. for each scan q: data.qpos[adr] = q, qvel = 0
  4. mujoco.mj_forward(model, data)          # analysis only, NEVER mj_step
  5. g = data.qfrc_bias[dof_adr]             # gravity of the suspended chain

A uniform 200-point scan is interpolated (np.interp) at runtime. Local
numpy-only unit tests use a synthetic quadratic g(q) instead; mujoco is
required only for the real run (gradmotion image).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "spi_identify" / "resources" / "mjcf" / "xyber_x1_flat.xml"


def mujoco_available() -> bool:
    try:
        import mujoco  # noqa: F401
        return True
    except Exception:
        return False


def build_gravity_lut(joint_name, model_path=None, q_range=None, n=200):
    """Gravity-torque LUT g(q) [Nm] for one joint (suspended, contact-free)."""
    import mujoco

    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    model = mujoco.MjModel.from_xml_path(str(path))
    model.opt.disableflags |= int(mujoco.mjtDisableFlag.mjDSBL_CONTACT)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name))
    if jid < 0:
        raise ValueError("joint '%s' not found in model %s" % (joint_name, path))
    qadr = int(model.jnt_qposadr[jid])
    dadr = int(model.jnt_dofadr[jid])
    if q_range is None:
        if bool(model.jnt_limited[jid]):
            rng = model.jnt_range[jid]
            q_range = (float(rng[0]), float(rng[1]))
        else:
            q_range = (-float(np.pi), float(np.pi))
    lo, hi = float(q_range[0]), float(q_range[1])
    data = mujoco.MjData(model)
    q_grid = np.linspace(lo, hi, int(n))
    g_grid = np.zeros_like(q_grid)
    for i, qv in enumerate(q_grid):
        data.qpos[qadr] = qv
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        g_grid[i] = data.qfrc_bias[dadr]
    return {"joint": str(joint_name), "model": str(path), "source": "mujoco_qfrc_bias",
            "q_range": (lo, hi), "q_grid": q_grid, "g_grid": g_grid}


def zero_lut(joint_name, q_range=(-np.pi, np.pi), n=200):
    """Degraded fallback LUT g(q)=0 (flagged; dynamics gate may then FAIL)."""
    q_grid = np.linspace(float(q_range[0]), float(q_range[1]), int(n))
    return {"joint": str(joint_name), "model": None, "source": "zeros_degraded",
            "q_range": (float(q_range[0]), float(q_range[1])),
            "q_grid": q_grid, "g_grid": np.zeros_like(q_grid)}


def gravity_lookup(q, lut):
    """Interpolate g(q) [Nm] (linear, clamped at both ends)."""
    q = np.asarray(q, dtype=float)
    return np.interp(q, lut["q_grid"], lut["g_grid"])
