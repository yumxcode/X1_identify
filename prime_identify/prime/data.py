"""walk_diag CSV loader -> identification dataset arrays.

Adaptation note (no mocap available, unlike PRIME's Go2/G1 hardware):
- base orientation: IMU quaternion (imu_quat_*)
- base angular velocity: IMU gyro (imu_gyro_*, body frame == pinocchio LOCAL)
- base linear velocity / position: UNOBSERVED. Base height (needed for contact
  distances phi) is estimated per-frame by the support-foot constraint:
  z_base := z_base_nominal - min_i(world_z(foot_i))  (lowest foot on ground).
  Base x,y are irrelevant (translation invariance of the dynamics).
- joint pos/vel/torque: pos_*/vel_*/effort_* columns (100 Hz).

The unobserved base linear velocity is handled by the PFIE objective: its
residual rows are masked in Stage A and it is optimized as a free state in
Stage B (multiple shooting).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pinocchio as pin

from .dynamics import JOINT_ORDER, X1Dynamics


@dataclass
class WalkData:
    t: np.ndarray  # (T,) seconds
    dt: float
    q: np.ndarray  # (T, nq) pinocchio configuration
    v: np.ndarray  # (T, nv) measured velocity (base lin vel = 0 placeholder)
    u: np.ndarray  # (T, 12) joint torques
    gyro: np.ndarray  # (T, 3) IMU body angular velocity
    base_quat: np.ndarray  # (T, 4) [x,y,z,w]
    foot_z: np.ndarray  # (T, 4) world-frame foot heights (before leveling)
    v_mask: np.ndarray  # (nv,) True where v is measured (base lin vel False)
    v_out: np.ndarray = None  # (T, nv) step output velocity; default v[k+1]


def load_walk_diag(csv_path: str, dyn: X1Dynamics) -> WalkData:
    import csv as _csv

    with open(csv_path) as f:
        rdr = _csv.reader(f)
        hdr = next(rdr)
        rows = [r for r in rdr if r]

    col = {h: i for i, h in enumerate(hdr)}
    ts = np.array([float(r[col["timestamp_ns"]]) for r in rows]) * 1e-9
    t = ts - ts[0]
    T = len(rows)

    q = pin.neutral(dyn.model)
    quat = np.array(
        [
            [float(r[col[f"imu_quat_{k}"]]) for k in "xyzw"]
            for r in rows
        ]
    )
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)

    pos = np.zeros((T, 12))
    vel = np.zeros((T, 12))
    eff = np.zeros((T, 12))
    for j, name in enumerate(JOINT_ORDER):
        pos[:, j] = [float(r[col[f"pos_{name}"]]) for r in rows]
        vel[:, j] = [float(r[col[f"vel_{name}"]]) for r in rows]
        eff[:, j] = [float(r[col[f"effort_{name}"]]) for r in rows]

    gyro = np.array(
        [[float(r[col[f"imu_gyro_{k}"]]) for k in "xyz"] for r in rows]
    )

    q_arr = np.tile(q, (T, 1))
    q_arr[:, 3:7] = quat
    q_arr[:, 7:] = pos

    # --- base height from support-foot leveling -----------------------
    # world z of each contact point given base pose with z = 0 reference;
    # then shift base so the lowest foot sits exactly on the ground.
    foot_z_ref = np.zeros((T, dyn.n_contacts))
    for k in range(T):
        pin.forwardKinematics(dyn.model, dyn.data, _set_z(q_arr[k], 0.0))
        pin.updateFramePlacements(dyn.model, dyn.data)
        for i, fid in enumerate(dyn.contact_frame_ids):
            foot_z_ref[k, i] = dyn.data.oMf[fid].translation[2]
    z_base = -foot_z_ref.min(axis=1)  # lowest foot at z=0
    # smooth (walk: swing foot lowest point ~ touchdown, ok per-frame)
    q_arr[:, 2] = z_base

    v = np.zeros((T, dyn.nv))
    v[:, 3:6] = gyro  # LOCAL body angular velocity
    v[:, 6:] = vel

    v_mask = np.ones(dyn.nv, dtype=bool)
    v_mask[:3] = False  # base linear velocity unobserved

    dt = float(np.median(np.diff(t)))
    v_out = np.vstack([v[1:], v[-1:].reshape(1, -1)])
    return WalkData(
        t=t,
        dt=dt,
        q=q_arr,
        v=v,
        u=eff,
        gyro=gyro,
        base_quat=quat,
        foot_z=foot_z_ref,
        v_mask=v_mask,
        v_out=v_out,
    )


def _set_z(q: np.ndarray, z: float) -> np.ndarray:
    qq = q.copy()
    qq[2] = z
    return qq
