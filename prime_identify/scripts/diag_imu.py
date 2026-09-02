"""IMU accelerometer semantics check: reconstruct base linear acceleration.

Specific force convention: f_b = R^T (a_world - g_world), g_world=[0,0,-9.81]
=> a_local (pinocchio freeflyer tangent) ~= f_b + R^T g_w  (ignoring w x v)
Sanity: at rest a_local ~ 0; during walking |a| ~ O(1-3) m/s^2.
"""
import os
import sys

import numpy as np
import pinocchio as pin

sys.path.insert(0, "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify")
from prime.data import load_walk_diag
from prime.dynamics import X1Dynamics

URDF = "/Users/yumx/code/robot_x/X1/X1_辨识/X1_train/resources/robots/x1/urdf/x1.urdf"
CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "raw", "walk_diag_20260824_103222.csv")


def quat_to_R(quat_xyzw):
    q = np.asarray(quat_xyzw, dtype=float)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main():
    import csv as _csv

    with open(CSV) as f:
        r = _csv.reader(f)
        hdr = next(r)
        rows = list(r)
    col = {h: i for i, h in enumerate(hdr)}
    T = len(rows)
    f_b = np.array([[float(x[col[f"imu_accel_{k}"]]) for k in "xyz"] for x in rows])
    quat = np.array([[float(x[col[f"imu_quat_{k}"]]) for k in "xyzw"] for x in rows])
    g_w = np.array([0.0, 0.0, -9.81])

    a_loc = np.zeros((T, 3))
    for k in range(T):
        R = quat_to_R(quat[k])
        a_loc[k] = f_b[k] + R.T @ g_w  # d(v_local)/dt estimate (w x v neglected)

    print(f"T={T}")
    print(f"a_local mean: {np.round(a_loc.mean(axis=0), 3)}")
    print(f"a_local std : {np.round(a_loc.std(axis=0), 3)}")
    print(f"|a_local| mean {np.linalg.norm(a_loc, axis=1).mean():.3f} "
          f"p95 {np.percentile(np.linalg.norm(a_loc, axis=1), 95):.3f} m/s^2")
    # HF noise level: diff std
    print(f"diff(a_local) std: {np.round(np.diff(a_loc, axis=0).std(axis=0), 3)}")
    # step-frequency content: simple check of az periodicity
    az = a_loc[:, 2] - a_loc[:, 2].mean()
    F = np.abs(np.fft.rfft(az))
    freqs = np.fft.rfftfreq(len(az), 0.01)
    k_peak = np.argmax(F[1:]) + 1
    print(f"az dominant freq: {freqs[k_peak]:.2f} Hz (expect ~1-2 Hz step rate)")
    np.save("/tmp/a_local_raw.npy", a_loc)
    np.save("/tmp/f_b.npy", f_b)
    np.save("/tmp/quat.npy", quat)


if __name__ == "__main__":
    main()
