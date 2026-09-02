"""Window/impulse-form Z-projection estimator (EIV-mitigated).

Integrates the pointwise dynamics over a window of L steps with a constant
gated-contact set:
    Z^T [ M(q~) dv + dt_L * h(q~,v~) - dt_L * B u~ ] = 0
    dv = [dv_b (IMU-integrated, local); dv_j (encoders)]
Regressed via Y(q~, v~, a~=dv/dt_L) * dt_L (linear in p). Window averaging
cuts the effective acceleration noise ~L x vs the instantaneous form, which
tames the errors-in-variables bias that broke the instantaneous estimator.
"""
import os
import sys

import numpy as np
import pinocchio as pin

sys.path.insert(0, "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify")
sys.path.insert(0, "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify/scripts")
from imu_ident import null_basis, G_W, quat_to_R
from prime.dynamics import X1Dynamics

URDF = "/Users/yumx/code/robot_x/X1/X1_辨识/X1_train/resources/robots/x1/urdf/x1.urdf"
CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "raw", "walk_diag_20260824_103222.csv")


def gate_key(dyn, k, phi_cache, thresh):
    return tuple(bool(x) for x in (phi_cache[k] < thresh))


def build_windows(dyn, wd, dv_b, L=10, phi_thresh=0.03, a_sig=(0.35, 0.05),
                  max_windows=4000, seed=0):
    """dv_b: (T,3) IMU-integrated base velocity delta per step-to-step... here
    dv_b[k] = local base velocity change over [k, k+L] (computed by caller).
    Returns design (Yw, rw, sig_rows, p_nom, names130, windows)."""
    n = len(wd.t) - L
    model = dyn.model
    p_nom = np.concatenate(
        [model.inertias[j].toDynamicParameters() for j in range(1, model.njoints)]
    )
    names130 = [model.names[j] for j in range(1, model.njoints)]

    # phi cache
    phi = np.zeros((n + L, dyn.n_contacts))
    for k in range(n + L):
        _, phi[k] = dyn.kinematics(wd.q[k])
    gate = phi < phi_thresh

    sig_dv = np.full(18, a_sig[1])   # dv_j noise rad/s (2-endpoint diff)
    sig_dv[:3] = a_sig[0]            # dv_b noise m/s

    Ys, rs, sig_rows, wins = [], [], [], []
    rng = np.random.default_rng(seed)
    for k in range(0, n, max(1, L // 5)):
        seg = gate[k:k + L + 1]
        if not (seg[0] == seg[-1]).all() or not (seg == seg[0]).all():
            continue  # contact set must be constant across the window
        mid = k + L // 2
        active = seg[0]
        J = np.zeros((3 * dyn.n_contacts, 18))
        for i in range(dyn.n_contacts):
            if active[i]:
                J[3 * i:3 * i + 3] = dyn._Jc[3 * i:3 * i + 3]
        Z = null_basis(J)
        dv = np.zeros(18)
        dv[:3] = dv_b[k]
        dv[6:] = wd.v_out[k + L][6:] - wd.v[k][6:]
        a_win = dv / (L * wd.dt)
        u_win = wd.u[k:k + L + 1].mean(axis=0)
        Y = dyn.regressor(wd.q[mid], wd.v[mid], a_win)
        Bu = np.zeros(18)
        Bu[6:] = u_win
        r = Y @ p_nom - Bu
        Ys.append(Z.T @ Y)
        rs.append(Z.T @ r)
        sig_rows.append(np.sqrt(((Z.T * (sig_dv / (L * wd.dt))[None, :]) ** 2).sum(axis=1)))
        wins.append(k)
        if len(Ys) >= max_windows:
            break
    Ys = np.vstack(Ys)
    rs = np.concatenate(rs)
    sig_rows = np.concatenate(sig_rows)
    print(f"window design: {Ys.shape} from {len(wins)} windows (L={L})")
    return Ys, rs, sig_rows, p_nom, names130, wins


def dvb_from_imu(f_b, quat, L, dt=0.01):
    """Local-frame base velocity change over [k, k+L]: trapezoid of
    (f_b + R^T g_w) — no double integration, no drift."""
    T = len(f_b)
    a_loc = np.zeros((T, 3))
    for k in range(T):
        R = quat_to_R(quat[k])
        a_loc[k] = f_b[k] + R.T @ G_W
    dv_b = np.zeros((T - L, 3))
    for k in range(T - L):
        dv_b[k] = (0.5 * (a_loc[k] + a_loc[k + L]) * L * dt)
    return dv_b
