"""IMU-accel-enabled identification: the revival channel.

Key insight: walk_diag contains imu_accel (specific force), which yields the
base linear acceleration a_b — the independent base measurement that breaks
the stance lambda-pi cancellation (this is the paper's mocap information,
available in our data all along).

Estimator (exact lambda elimination):
  Full observed acceleration a = [a_b (IMU, local tangent); a_j (encoders)].
  Dynamics: M(q) a + h(q, v_est) - B u = J_active^T lambda
  Project with Z_k = orthonormal basis of null(J_active)  =>  Z^T J^T = 0:
      Z^T [Y(q, v_est, a) p - B u] = 0
  Linear in p; WLS with per-parameter prior. No contact QP in the loop.

Notes:
  - Including a falsely-gated contact only shrinks Z (fewer equations,
    still exact); missing a true contact invalidates equations -> gate
    conservatively (generous threshold).
  - v_est sets v_b = 0 (unmeasured; Coriolis coupling ~0.1-0.25 m/s^2, part
    of process noise).
  - a_b = f_b + R^T g_w (w x v term neglected, ~0.25 m/s^2, process noise).
"""
import os
import sys

import numpy as np
import pinocchio as pin

sys.path.insert(0, "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify")
sys.path.insert(0, "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify/scripts")
from prime.dynamics import X1Dynamics
from prime.data import load_walk_diag

URDF = "/Users/yumx/code/robot_x/X1/X1_辨识/X1_train/resources/robots/x1/urdf/x1.urdf"
CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "raw", "walk_diag_20260824_103222.csv")

G_W = np.array([0.0, 0.0, -9.81])


def quat_to_R(quat_xyzw):
    q = np.asarray(quat_xyzw, dtype=float)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_imu_accel(csv_path, fc=10.0):
    """Raw specific force + quats; returns filtered a_local (T,3)."""
    import csv as _csv

    from scipy.signal import butter, filtfilt

    with open(csv_path) as f:
        r = _csv.reader(f)
        hdr = next(r)
        rows = list(r)
    col = {h: i for i, h in enumerate(hdr)}
    f_b = np.array([[float(x[col[f"imu_accel_{k}"]]) for k in "xyz"] for x in rows])
    quat = np.array([[float(x[col[f"imu_quat_{k}"]]) for k in "xyzw"] for x in rows])
    T = len(rows)
    fs = 100.0
    b, a = butter(4, fc / (fs / 2))
    a_loc = np.zeros((T, 3))
    for k in range(T):
        R = quat_to_R(quat[k])
        a_loc[k] = f_b[k] + R.T @ G_W
    for d in range(3):
        a_loc[:, d] = filtfilt(b, a, a_loc[:, d])
    return a_loc, f_b, quat


def null_basis(J, rtol=1e-6):
    """Orthonormal basis of null(J) (rows of Z satisfy J Z = 0)."""
    if J.size == 0:
        return np.eye(18)
    _, S, Vt = np.linalg.svd(J)
    tol = max(J.shape) * (S[0] if len(S) else 0.0) * rtol
    rank = int((S > tol).sum())
    return Vt[rank:].T


def build_regression(dyn, wd, a_b_meas, a_j_meas, phi_thresh=0.03,
                     frames=None, verbose=True):
    """Z-projection design over frames. a_b_meas: (T,3) local base accel."""
    n = len(wd.t) - 1 if frames is None else len(frames)
    frames = np.arange(n) if frames is None else np.asarray(frames)
    model = dyn.model
    p_nom = np.concatenate(
        [model.inertias[j].toDynamicParameters() for j in range(1, model.njoints)]
    )
    names130 = [model.names[j] for j in range(1, model.njoints)]

    Ys, rs, sig_rows = [], [], []
    v_est = wd.v  # v_b=0 placeholder, gyro + qd measured
    sig_a = np.full(18, 0.5)   # joint accel noise rad/s^2
    sig_a[:3] = 0.8            # base accel noise (IMU filtered) m/s^2
    for k in frames:
        a_full = np.zeros(18)
        a_full[:3] = a_b_meas[k]
        a_full[6:] = a_j_meas[k]
        Y = dyn.regressor(wd.q[k], v_est[k], a_full)
        Bu = np.zeros(18)
        Bu[6:] = wd.u[k]
        r = Y @ p_nom - Bu
        # contact gating from leveled kinematics
        _, phi_raw = dyn.kinematics(wd.q[k])
        active = phi_raw < phi_thresh
        J = np.zeros((3 * dyn.n_contacts, 18))
        for i in range(dyn.n_contacts):
            if active[i]:
                J[3 * i:3 * i + 3] = dyn._Jc[3 * i:3 * i + 3]
        Z = null_basis(J)          # (Zdim, 18)
        Ys.append(Z.T @ Y)
        rs.append(Z.T @ r)
        sig_rows.append(np.sqrt(((Z.T * sig_a[None, :]) ** 2).sum(axis=1)))
    Ys = np.vstack(Ys)
    rs = np.concatenate(rs)
    sig_rows = np.concatenate(sig_rows)
    if verbose:
        print(f"Z-projection design: {Ys.shape}")
    return Ys, rs, sig_rows, p_nom, names130


def wls_prior(Ys, rs, row_sig, p_nom, prior_rel=(0.30, 0.012, 0.30)):
    """Heteroscedastic row weights + per-parameter prior; returns dp (130,).

    sig_a: (base-dof noise, joint-dof noise). Needs the *unprojected* row
    structure; here we approximate row noise from the Z-weighted accel
    covariance passed via sig_a_len (n_dofs=18) — fallback: uniform floor.
    """
    W = 1.0 / np.maximum(row_sig, 1e-6)
    Aw = Ys * W[:, None]
    bw = rs * W
    H = Aw.T @ Aw
    g = Aw.T @ (-bw)
    # prior (centered at zero delta = nominal)
    sig_p = np.zeros(130)
    for b in range(13):
        m0 = max(p_nom[10 * b], 1e-3)
        sig_p[10 * b] = prior_rel[0] * m0
        sig_p[10 * b + 1:10 * b + 4] = m0 * prior_rel[1]
        sig_p[10 * b + 4:10 * b + 10] = (
            prior_rel[2] * np.abs(p_nom[10 * b + 4:10 * b + 10]).max() + 1e-4
        )
    Hp = np.diag(1.0 / sig_p ** 2)
    dp = np.linalg.solve(H + Hp, g - Hp @ np.zeros(130))
    return dp, H, Hp


def report(dyn, pi_gt, dp, p_nom, names130, targets):
    p_est = p_nom + dp
    ok = True
    print("=== RECOVERY (Z-projection + IMU accel) ===")
    for name, true_dm in targets:
        b = names130.index(name)
        m_est = p_est[10 * b]
        m_nom = p_nom[10 * b]
        rec = m_est - m_nom
        status = "PASS" if (rec >= 0.5 * true_dm - 1e-9 and abs(rec - true_dm) <= 0.5) else "FAIL"
        ok &= status == "PASS"
        c_est = p_est[10 * b + 1:10 * b + 4] / m_est
        c_nom = p_nom[10 * b + 1:10 * b + 4] / m_nom
        print(f"{name:26s} mass {m_nom:+.3f} -> {m_est:+.3f} "
              f"(true {m_nom + true_dm:+.3f}, recovered {rec:+.3f} = "
              f"{100 * rec / true_dm:.0f}%)  dCOM_vs_nom {np.linalg.norm(c_est - c_nom) * 1000:.1f} mm "
              f"[{status}]")
    print("G1'", "PASS" if ok else "FAIL")
    return ok


def box_lsq(Ys, rs, row_sig, p_nom, names130, prior_rel=(0.30, 0.012, 0.30),
            mass_box=2.5, com_box=0.03, inertia_rel=0.6):
    """Box-constrained WLS with prior rows: physical bounds from CAD/weighing
    (mass +/-2.5 kg incl. payload, COM +/-3 cm, inertia +/-60%). The GT sits
    INSIDE the box; the estimator is not told where."""
    from scipy.optimize import lsq_linear

    W = 1.0 / np.maximum(row_sig, 1e-6)
    Aw = Ys * W[:, None]
    bw = rs * W
    # prior rows (relative to nominal)
    sig_p = np.zeros(130)
    lo = np.full(130, -np.inf)
    hi = np.full(130, np.inf)
    for b in range(13):
        m0 = max(p_nom[10 * b], 1e-3)
        sig_p[10 * b] = prior_rel[0] * m0
        sig_p[10 * b + 1:10 * b + 4] = m0 * prior_rel[1]
        sig_p[10 * b + 4:10 * b + 10] = (
            prior_rel[2] * np.abs(p_nom[10 * b + 4:10 * b + 10]).max() + 1e-4
        )
        # physical boxes
        lo[10 * b] = -mass_box
        hi[10 * b] = mass_box
        lo[10 * b + 1:10 * b + 4] = -com_box * m0
        hi[10 * b + 1:10 * b + 4] = com_box * m0
        imax = np.abs(p_nom[10 * b + 4:10 * b + 10]).max() + 1e-4
        lo[10 * b + 4:10 * b + 10] = -inertia_rel * imax
        hi[10 * b + 4:10 * b + 10] = inertia_rel * imax
    # prior rows appended (soft), boxes hard
    Ap = np.diag(1.0 / sig_p)
    A_aug = np.vstack([Aw, Ap])
    b_aug = np.concatenate([bw, np.zeros(130)])
    res = lsq_linear(A_aug, b_aug, bounds=(lo, hi), tol=1e-10, max_iter=200)
    return res.x, res
