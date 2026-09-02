"""Swing-channel regression v3: full data + Bayesian prior weighting.

- All 1500 frames (swing subsets ~3.7x more data than n=400 probe)
- Per-parameter prior scales (mass 30%, mc 1 cm-equivalent, inertia 30%)
  -> directions with weak data regress to nominal instead of exploding
- Mirror-tied legs (linear map in pi space)
"""
import os
import sys
import numpy as np
import pinocchio as pin

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "prime_identify"))
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "prime_identify"))
from prime.dynamics import X1Dynamics, make_param_transform
from prime.data import load_walk_diag
from selftest_sim import perturb_pi_from, BODY_NAME, KNEE_L, KNEE_R

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
URDF = os.path.join(REPO, "X1_train", "resources", "robots", "x1", "urdf", "x1.urdf")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV = os.path.join(REPO, "data/raw", "walk_diag_20260824_103222.csv")

LEG_PAIRS = [
    ("left_hip_pitch_joint", "right_hip_pitch_joint"),
    ("left_hip_roll_joint", "right_hip_roll_joint"),
    ("left_hip_yaw_joint", "right_hip_yaw_joint"),
    ("left_knee_pitch_joint", "right_knee_pitch_joint"),
    ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
    ("left_ankle_roll_joint", "right_ankle_roll_joint"),
]


def main(n_frames=1500, swing_thresh=0.025, verbose=True):
    dyn = X1Dynamics(URDF)
    wd = load_walk_diag(CSV, dyn)
    n = min(n_frames, len(wd.t) - 1)
    names130 = [dyn.model.names[j] for j in range(1, dyn.model.njoints)]

    q0 = pin.neutral(dyn.model)
    pin.forwardKinematics(dyn.model, dyn.data, q0)
    pin.updateFramePlacements(dyn.model, dyn.data)
    B_mir = {}
    for ln, rn in LEG_PAIRS:
        jl = dyn.model.getJointId(ln)
        jr = dyn.model.getJointId(rn)
        R_rel = dyn.data.oMi[jl].rotation.T @ dyn.data.oMi[jr].rotation
        B_mir[ln] = make_param_transform(R_rel)

    pi_gt = perturb_pi_from(dyn.pi_nominal, dyn, BODY_NAME, +2.0, [0.02, 0.0, -0.03], 1.2)
    pi_gt = perturb_pi_from(pi_gt, dyn, KNEE_L, +0.4, [0.01, 0.0, 0.0], 1.1)
    pi_gt = perturb_pi_from(pi_gt, dyn, KNEE_R, +0.4, [0.01, 0.0, 0.0], 1.1)
    theta_gt = dyn.pi_to_theta(pi_gt)

    dyn.set_theta(theta_gt)
    rng = np.random.default_rng(7)
    v_out_syn = np.zeros((n, dyn.nv))
    for k in range(n):
        vp, imp, conv = dyn.solve_contact_step(wd.q[k], wd.v[k], wd.u[k], wd.dt)
        v_out_syn[k] = vp if np.all(np.isfinite(vp)) else wd.v[k]
    v_out_syn[:, 6:] += rng.normal(0, 0.004, (n, 12))

    dyn.set_theta(dyn.theta_hat)
    swingL = np.zeros(n, bool)
    swingR = np.zeros(n, bool)
    for k in range(n):
        pin.forwardKinematics(dyn.model, dyn.data, wd.q[k])
        pin.updateFramePlacements(dyn.model, dyn.data)
        zl = min(dyn.data.oMf[f].translation[2] for f in dyn.contact_frame_ids[:2])
        zr = min(dyn.data.oMf[f].translation[2] for f in dyn.contact_frame_ids[2:])
        swingL[k] = zl > swing_thresh
        swingR[k] = zr > swing_thresh
    if verbose:
        print(f"frames {n}: swingL={swingL.sum()} swingR={swingR.sum()}")

    p_nom = np.concatenate(
        [dyn.model.inertias[j].toDynamicParameters() for j in range(1, dyn.model.njoints)]
    )

    def block_cols(side):
        C = np.zeros((130, 60))
        for bi, (ln, rn) in enumerate(LEG_PAIRS):
            bl = 10 * names130.index(ln)
            br = 10 * names130.index(rn)
            if side == "L":
                C[bl:bl + 10, 10 * bi:10 * bi + 10] = np.eye(10)
            else:
                C[br:br + 10, 10 * bi:10 * bi + 10] = B_mir[ln]
        return C

    CL, CR = block_cols("L"), block_cols("R")
    x_nom = np.zeros(60)
    for bi, (ln, rn) in enumerate(LEG_PAIRS):
        bl = 10 * names130.index(ln)
        x_nom[10 * bi:10 * bi + 10] = p_nom[bl:bl + 10]

    Ys, rs = [], []
    for k in range(n):
        a = (v_out_syn[k] - wd.v[k]) / wd.dt
        Y = dyn.regressor(wd.q[k], wd.v[k], a)
        Bu = np.zeros(dyn.nv)
        Bu[6:] = wd.u[k]
        r = Y @ p_nom - Bu
        if swingL[k]:
            Ys.append(Y[list(range(6, 12))] @ CL)
            rs.append(r[list(range(6, 12))])
        if swingR[k]:
            Ys.append(Y[list(range(12, 18))] @ CR)
            rs.append(r[list(range(12, 18))])
    Ys = np.vstack(Ys)
    rs = np.concatenate(rs)
    n_rows = Ys.shape[0]

    # row noise scale: sigma_r per row ~ ||dtau/da|| * sigma_a; approximate
    # uniformly by residual MAD later; first pass unweighted with prior
    # prior scales in x-space (tied left params)
    sig = np.zeros(60)
    for bi, (ln, rn) in enumerate(LEG_PAIRS):
        bl = 10 * names130.index(ln)
        m0 = p_nom[bl]
        sig[10 * bi] = 0.30 * m0                    # mass
        sig[10 * bi + 1:10 * bi + 4] = m0 * 0.012   # mc (COM ~1.2 cm)
        sig[10 * bi + 4:10 * bi + 10] = 0.30 * np.abs(p_nom[bl + 4:bl + 10]).max() + 1e-4

    W_p = np.diag(1.0 / sig**2)
    H = Ys.T @ Ys + W_p
    g = Ys.T @ (-rs)
    dx = np.linalg.solve(H, g)
    x_est = x_nom + dx

    # report
    res_before = rs
    res_after = Ys @ dx + rs
    if verbose:
        print(f"rows={n_rows}  resid RMS: before {np.sqrt((res_before**2).mean()):.3f} -> after {np.sqrt((res_after**2).mean()):.3f} Nm")
        ki = LEG_PAIRS.index((KNEE_L, KNEE_R))
        dm = dx[10 * ki]
        print(f"TIED knee dm = {dm:+.4f} kg (true +0.4 per side)")
        for bi, (ln, rn) in enumerate(LEG_PAIRS):
            print(f"  {ln:26s} dm={dx[10*bi]:+.4f}")
        xl = x_est[10 * ki:10 * ki + 10]
        c_est = xl[1:4] / xl[0]
        gi = dyn.bodies.index(dyn.model.getJointId(KNEE_L))
        c_gt = pi_gt[gi, 1:4] / pi_gt[gi, 0]
        print(f"  knee COM err: {np.linalg.norm(c_est - c_gt) * 1000:.1f} mm")
        # inertia delta direction cosine
        dI_est = x_est[10*ki+4:10*ki+10] - x_nom[10*ki+4:10*ki+10]
        dI_gt = pi_gt[gi, 4:10] - dyn.pi_nominal[gi, 4:10]
        cos = dI_est @ dI_gt / (np.linalg.norm(dI_est) * np.linalg.norm(dI_gt) + 1e-12)
        print(f"  knee inertia delta dir cos: {cos:.3f}")
    return dyn, pi_gt, x_est, x_nom, B_mir


if __name__ == "__main__":
    main()
