"""Ceiling analysis: residual at tied-GT vs fitted params vs unregularized LS.

Determines whether the knee/hip_yaw split is unidentifiable from this data
(residual at GT ~ residual at fit) or the estimator is suboptimal.
"""
import os
import sys
import numpy as np
import pinocchio as pin

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "prime_identify"))
URDF = os.path.join(REPO, "X1_train", "resources", "robots", "x1", "urdf", "x1.urdf")
CSV = os.path.join(REPO, "x1_data", "walk_diag_20260824_103222.csv")
LEG_PAIRS = [
    ("left_hip_pitch_joint", "right_hip_pitch_joint"),
    ("left_hip_roll_joint", "right_hip_roll_joint"),
    ("left_hip_yaw_joint", "right_hip_yaw_joint"),
    ("left_knee_pitch_joint", "right_knee_pitch_joint"),
    ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
    ("left_ankle_roll_joint", "right_ankle_roll_joint"),
]
from prime.dynamics import X1Dynamics, make_param_transform
from prime.data import load_walk_diag
from selftest_sim import perturb_pi_from, KNEE_L


def build(n_frames=1500, swing_thresh=0.025, noise=0.004, seed=7):
    dyn = X1Dynamics(URDF)
    wd = load_walk_diag(CSV, dyn)
    n = min(n_frames, len(wd.t) - 1)
    names130 = [dyn.model.names[j] for j in range(1, dyn.model.njoints)]
    q0 = pin.neutral(dyn.model)
    pin.forwardKinematics(dyn.model, dyn.data, q0)
    pin.updateFramePlacements(dyn.model, dyn.data)
    B_mir = {}
    for ln, rn in LEG_PAIRS:
        jl, jr = dyn.model.getJointId(ln), dyn.model.getJointId(rn)
        B_mir[ln] = make_param_transform(
            dyn.data.oMi[jl].rotation.T @ dyn.data.oMi[jr].rotation
        )
    pi_gt = perturb_pi_from(dyn.pi_nominal, dyn, "root_joint", +2.0, [0.02, 0, -0.03], 1.2)
    pi_gt = perturb_pi_from(pi_gt, dyn, KNEE_L, +0.4, [0.01, 0, 0], 1.1)
    pi_gt = perturb_pi_from(pi_gt, dyn, "right_knee_pitch_joint", +0.4, [0.01, 0, 0], 1.1)
    theta_gt = dyn.pi_to_theta(pi_gt)
    dyn.set_theta(theta_gt)
    rng = np.random.default_rng(seed)
    v_out_syn = np.zeros((n, dyn.nv))
    for k in range(n):
        vp, imp, conv = dyn.solve_contact_step(wd.q[k], wd.v[k], wd.u[k], wd.dt)
        v_out_syn[k] = vp if np.all(np.isfinite(vp)) else wd.v[k]
    v_out_syn[:, 6:] += rng.normal(0, noise, (n, 12))
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

    x_gt = np.zeros(60)
    x_nom = np.zeros(60)
    for bi, (ln, rn) in enumerate(LEG_PAIRS):
        bl = 10 * names130.index(ln)
        x_nom[10 * bi:10 * bi + 10] = p_nom[bl:bl + 10]
        jid = dyn.model.getJointId(ln)
        if jid in dyn.bodies:
            x_gt[10 * bi:10 * bi + 10] = pi_gt[dyn.bodies.index(jid)]
        else:  # excluded bodies are unperturbed (GT == nominal)
            x_gt[10 * bi:10 * bi + 10] = p_nom[bl:bl + 10]
    return Ys, rs, x_gt, x_nom, dyn


def rms(v):
    return float(np.sqrt((np.asarray(v) ** 2).mean()))


if __name__ == "__main__":
    import sys as _s
    noise = float(_s.argv[1]) if len(_s.argv) > 1 else 0.004
    Ys, rs, x_gt, x_nom, dyn = build(noise=noise)
    r_nom = rs
    r_gt = Ys @ (x_gt - x_nom) + rs
    dx_ls, *_ = np.linalg.lstsq(Ys, -rs, rcond=1e-8)
    r_ls = Ys @ dx_ls + rs
    ki = LEG_PAIRS.index((KNEE_L, "right_knee_pitch_joint"))
    hy = LEG_PAIRS.index(("left_hip_yaw_joint", "right_hip_yaw_joint"))
    print(f"noise sigma_v={noise}")
    print(f"resid RMS: nominal {rms(r_nom):.4f} | tied-GT {rms(r_gt):.4f} | unreg-LS {rms(r_ls):.4f}")
    print(f"unreg-LS: knee dm={dx_ls[10*ki]:+.4f} hip_yaw dm={dx_ls[10*hy]:+.4f} (GT: +0.4 / 0)")
    print(f"cond(Ys)={np.linalg.cond(Ys):.2e}")
    # null-space projection of the GT delta: how much of (x_gt-x_nom) lies in
    # the near-null space of Ys (singular values < eps*smax)?
    U, S, Vt = np.linalg.svd(Ys, full_matrices=False)
    d = x_gt - x_nom
    for rel in (1e-3, 1e-2, 5e-2):
        k = int((S > rel * S.max()).sum())
        # component of d in directions with sigma < rel*smax
        coef = Vt @ d
        frac = np.sqrt(np.sum(coef[k:] ** 2) / np.sum(coef ** 2))
        print(f"  |GT-delta| fraction in dirs with sigma<{rel:.0e}*smax: {frac:.3f} (rank {k}/60)")
