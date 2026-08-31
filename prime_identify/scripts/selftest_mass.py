"""Mass-only scalar identification via stance Z-equations + IMU base mass.

Strategy (EIV-robust):
  1. base mass: scalar LS on the base-mass column (IMU accel channel)
  2. leg masses: per-body scalar LS on stance Z-equations, gravity columns
     (each body's mass enters as m*g*lever — strong, coherent signal)
COM/inertia remain at nominal (documented unidentifiable from this data).
"""
import sys

import numpy as np

sys.path.insert(0, "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify")
sys.path.insert(0, "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify/scripts")
from imu_ident import URDF, CSV, build_regression, report
from imu_window import build_windows
from prime.dynamics import X1Dynamics
from prime.data import load_walk_diag
from selftest_sim import perturb_pi_from, BODY_NAME, KNEE_L, KNEE_R

BODIES13 = None


def main(n_frames=1500, L=10, sigma_ab=0.6, sigma_aj=0.004, seed=7):
    dyn = X1Dynamics(URDF)
    wd = load_walk_diag(CSV, dyn)
    n = min(n_frames, len(wd.t) - L)
    names130 = [dyn.model.names[j] for j in range(1, dyn.model.njoints)]

    pi_gt = perturb_pi_from(dyn.pi_nominal, dyn, BODY_NAME, +2.0, [0.02, 0.0, -0.03], 1.2)
    pi_gt = perturb_pi_from(pi_gt, dyn, KNEE_L, +0.4, [0.01, 0.0, 0.0], 1.1)
    pi_gt = perturb_pi_from(pi_gt, dyn, KNEE_R, +0.4, [0.01, 0.0, 0.0], 1.1)
    theta_gt = dyn.pi_to_theta(pi_gt)

    dyn.set_theta(theta_gt)
    rng = np.random.default_rng(seed)
    v_out = np.zeros((n + L, dyn.nv))
    a_b_true = np.zeros((n, 3))
    for k in range(n + L - 1):
        vp, imp, conv = dyn.solve_contact_step(wd.q[k], wd.v[k], wd.u[k], wd.dt)
        v_out[k] = vp if np.all(np.isfinite(vp)) else wd.v[k]
        if k < n:
            a_b_true[k] = (v_out[k, :3] - wd.v[k, :3]) / wd.dt

    dv_b_meas = np.zeros((n, 3))
    for k in range(n):
        acc = np.zeros(3)
        for kk in range(k, k + L):
            acc = acc + (v_out[kk, :3] - wd.v[kk, :3])
        dv_b_meas[k] = acc * wd.dt
    dv_b_meas += rng.normal(0, sigma_ab, (n, 3))

    v_out_meas = v_out.copy()
    v_out_meas[:n, 6:] += rng.normal(0, sigma_aj, (n, 12))

    class WD:
        pass

    w = WD()
    w.t, w.q, w.v, w.u, w.dt, w.v_out = wd.t, wd.q, wd.v, wd.u, wd.dt, v_out_meas
    dyn.set_theta(dyn.theta_hat)
    Ys, rs, sig_rows, p_nom, names130, wins = build_windows(
        dyn, w, dv_b_meas, L=L, a_sig=(sigma_ab, sigma_aj * np.sqrt(2))
    )

    # mass columns only (13)
    mcols = [10 * b for b in range(13)]
    Ym = Ys[:, mcols]
    rm = Ym.sum(axis=1)  # not used; per-body scalars below

    print("=== per-body scalar mass LS (window Z-equations) ===")
    est = {}
    for b in range(13):
        col = Ym[:, b]
        denom = col @ col
        if denom < 1e-8:
            est[b] = 0.0
            continue
        dm = -(col @ rs) / denom
        # physical box +-2.5 kg (base incl payload), +-0.75 legs
        box = 2.5 if b == 0 else 0.75
        est[b] = float(np.clip(dm, -box, box))
    ok = True
    print(f"{'body':26s} {'dm_est':>8s} {'dm_true':>8s}  status")
    for b in range(13):
        name = names130[b]
        gi = dyn.bodies.index(dyn.model.getJointId(name)) if name in [dyn.model.names[j] for j in dyn.bodies] else None
        true_dm = 0.0
        if name == BODY_NAME:
            true_dm = 2.0
        elif name in (KNEE_L, KNEE_R):
            true_dm = 0.4
        dm = est[b]
        if true_dm > 0:
            status = "PASS" if abs(dm - true_dm) <= 0.5 and dm >= 0.5 * true_dm else "FAIL"
            ok &= status == "PASS"
        else:
            status = ""
        print(f"{name:26s} {dm:+8.3f} {true_dm:+8.3f}  {status}")
    print("G1''(mass-only)", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
