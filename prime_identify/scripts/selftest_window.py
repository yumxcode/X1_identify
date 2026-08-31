"""G1'' synthetic self-test: window/impulse Z-projection estimator.

Gate: base & knees mass recovery >=50% and within 0.5 kg (PASS_CRITERIA G1.2).
"""
import sys

import numpy as np

sys.path.insert(0, "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify")
sys.path.insert(0, "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify/scripts")
from imu_ident import wls_prior, box_lsq, report
from imu_window import URDF, CSV, build_windows, dvb_from_imu
from prime.dynamics import X1Dynamics
from prime.data import load_walk_diag
from selftest_sim import perturb_pi_from, BODY_NAME, KNEE_L, KNEE_R


def main(n_frames=1500, L=10, sigma_ab=0.6, sigma_aj=0.004, seed=7):
    dyn = X1Dynamics(URDF)
    wd = load_walk_diag(CSV, dyn)
    n = min(n_frames, len(wd.t) - L)
    names130 = [dyn.model.names[j] for j in range(1, dyn.model.njoints)]

    pi_gt = perturb_pi_from(dyn.pi_nominal, dyn, BODY_NAME, +2.0, [0.02, 0.0, -0.03], 1.2)
    pi_gt = perturb_pi_from(pi_gt, dyn, KNEE_L, +0.4, [0.01, 0.0, 0.0], 1.1)
    pi_gt = perturb_pi_from(pi_gt, dyn, KNEE_R, +0.4, [0.01, 0.0, 0.0], 1.1)
    theta_gt = dyn.pi_to_theta(pi_gt)

    # GT rollout for joint v_out
    dyn.set_theta(theta_gt)
    rng = np.random.default_rng(seed)
    v_out = np.zeros((n + L, dyn.nv))
    a_b_true = np.zeros((n, 3))
    for k in range(n + L - 1):
        vp, imp, conv = dyn.solve_contact_step(wd.q[k], wd.v[k], wd.u[k], wd.dt)
        v_out[k] = vp if np.all(np.isfinite(vp)) else wd.v[k]
        if k < n:
            a_b_true[k] = (v_out[k, :3] - wd.v[k, :3]) / wd.dt
    v_out[n + L - 1] = wd.v[n + L - 1]

    # window base dv: true (from GT) + IMU-level noise
    dv_b_true = np.zeros((n, 3))
    for k in range(n):
        # exact local base dv over the window from GT one-step velocities
        acc = np.zeros(3)
        for kk in range(k, k + L):
            acc = acc + (v_out[kk, :3] - wd.v[kk, :3])
        dv_b_true[k] = acc * wd.dt
    dv_b_meas = dv_b_true + rng.normal(0, sigma_ab, (n, 3))

    # joint v_out + encoder noise
    v_out_meas = v_out.copy()
    v_out_meas[:n, 6:] += rng.normal(0, sigma_aj, (n, 12))

    class WD:  # lightweight view carrying what build_windows needs
        pass

    w = WD()
    w.t, w.q, w.v, w.u, w.dt, w.v_out = wd.t, wd.q, wd.v, wd.u, wd.dt, v_out_meas
    dyn.set_theta(dyn.theta_hat)
    Ys, rs, sig_rows, p_nom, names130, wins = build_windows(
        dyn, w, dv_b_meas, L=L, a_sig=(sigma_ab, sigma_aj * np.sqrt(2))
    )
    dp, _ = box_lsq(Ys, rs, sig_rows, p_nom, names130)
    print(f"resid RMS: nominal {float(np.sqrt((rs**2).mean())):.3f} -> "
          f"fit {float(np.sqrt(((Ys @ dp + rs)**2).mean())):.3f}")
    x_gt = p_nom.copy()
    for name, dm in ((BODY_NAME, 2.0), (KNEE_L, 0.4), (KNEE_R, 0.4)):
        b = names130.index(name)
        gi = dyn.bodies.index(dyn.model.getJointId(name))
        x_gt[10 * b:10 * b + 10] = pi_gt[gi]
    print(f"resid RMS at GT: {float(np.sqrt(((Ys @ (x_gt - p_nom) + rs)**2).mean())):.3f}")
    ok = report(dyn, pi_gt, dp, p_nom, names130,
                ((BODY_NAME, 2.0), (KNEE_L, 0.4), (KNEE_R, 0.4)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
