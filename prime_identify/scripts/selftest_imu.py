"""G1' synthetic self-test for the IMU-accel Z-projection estimator.

GT: base +2.0 kg / COM / I*1.2; knees +0.4 kg each. Synthesized a_b/a_j
include realistic noise. PASS gate: mass recovery >=50% and within 0.5 kg.
"""
import sys

import numpy as np

sys.path.insert(0, "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify")
sys.path.insert(0, "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify/scripts")
from imu_ident import (
    URDF,
    CSV,
    build_regression,
    load_imu_accel,
    report,
    wls_prior,
)
from prime.dynamics import X1Dynamics
from prime.data import load_walk_diag
from selftest_sim import perturb_pi_from, BODY_NAME, KNEE_L, KNEE_R


def main(n_frames=1500, sigma_ab=0.6, sigma_aj=0.5, seed=7):
    dyn = X1Dynamics(URDF)
    wd = load_walk_diag(CSV, dyn)
    n = min(n_frames, len(wd.t) - 1)
    names130 = [dyn.model.names[j] for j in range(1, dyn.model.njoints)]

    pi_gt = perturb_pi_from(dyn.pi_nominal, dyn, BODY_NAME, +2.0, [0.02, 0.0, -0.03], 1.2)
    pi_gt = perturb_pi_from(pi_gt, dyn, KNEE_L, +0.4, [0.01, 0.0, 0.0], 1.1)
    pi_gt = perturb_pi_from(pi_gt, dyn, KNEE_R, +0.4, [0.01, 0.0, 0.0], 1.1)
    theta_gt = dyn.pi_to_theta(pi_gt)

    # synthesize one-step outputs with GT
    dyn.set_theta(theta_gt)
    rng = np.random.default_rng(seed)
    v_out = np.zeros((n, dyn.nv))
    a_b_true = np.zeros((n, 3))
    for k in range(n):
        vp, imp, conv = dyn.solve_contact_step(wd.q[k], wd.v[k], wd.u[k], wd.dt)
        if np.all(np.isfinite(vp)):
            v_out[k] = vp
        else:
            v_out[k] = wd.v[k]
        a_b_true[k] = (v_out[k, :3] - wd.v[k, :3]) / wd.dt
    a_j_meas = (v_out[:, 6:] - wd.v[k + 1 - 1, 6:]) * 0  # placeholder, set below
    a_j_meas = (v_out[:, 6:] - wd.v[:n, 6:]) / wd.dt
    a_j_meas += rng.normal(0, sigma_aj, (n, 12))
    a_b_meas = a_b_true + rng.normal(0, sigma_ab, (n, 3))
    print(f"synth: n={n}, sigma_ab={sigma_ab}, sigma_aj={sigma_aj}, "
          f"|a_b| true mean {np.linalg.norm(a_b_true, axis=1).mean():.2f} m/s^2")

    dyn.set_theta(dyn.theta_hat)
    Ys, rs, sig_rows, p_nom, names130 = build_regression(dyn, wd, a_b_meas, a_j_meas)
    dp, H, Hp = wls_prior(Ys, rs, sig_rows, p_nom)
    print(f"cond(H) = {np.linalg.cond(H):.2e}")
    res_before = float(np.sqrt((rs ** 2).mean()))
    res_after = float(np.sqrt(((Ys @ dp + rs) ** 2).mean()))
    print(f"resid RMS (Z-space): nominal {res_before:.2f} -> fit {res_after:.2f}")
    # GT reference
    x_gt = p_nom.copy()
    for name, dm in ((BODY_NAME, 2.0), (KNEE_L, 0.4), (KNEE_R, 0.4)):
        b = names130.index(name)
        gi = dyn.bodies.index(dyn.model.getJointId(name))
        x_gt[10 * b:10 * b + 10] = pi_gt[gi]
    r_gt = Ys @ (x_gt - p_nom) + rs
    print(f"resid RMS at tied-GT: {float(np.sqrt((r_gt ** 2).mean())):.2f}")
    ok = report(dyn, pi_gt, dp, p_nom, names130,
                ((BODY_NAME, 2.0), (KNEE_L, 0.4), (KNEE_R, 0.4)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
