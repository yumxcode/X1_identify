"""FD-gradient optimizer on the acceptance metric (G2.1-aligned).

Optimizes theta (90-dim log-Cholesky) by L-BFGS-B with finite-difference
gradients of the acceptance cost (contact-consistent residual, lam re-solved
per candidate). Flat directions (rigid-contact-canceled, e.g. base mass) stay
near the prior thanks to the prior term in the cost.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify")
from prime.dynamics import X1Dynamics
from prime.data import load_walk_diag
from prime.pfie import PFIE, PFIEConfig

URDF = "/Users/yumx/code/robot_x/X1/X1_辨识/X1_train/resources/robots/x1/urdf/x1.urdf"
CSV = "/Users/yumx/code/robot_x/X1/X1_辨识/x1_data/walk_diag_20260824_103222.csv"


def run(mode="sim", n_frames=400, n_eval=80, maxiter=25):
    dyn = X1Dynamics(URDF)
    wd = load_walk_diag(CSV, dyn)
    theta_hat = dyn.theta_hat.copy()
    n = min(n_frames, len(wd.t) - 1)

    if mode == "sim":
        sys.path.insert(0, "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify/scripts")
        from selftest_sim import perturb_pi_from, BODY_NAME, KNEE_L, KNEE_R
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
        v_out_syn[:, 3:6] += rng.normal(0, 0.005, (n, 3))
        wd.v_out = v_out_syn

    pf = PFIE(dyn, wd, PFIEConfig(n_iters=1, holdout=0.0))
    rng_e = np.random.default_rng(1)
    all_idx = np.arange(0, n)
    ev_idx = all_idx[rng_e.choice(n, n_eval, replace=False)]
    # fixed contact-noise seed per candidate: PGS warm start could otherwise
    # add jitter to the cost surface
    w_prior = 3e-2

    def cost(x):
        c = pf._accept_cost(x, ev_idx)
        if not np.isfinite(c):
            return 1e12
        d = x - theta_hat
        return c + w_prior * float(d @ d)

    from scipy.optimize import minimize

    t0 = time.time()
    hist = []

    def cb(xk):
        hist.append(cost(xk))
        print(f"  [{len(hist):2d}] cost={hist[-1]:.2f} mass={dyn.total_mass():.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    res = minimize(
        cost, theta_hat, method="L-BFGS-B",
        options={"maxiter": maxiter, "maxfun": 25 * 95, "eps": 2e-2, "ftol": 1e-4},
        callback=cb,
    )
    theta_id = res.x

    if mode == "sim":
        # recovery report
        pi_gt_g = dyn.theta_to_pi_groups(theta_gt)
        pi_hat_g = dyn.theta_to_pi_groups(theta_hat)
        pi_id_g = dyn.theta_to_pi_groups(theta_id)
        print("\n=== RECOVERY (base + knees) ===")
        names = {0: "root", 4: "left_knee", 5: "right_knee"}
        for gi, nm in names.items():
            rec = pi_id_g[gi, 0] - pi_hat_g[gi, 0]
            tru = pi_gt_g[gi, 0] - pi_hat_g[gi, 0]
            c_err = np.linalg.norm(pi_id_g[gi, 1:4] / pi_id_g[gi, 0]
                                   - pi_gt_g[gi, 1:4] / pi_gt_g[gi, 0]) * 1000
            print(f"{nm:11s} mass rec {rec:+.3f} / true {tru:+.3f} "
                  f"({100*rec/tru:.0f}%)  COM err {c_err:.1f} mm")
    else:
        c0 = pf._accept_cost(theta_hat, all_idx)
        c1 = pf._accept_cost(theta_id, all_idx)
        print(f"\nreal: nominal {c0:.1f} -> identified {c1:.1f} "
              f"({100*(c1/c0-1):+.1f}%)")
        pi_id_g = dyn.theta_to_pi_groups(theta_id)
        pi_hat_g = dyn.theta_to_pi_groups(theta_hat)
        for gi in range(dyn.n_groups):
            dm = pi_id_g[gi, 0] - pi_hat_g[gi, 0]
            if abs(dm) > 0.02:
                print(f"  {dyn.model.names[dyn.bodies[gi]]:26s} dm={dm:+.3f} kg")
    return theta_id


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sim"
    nf = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    mi = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    run(mode, nf, maxiter=mi)
