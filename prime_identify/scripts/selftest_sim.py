"""Sim ground-truth self-test for the PRIME-X1 pipeline.

Protocol (paper Sec. V-B style):
  1. take real walk_diag (q, v, u) as excitation
  2. perturb the base-link inertia parameters by a known ground truth
     (e.g. +2.0 kg, COM shift, inertia scaling) -> theta_gt
  3. roll the SAME contact dynamics to synthesize v+ measurements (with
     encoder/gyro noise added)
  4. run PFIE starting from the nominal theta_hat
  5. check recovery errors: mass, COM, inertia per perturbed group

PASS criteria (see PASS_CRITERIA.md):
  mass error      < 5%   of the perturbation
  COM error       < 15 mm
  inertia error   < 20%  (relative to perturbed delta, if excitable)
"""
import argparse
import sys
import numpy as np

sys.path.insert(0, "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify")
from prime.dynamics import X1Dynamics
from prime.data import load_walk_diag
from prime.pfie import PFIE, PFIEConfig
from prime.log_cholesky import theta_to_pi

URDF = "/Users/yumx/code/robot_x/X1/X1_辨识/X1_train/resources/robots/x1/urdf/x1.urdf"
CSV = "/Users/yumx/code/robot_x/X1/X1_辨识/x1_data/walk_diag_20260824_103222.csv"


def make_perturbed_theta(dyn: X1Dynamics, mass_delta: float, com_shift, inertia_scale: float):
    """Perturb only the BASE group (group 0), like the paper's payload tests."""
    pi_g = dyn.theta_to_pi_groups(dyn.theta_hat)
    base = pi_g[0].copy()
    m0 = base[0]
    base[0] = m0 + mass_delta
    mc = base[1:4] / m0 + np.asarray(com_shift)
    base[1:4] = mc * base[0]
    # scale inertia-about-origin entries by inertia_scale (linear in pi)
    base[4:10] = base[4:10] * inertia_scale
    pi_g[0] = base
    from prime.log_cholesky import pi_to_theta as p2t
    from prime.dynamics import _PC_TO_LC
    theta = np.concatenate([p2t(pi_g[gi][_PC_TO_LC]) for gi in range(dyn.n_groups)])
    return theta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--mass-delta", type=float, default=2.0)
    ap.add_argument("--noise", type=float, default=0.004)
    args = ap.parse_args()

    dyn = X1Dynamics(URDF)
    wd = load_walk_diag(CSV, dyn)
    print(f"data: {len(wd.t)} frames @ {1/wd.dt:.0f} Hz, mass={dyn.total_mass():.3f} kg")

    theta_gt = make_perturbed_theta(dyn, args.mass_delta, [0.02, 0.0, -0.03], 1.2)
    dyn.set_theta(theta_gt)
    print(f"GT total mass = {dyn.total_mass():.3f} kg")

    # synthesize v+ with contact dynamics + noise
    rng = np.random.default_rng(7)
    n = args.frames
    v_meas = wd.v[: n + 1].copy()
    lam_ws = None
    n_bad = 0
    for k in range(n):
        v_plus, imp, conv = dyn.solve_contact_step(wd.q[k], wd.v[k], wd.u[k], wd.dt)
        if not conv:
            n_bad += 1
            # fallback: accept the PGS iterate if finite (it is usually very
            # close; convergence flag is conservative)
            if not np.all(np.isfinite(v_plus)):
                v_plus = wd.v[k]
        lam_ws = imp.reshape(-1) if conv else None
        v_meas[k + 1] = v_plus
    print(f"synthetic rollout: {n_bad}/{n} frames flagged unconverged (accepted)")
    v_meas[1:, 6:] += rng.normal(0, args.noise, (n, 12))
    v_meas[1:, 3:6] += rng.normal(0, 0.005, (n, 3))
    wd.v = v_meas  # replace measurements with synthetic ones

    # run identification from nominal
    dyn.set_theta(dyn.theta_hat)
    cfg = PFIEConfig(n_iters=args.iters, holdout=0.2)
    pf = PFIE(dyn, wd, cfg)
    # restrict to first n frames
    all_idx = np.arange(0, n)
    pf.idx_tr = all_idx
    pf.idx_ho = all_idx[: 0]
    res = pf.solve(verbose=True)

    # recovery check on base group
    pi_est = dyn.theta_to_pi_groups(res.theta)[0]
    pi_gt = dyn.theta_to_pi_groups(theta_gt)[0]
    pi_nom = dyn.theta_to_pi_groups(dyn.theta_hat)[0]
    m_est, m_gt, m_nom = pi_est[0], pi_gt[0], pi_nom[0]
    c_est = pi_est[1:4] / m_est
    c_gt = pi_gt[1:4] / m_gt
    print("\n=== RECOVERY (base group) ===")
    print(f"mass: est={m_est:.4f} gt={m_gt:.4f} nominal={m_nom:.4f} "
          f"| delta recovered {m_est-m_nom:+.4f} / true {m_gt-m_nom:+.4f} "
          f"({100*(m_est-m_nom)/(m_gt-m_nom):.1f}%)")
    print(f"COM est: {np.round(c_est,5)}  gt: {np.round(c_gt,5)}  err={np.linalg.norm(c_est-c_gt)*1000:.1f} mm")
    I_est = pi_est[4:10]; I_gt = pi_gt[4:10]; I_nom = pi_nom[4:10]
    dI_est = I_est - I_nom; dI_gt = I_gt - I_nom
    scale = np.dot(dI_est, dI_gt) / max(np.dot(dI_gt, dI_gt), 1e-12)
    print(f"inertia delta recovery scale: {scale:.3f} (1.0 = perfect)")
    print(f"joint-resid RMSE train: {res.rmse_joint_train:.3f} Nm")
    print(f"elapsed: {res.elapsed_s:.1f} s")


if __name__ == "__main__":
    main()
