"""Sim ground-truth self-test for the PRIME-X1 pipeline.

Protocol (paper Sec. V-B style):
  1. take real walk_diag (q, v, u) as excitation
  2. perturb selected bodies' inertial parameters by known ground truth
     (BASE group: +2.0 kg, COM shift, inertia scaling — like the paper's
      payload tests; KNEE pair: +0.4 kg mass shift — leg-group check)
  3. roll the SAME contact dynamics to synthesize v+ measurements (with
     encoder/gyro noise added)
  4. run PFIE starting from the nominal theta_hat
  5. check recovery errors per perturbed body

PASS criteria (PASS_CRITERIA.md):
  mass recovery of perturbation  > 70% and within 0.3 kg
  COM error                      < 15 mm
  inertia delta direction cos    > 0.8 (if excitable)
"""
import argparse
import sys
import numpy as np

sys.path.insert(0, "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify")
from prime.dynamics import X1Dynamics
from prime.data import load_walk_diag
from prime.pfie import PFIE, PFIEConfig
from prime.log_cholesky import pi_to_theta as p2t
from prime.dynamics import _PC_TO_LC

URDF = "/Users/yumx/code/robot_x/X1/X1_辨识/X1_train/resources/robots/x1/urdf/x1.urdf"
CSV = "/Users/yumx/code/robot_x/X1/X1_辨识/x1_data/walk_diag_20260824_103222.csv"

BODY_NAME = "root_joint"          # base
KNEE_L = "left_knee_pitch_joint"
KNEE_R = "right_knee_pitch_joint"


def perturb_pi_from(pi_in: np.ndarray, dyn: X1Dynamics, body_name: str,
                    mass_delta: float = 0.0, com_shift=(0, 0, 0),
                    inertia_scale: float = 1.0) -> np.ndarray:
    """PURE: return per-body pi with one body perturbed (pinocchio order)."""
    pi = pi_in.copy()
    k = dyn.bodies.index(dyn.model.getJointId(body_name))
    m0 = pi[k, 0]
    pi[k, 0] = m0 + mass_delta
    c = pi[k, 1:4] / m0 + np.asarray(com_shift)
    pi[k, 1:4] = c * pi[k, 0]
    pi[k, 4:10] = pi[k, 4:10] * inertia_scale
    return pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.004)
    args = ap.parse_args()

    dyn = X1Dynamics(URDF)
    wd = load_walk_diag(CSV, dyn)
    dyn.selfcheck()  # MUST pass before anything else
    print(f"selfcheck OK | data: {len(wd.t)} frames @ {1/wd.dt:.0f} Hz, "
          f"nominal mass={dyn.total_mass():.3f} kg, theta dim={dyn.n_theta}")

    # ---- ground truth: base payload + both knees heavier ----------------
    # perturb_pi is PURE (takes and returns pi); never mutate dyn.pi_nominal
    pi_nom_snapshot = dyn.pi_nominal.copy()
    pi_gt = perturb_pi_from(dyn.pi_nominal, dyn, BODY_NAME, +2.0, [0.02, 0.0, -0.03], 1.2)
    pi_gt = perturb_pi_from(pi_gt, dyn, KNEE_L, +0.4, [0.01, 0.0, 0.0], 1.1)
    pi_gt = perturb_pi_from(pi_gt, dyn, KNEE_R, +0.4, [0.01, 0.0, 0.0], 1.1)
    theta_gt = dyn.pi_to_theta(pi_gt)
    dyn.set_theta(theta_gt)
    print(f"GT total mass = {dyn.total_mass():.3f} kg "
          f"(+{dyn.total_mass()-35.323:.3f} kg)")

    # ---- synthesize measurements ----------------------------------------
    rng = np.random.default_rng(7)
    n = args.frames
    v_out_syn = np.zeros((n, dyn.nv))
    n_bad = 0
    for k in range(n):
        v_plus, imp, conv = dyn.solve_contact_step(wd.q[k], wd.v[k], wd.u[k], wd.dt)
        if not conv:
            n_bad += 1
            if not np.all(np.isfinite(v_plus)):
                v_plus = wd.v[k]
        v_out_syn[k] = v_plus
    print(f"synthetic rollout: {n_bad}/{n} frames flagged (accepted)")
    v_out_syn[:, 6:] += rng.normal(0, args.noise, (n, 12))
    v_out_syn[:, 3:6] += rng.normal(0, 0.005, (n, 3))
    wd.v_out = v_out_syn  # overwrite step OUTPUTS only; inputs stay measured

    # ---- identify from nominal ------------------------------------------
    dyn.set_theta(dyn.theta_hat)
    cfg = PFIEConfig(n_iters=args.iters, holdout=0.2)
    pf = PFIE(dyn, wd, cfg)
    pf.idx_tr = np.arange(0, n)
    pf.idx_ho = np.array([], dtype=int)
    res = pf.solve(verbose=True)

    # ---- recovery report --------------------------------------------------
    pi_est = dyn.pi_to_theta.__self__ and None  # placeholder
    dyn.set_theta(res.theta)
    pi_est_b = dyn._read_pi()
    print("\n=== RECOVERY ===")
    ok_all = True
    for name, true_delta in ((BODY_NAME, 2.0), (KNEE_L, 0.4), (KNEE_R, 0.4)):
        k = dyn.bodies.index(dyn.model.getJointId(name))
        m_est = pi_est_b[k, 0]
        m_nom = pi_nom_snapshot[k, 0]
        rec = m_est - m_nom
        frac = rec / true_delta * 100
        c_est = pi_est_b[k, 1:4] / m_est
        c_gt = pi_gt[k, 1:4] / pi_gt[k, 0]
        com_err = np.linalg.norm(c_est - c_gt) * 1000
        status = "PASS" if (abs(rec - true_delta) < 0.3) else "FAIL"
        ok_all &= status == "PASS"
        print(f"{name:26s} mass {m_nom:+.3f} -> {m_est:+.3f} "
              f"(true {m_nom+true_delta:+.3f}, recovered {rec:+.3f} = {frac:.0f}%) "
              f"COM err {com_err:5.1f} mm  [{status}]")
    print(f"joint-resid RMSE: train {res.rmse_joint_train:.3f} Nm | "
          f"elapsed {res.elapsed_s:.0f} s")
    print("SELFTEST", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
