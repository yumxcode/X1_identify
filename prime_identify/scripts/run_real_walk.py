"""Run PFIE identification on real walk_diag data.

Reports:
  - parameter changes vs (projected) URDF nominal
  - total mass estimate (GRF-balance anchored)
  - torque-consistency residual before/after (train + holdout)
  - FIE-style cost comparison (with vs without identification)
Outputs results to results/real_walk_ident.json
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify")
from prime.dynamics import X1Dynamics
from prime.data import load_walk_diag
from prime.pfie import PFIE, PFIEConfig

URDF = "/Users/yumx/code/robot_x/X1/X1_辨识/X1_train/resources/robots/x1/urdf/x1.urdf"
CSV = "/Users/yumx/code/robot_x/X1/X1_辨识/x1_data/walk_diag_20260824_103222.csv"


def fie_cost(dyn, wd, frames, theta):
    """Contact-consistent torque residual (the paper's FIE measurement term):
    sum ||Y p - Bu - J^T lam/dt||^2 over swing+trusted rows."""
    dyn.set_theta(theta)
    lam_ws = None
    tot = 0.0
    n = 0
    m = np.ones(dyn.nv, dtype=bool)
    m[:3] = False
    for jn in ("left_ankle_pitch", "left_ankle_roll",
               "right_ankle_pitch", "right_ankle_roll"):
        m[dyn.joint_index(jn)] = False
    Fz_sum = []
    for k in frames:
        vp, imp, conv = dyn.solve_contact_step(wd.q[k], wd.v[k], wd.u[k], wd.dt)
        if not conv or not np.all(np.isfinite(imp)):
            continue
        Fz_sum.append(imp[:, 2].sum() / wd.dt)
        a = (wd.v_out[k] - wd.v[k]) / wd.dt
        Y = dyn.regressor(wd.q[k], wd.v[k], a)
        Bu = np.zeros(dyn.nv)
        Bu[dyn.v_joint] = wd.u[k]
        Jc, _ = dyn.kinematics(wd.q[k])
        r = Y @ np.concatenate(
            [dyn.model.inertias[j].toDynamicParameters() for j in range(1, dyn.model.njoints)]
        ) - Bu - Jc.T @ imp.reshape(-1) / wd.dt
        tot += float(np.sum(r[m] ** 2))
        n += 1
    return tot / max(n, 1), float(np.mean(Fz_sum)) if Fz_sum else 0.0, n


def main():
    t0 = time.time()
    dyn = X1Dynamics(URDF)
    wd = load_walk_diag(CSV, dyn)
    dyn.selfcheck()
    print(f"data {len(wd.t)} frames; nominal mass {dyn.total_mass():.3f} kg")

    cfg = PFIEConfig(n_iters=8, holdout=0.3)
    pf = PFIE(dyn, wd, cfg)
    res = pf.solve(verbose=True)

    print("\n=== FIE-style cost (per-frame, trusted rows) ===")
    c_nom_tr, Fz_nom, n1 = fie_cost(dyn, wd, pf.idx_tr, dyn.theta_hat)
    c_id_tr, Fz_id, n2 = fie_cost(dyn, wd, pf.idx_tr, res.theta)
    c_nom_ho, _, n3 = fie_cost(dyn, wd, pf.idx_ho, dyn.theta_hat)
    c_id_ho, _, n4 = fie_cost(dyn, wd, pf.idx_ho, res.theta)
    print(f"train: nominal {c_nom_tr:.1f} -> identified {c_id_tr:.1f} "
          f"({100*(c_id_tr/c_nom_tr-1):+.1f}%)")
    print(f"holdout: nominal {c_nom_ho:.1f} -> identified {c_id_ho:.1f} "
          f"({100*(c_id_ho/c_nom_ho-1):+.1f}%)")
    print(f"mean GRF sum (nominal theta): {Fz_nom:.1f} N "
          f"-> mass estimate {Fz_nom/9.81:.2f} kg (URDF {dyn.total_mass():.2f})")

    # parameter changes
    dyn.set_theta(res.theta)
    pi_est = dyn._read_pi()
    print("\n=== identified vs nominal (mass, kg) ===")
    out = {
        "elapsed_s": time.time() - t0,
        "n_train": len(pf.idx_tr),
        "n_holdout": len(pf.idx_ho),
        "cost_train": {"nominal": c_nom_tr, "identified": c_id_tr},
        "cost_holdout": {"nominal": c_nom_ho, "identified": c_id_ho},
        "grf_mass_estimate_kg": Fz_nom / 9.81,
        "nominal_mass_kg": dyn.total_mass() if False else 35.323,
        "bodies": {},
    }
    for kk, b in enumerate(dyn.bodies):
        name = dyn.model.names[b]
        dm = pi_est[kk, 0] - dyn.pi_nominal[kk, 0]
        out["bodies"][name] = {
            "m_nom": float(dyn.pi_nominal[kk, 0]),
            "m_est": float(pi_est[kk, 0]),
            "dm": float(dm),
        }
        print(f"  {name:26s} {dyn.pi_nominal[kk,0]:7.3f} -> {pi_est[kk,0]:7.3f} "
              f"({dm:+.3f})")
    dyn.set_theta(res.theta)
    out["total_mass_est_kg"] = dyn.total_mass()
    import os
    os.makedirs("/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify/results", exist_ok=True)
    with open("/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify/results/real_walk_ident.json", "w") as f:
        json.dump({"result": out,
                   "theta": res.theta.tolist(),
                   "theta_hat": dyn.theta_hat.tolist()}, f, indent=1)
    print(f"\nsaved results/real_walk_ident.json | total time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
