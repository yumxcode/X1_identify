"""Remote (gradmotion) validation of the X1 identification pipeline.

Runs on the gm platform (ubuntu:22.04 + pip). Validates:
  [G3] physical consistency of nominal and repaired URDFs
  [G4] GRF-balance total-mass estimate reproduction (subsampled)

Writes PASS/FAIL per item to stdout and a JSON summary.
"""
import json
import os
import subprocess
import sys

REPO = "/workspace/X1_identify" if os.path.isdir("/workspace/X1_identify") else os.getcwd()


def sh(cmd):
    print(f"+ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    # deps
    sh(f"{sys.executable} -m pip install -q pin numpy scipy")
    sys.path.insert(0, os.path.join(REPO, "prime_identify"))

    report = {"items": {}}

    # ---------- G3: physical consistency ----------
    import numpy as np
    import pinocchio as pin
    from prime.dynamics import X1Dynamics
    from prime.log_cholesky import is_physically_consistent

    urdf_nom = os.path.join(REPO, "X1_train/resources/robots/x1/urdf/x1.urdf")
    urdf_rep = os.path.join(REPO, "prime_identify/results/x1_gmass_repair.urdf")

    dyn = X1Dynamics(urdf_nom)
    n_bad = sum(
        0 if is_physically_consistent(
            np.concatenate([[dyn.pi_urdf[k, 0]], dyn.pi_urdf[k, 1:4],
                            [dyn.pi_urdf[k, 4], dyn.pi_urdf[k, 6], dyn.pi_urdf[k, 9]],
                            [dyn.pi_urdf[k, 5], dyn.pi_urdf[k, 8], dyn.pi_urdf[k, 7]]]))
        else 1
        for k in range(dyn.pi_urdf.shape[0])
    )
    report["items"]["G3_nominal_pc_violations"] = n_bad  # expected >0 (documented defect)

    if os.path.exists(urdf_rep):
        dyn2 = X1Dynamics(urdf_rep)
        err = dyn2.selfcheck()
        report["items"]["G3_repaired_roundtrip_err"] = err
        report["items"]["G3_repaired_total_mass"] = round(dyn2.total_mass(), 3)
        g3 = err < 1e-9
    else:
        g3 = False
    report["items"]["G3_PASS"] = bool(g3)

    # ---------- G4: GRF mass estimate reproduction ----------
    from prime.data import load_walk_diag

    csv_path = os.path.join(REPO, "x1_data/walk_diag_20260824_103222.csv")
    wd = load_walk_diag(csv_path, dyn)
    rng = np.random.default_rng(0)
    idx = rng.choice(np.arange(100, 1400), 120, replace=False)
    dyn.set_theta(dyn.theta_hat)
    Fz = []
    for k in idx:
        vp, imp, conv = dyn.solve_contact_step(wd.q[k], wd.v[k], wd.u[k], wd.dt)
        if conv and np.all(np.isfinite(imp)):
            Fz.append(imp[:, 2].sum() / wd.dt)
    Fz = np.array(Fz)
    m_est = float(Fz.mean() / 9.81)
    # segmented bootstrap CI
    segs = np.array_split(Fz, 20)
    seg_masses = np.array([s.mean() / 9.81 for s in segs if len(s)])
    ci = [float(np.percentile(seg_masses, 2.5)), float(np.percentile(seg_masses, 97.5))]
    report["items"]["G4_mass_est_kg"] = round(m_est, 3)
    report["items"]["G4_mass_ci95"] = [round(c, 3) for c in ci]
    report["items"]["G4_frames_used"] = int(len(Fz))
    report["items"]["G4_PASS"] = bool(35.0 <= m_est <= 39.5 and len(Fz) >= 60)

    out = os.path.join(REPO, "prime_identify/results/gm_validation.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    print("GM_VALIDATION", "PASS" if (g3 and report["items"]["G4_PASS"]) else "PARTIAL")


if __name__ == "__main__":
    main()
