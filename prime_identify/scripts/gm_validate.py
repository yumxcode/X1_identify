"""Remote (gradmotion) AND local validation of the X1 identification results.

Runs identically in both environments (deps: pinocchio, numpy). Validates:
  [G3] physical consistency of the nominal and mass-anchored URDFs using the
       CORRECT origin-inertia convention (pc_min_eig_params; see
       prime/dynamics.py for the convention adjudication)
  [G4] GRF-balance total-mass estimate (same fixed seed/frames locally and
       remotely for an apples-to-apples comparison)

Writes a JSON summary to prime_identify/results/gm_validation.json.
"""
import json
import os
import subprocess
import sys

REPO = os.environ.get("X1_VALIDATION_ROOT") or (
    "/workspace/X1_identify" if os.path.isdir("/workspace/X1_identify") else os.getcwd()
)


def sh(cmd):
    print(f"+ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def ensure_deps():
    try:
        import pinocchio  # noqa: F401

        return
    except ImportError:
        pass
    # pip builds fail on this image; conda-forge ships prebuilt wheels
    sh("conda install -y -c conda-forge pinocchio")


def main():
    ensure_deps()
    import numpy as np  # noqa: F401
    import pinocchio as pin

    sys.path.insert(0, os.path.join(REPO, "prime_identify"))
    from prime.dynamics import X1Dynamics, pc_min_eig_origin
    from prime.data import load_walk_diag

    report = {"items": {}}

    # ---------- G3: physical consistency (origin convention) ----------
    urdf_nom = os.path.join(REPO, "X1_train/resources/robots/x1/urdf/x1.urdf")
    urdf_anc = os.path.join(REPO, "prime_identify/results/x1_gmass_anchored.urdf")
    if not os.path.exists(urdf_nom):
        urdf_nom = os.path.join(REPO, "urdf/x1.urdf")  # validate-lite layout

    model = pin.buildModelFromUrdf(urdf_nom, pin.JointModelFreeFlyer())
    n_bad = sum(
        1 for j in range(1, model.njoints) if pc_min_eig_origin(model, j) <= 1e-9
    )
    report["items"]["G3_nominal_pc_violations"] = n_bad  # adjudicated: 0

    g3 = False
    if os.path.exists(urdf_anc):
        dyn2 = X1Dynamics(urdf_anc)
        err = dyn2.selfcheck()
        worst = min(
            pc_min_eig_origin(dyn2.model, j) for j in range(1, dyn2.model.njoints)
        )
        report["items"]["G3_anchored_roundtrip_err"] = err
        report["items"]["G3_anchored_total_mass"] = round(dyn2.total_mass(), 3)
        report["items"]["G3_anchored_worst_min_eig"] = worst
        g3 = err < 1e-9 and worst > 0.0
    report["items"]["G3_PASS"] = bool(g3)

    # ---------- G4: GRF mass estimate (fixed seed => local==remote) ----------
    dyn = X1Dynamics(urdf_nom)
    csv_path = os.path.join(REPO, "data/raw/walk_diag_20260824_103222.csv")
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
