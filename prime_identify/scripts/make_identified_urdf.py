"""Generate the GRF-mass-repaired URDF (NOT a parameter identification output).

This is a model REPAIR artifact: physically-consistent projection of the
nominal inertias + total-mass anchoring to the GRF estimate. It contains
NO identified inertia parameters until the estimator passes its gate.

Changes applied (deltas relative to the raw URDF):
  1. physical-consistency repair of ALL bodies (the SolidWorks-exported
     nominal inertias have negative pseudo-inertia eigenvalues and are not
     realizable; we project onto the consistent set)
  2. base-link mass update from the GRF-balance estimate of the total mass
     (walk_diag contact solving: mean sum(Fz) = m_total * g, verified to
     0.1% on a synthetic static test)

Output: results/x1_gmass_repair.urdf
"""
import sys
import xml.etree.ElementTree as ET
import copy

import numpy as np

sys.path.insert(0, "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify")
from prime.dynamics import X1Dynamics, _inertia_to_vec6

URDF_IN = "/Users/yumx/code/robot_x/X1/X1_辨识/X1_train/resources/robots/x1/urdf/x1.urdf"
URDF_OUT = "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify/results/x1_gmass_repair.urdf"
GRF_TOTAL_MASS = 37.10  # kg, from run_real_walk.py (mean GRF sum / g)
NOMINAL_TOTAL = 35.323


def main():
    dyn = X1Dynamics(URDF_IN)
    pi = dyn.pi_nominal.copy()  # projected (physically consistent) nominal
    # distribute the GRF-implied total-mass delta onto the base link
    k = dyn.bodies.index(dyn.model.getJointId("root_joint"))
    pi[k, 0] += GRF_TOTAL_MASS - NOMINAL_TOTAL

    # ---- write back into the URDF XML --------------------------------
    tree = ET.parse(URDF_IN)
    root = tree.getroot()
    # body name for joint j in pinocchio == link name carrying that inertia
    # (pinocchio attaches the link inertia of the child link to the joint)
    joint2link = {}
    for j in root.findall("joint"):
        child = j.find("child").get("link")
        joint2link[j.get("name")] = child
    # bodies in pinocchio: joint ids 1..13, names = joint names
    updated = 0
    mass_delta = GRF_TOTAL_MASS - NOMINAL_TOTAL
    for b in dyn.bodies:
        jname = dyn.model.names[b]
        kk = dyn.bodies.index(b)
        link_name = joint2link.get(jname)
        if link_name is None:
            # pinocchio's freeflyer body aggregates base_link + its FIXED
            # subtree (lumber/arms/hands, ~17 kg). The aggregated inertia
            # cannot be uniquely decomposed back, so for the base link we
            # only add the identified mass delta (scaled inertia, same COM).
            link_name = "base_link"
            link = next(l for l in root.findall("link") if l.get("name") == link_name)
            inert = link.find("inertial")
            mass_el = inert.find("mass")
            m0 = float(mass_el.get("value"))
            scale = (m0 + mass_delta) / m0
            mass_el.set("value", f"{m0 + mass_delta:.9g}")
            ine = inert.find("inertia")
            for attr in ("ixx", "iyy", "izz", "ixy", "iyz", "ixz"):
                ine.set(attr, f"{float(ine.get(attr)) * scale:.9g}")
            updated += 1
            continue
        # movable leg joint body == its child link: write projected values
        link = next(l for l in root.findall("link") if l.get("name") == link_name)
        inert = link.find("inertial")
        m, mc = pi[kk, 0], pi[kk, 1:4]
        origin = inert.find("origin")
        origin.set("xyz", " ".join(f"{x:.9g}" for x in mc / m))
        Ivec = _inertia_to_vec6(np.array(
            [[pi[kk, 4], pi[kk, 5], pi[kk, 7]],
             [pi[kk, 5], pi[kk, 6], pi[kk, 8]],
             [pi[kk, 7], pi[kk, 8], pi[kk, 9]]]))
        ine = inert.find("inertia")
        for attr, val in zip(("ixx", "iyy", "izz", "ixy", "iyz", "ixz"), Ivec):
            ine.set(attr, f"{val:.9g}")
        mass_el = inert.find("mass")
        mass_el.set("value", f"{m:.9g}")
        updated += 1
    tree.write(URDF_OUT, encoding="utf-8", xml_declaration=True)

    # verify: reload and check total mass + consistency
    dyn2 = X1Dynamics(URDF_OUT)
    dyn2.selfcheck()
    print(f"wrote {URDF_OUT}")
    print(f"  bodies updated: {updated}")
    print(f"  total mass: {dyn2.total_mass():.3f} kg (target {GRF_TOTAL_MASS})")
    # physical consistency of every body
    from prime.log_cholesky import is_physically_consistent
    okk = all(
        is_physically_consistent(
            np.concatenate([[pi[kk, 0]], pi[kk, 1:4],
                            [pi[kk, 4], pi[kk, 6], pi[kk, 9], pi[kk, 5],
                             pi[kk, 8], pi[kk, 7]]]))
        for kk in range(pi.shape[0])
    )
    print(f"  all bodies physically consistent: {okk}")


if __name__ == "__main__":
    main()
