"""Generate the GRF-mass-anchored URDF (canonical validation artifact).

NOT an identification output (per PASS_CRITERIA.md gates): the ONLY change
vs the raw URDF is the base_link mass, anchored so the total equals the
GRF-balance estimate from walk_diag contact solving (mean sum(Fz)/g).

All other links keep their raw URDF values verbatim. The raw URDF has been
adjudicated physically consistent: 0/13 pseudo-inertia violations under the
correct origin-inertia convention (prime.dynamics.pc_min_eig_params, with
bidirectional unit test). An earlier "13/13 violations" diagnosis was a
convention bug in the checker (treated origin inertia as COM inertia).

Output: results/x1_gmass_anchored.urdf
"""
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from prime.dynamics import X1Dynamics, pc_min_eig_origin  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
URDF_IN = os.path.join(REPO, "X1_train/resources/robots/x1/urdf/x1.urdf")
if not os.path.exists(URDF_IN):
    URDF_IN = os.path.join(REPO, "urdf/x1.urdf")  # validate-lite layout
URDF_OUT = os.path.join(
    REPO, "prime_identify/results/x1_gmass_anchored.urdf"
)
GRF_TOTAL_MASS = 37.10  # kg, run_real_walk.py GRF-balance estimate


def main():
    dyn = X1Dynamics(URDF_IN)
    nominal_total = dyn.total_mass()
    delta = GRF_TOTAL_MASS - nominal_total
    assert delta > 0, delta

    tree = ET.parse(URDF_IN)
    root = tree.getroot()
    link = next(l for l in root.findall("link") if l.get("name") == "base_link")
    inert = link.find("inertial")
    mass_el = inert.find("mass")
    m0 = float(mass_el.get("value"))
    mass_el.set("value", f"{m0 + delta:.9g}")
    # uniform scaling of a rigid body's pseudo-inertia preserves PD
    scale = (m0 + delta) / m0
    ine = inert.find("inertia")
    for attr in ("ixx", "iyy", "izz", "ixy", "iyz", "ixz"):
        ine.set(attr, f"{float(ine.get(attr)) * scale:.9g}")
    tree.write(URDF_OUT, encoding="utf-8", xml_declaration=True)

    dyn2 = X1Dynamics(URDF_OUT)
    dyn2.selfcheck()
    worst = min(pc_min_eig_origin(dyn2.model, j) for j in range(1, dyn2.model.njoints))
    print(f"wrote {URDF_OUT}")
    print(f"  total mass: {dyn2.total_mass():.3f} kg (target {GRF_TOTAL_MASS})")
    print(f"  worst pseudo-inertia min eig: {worst:+.3e} "
          f"({'PD' if worst > 0 else 'VIOLATION'})")
    assert abs(dyn2.total_mass() - GRF_TOTAL_MASS) < 1e-3
    assert worst > 0


if __name__ == "__main__":
    main()
