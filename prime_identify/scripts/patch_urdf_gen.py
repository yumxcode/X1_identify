"""Patch make_identified_urdf.py: base link delta-only update."""
import ast

path = "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify/scripts/make_identified_urdf.py"
src = open(path).read()
start = src.index("    updated = 0")
end = src.index("    tree.write(URDF_OUT")
new_block = '''    updated = 0
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
'''
src = src[:start] + new_block + src[end:]
open(path, "w").write(src)
ast.parse(src)
print("patched OK")
