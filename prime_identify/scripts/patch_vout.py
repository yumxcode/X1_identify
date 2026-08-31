"""Patch: separate step-OUTPUT velocities (v_out) from measured state (v).

WalkData.v_out[k] is the velocity the estimator predicts from (q[k], v[k],
u[k]); by default it is the measured v[k+1] (finite difference of the
encoder stream). The synthetic self-test overwrites ONLY v_out with the
ground-truth rollout outputs, keeping the INPUT state v intact.
"""
import ast
import re

# ---- data.py: add v_out field ----
p1 = "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify/prime/data.py"
src = open(p1).read()
src = src.replace(
    """    foot_z: np.ndarray  # (T, 4) world-frame foot heights (before leveling)
    v_mask: np.ndarray  # (nv,) True where v is measured (base lin vel False)""",
    """    foot_z: np.ndarray  # (T, 4) world-frame foot heights (before leveling)
    v_mask: np.ndarray  # (nv,) True where v is measured (base lin vel False)
    v_out: np.ndarray = None  # (T, nv) step output velocity; default v[k+1]""",
)
src = src.replace(
    """    return WalkData(
        t=t,
        dt=dt,
        q=q_arr,
        v=v,
        u=eff,
        gyro=gyro,
        base_quat=quat,
        foot_z=foot_z_ref,
        v_mask=v_mask,
    )""",
    """    v_out = np.vstack([v[1:], v[-1:][None, :]])
    return WalkData(
        t=t,
        dt=dt,
        q=q_arr,
        v=v,
        u=eff,
        gyro=gyro,
        base_quat=quat,
        foot_z=foot_z_ref,
        v_mask=v_mask,
        v_out=v_out,
    )""",
)
open(p1, "w").write(src)
ast.parse(src)

# ---- pfie.py: use v_out everywhere ----
p2 = "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify/prime/pfie.py"
src = open(p2).read()
n = src.count("wd.v[k + 1]") + src.count("self.wd.v[k + 1]")
src = src.replace("self.wd.v[k + 1]", "self.wd.v_out[k]")
src = src.replace("wd.v[k + 1]", "wd.v_out[k]")
open(p2, "w").write(src)
ast.parse(src)
print(f"patched; v_out refs: {n}")

# ---- run_real_walk.py: same ----
p3 = "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify/scripts/run_real_walk.py"
src = open(p3).read()
src = src.replace("wd.v[k + 1]", "wd.v_out[k]")
open(p3, "w").write(src)
ast.parse(src)

# ---- selftest_sim.py: overwrite ONLY v_out ----
p4 = "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify/scripts/selftest_sim.py"
src = open(p4).read()
old = """    rng = np.random.default_rng(7)
    n = args.frames
    v_meas = wd.v[: n + 1].copy()
    n_bad = 0
    for k in range(n):
        v_plus, imp, conv = dyn.solve_contact_step(wd.q[k], wd.v[k], wd.u[k], wd.dt)
        if not conv:
            n_bad += 1
            if not np.all(np.isfinite(v_plus)):
                v_plus = wd.v[k]
        v_meas[k + 1] = v_plus
    print(f"synthetic rollout: {n_bad}/{n} frames flagged (accepted)")
    v_meas[1:, 6:] += rng.normal(0, args.noise, (n, 12))
    v_meas[1:, 3:6] += rng.normal(0, 0.005, (n, 3))
    wd.v = v_meas  # replace measurements with synthetic ones"""
new = """    rng = np.random.default_rng(7)
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
    wd.v_out = v_out_syn  # overwrite step OUTPUTS only; inputs stay measured"""
assert old in src, "selftest synth block not found"
src = src.replace(old, new)
open(p4, "w").write(src)
ast.parse(src)
print("all patched OK")
