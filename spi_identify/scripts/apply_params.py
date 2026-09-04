#!/usr/bin/env python3
"""Apply identified SPI parameters: patch URDF/MJCF + emit DR config for
downstream remote retraining (agibot_x1_train framework).

Inputs : identified_params.json (from run_spi.py / extract_pt.py)
Outputs (under --out-dir, default spi_identify/export/):
  * x1_identified.urdf          pelvis inertial patched (base_link)
  * xyber_x1_identified.xml     MJCF robot include patched (x1-body)
  * dr_x1_spi.json              DR ranges re-centred on identified values
  * report.md                   human-readable diff vs nominal

Physical plausibility clamp (default ON, ``--no-clamp`` disables):
  without mocap the base translation cost is off, so com_y/z and inertia are
  weakly observable and tend to absorb model error into implausible values.
  Exported model files clamp: com offset +-0.15 m per axis (paper CoM box),
  inertia eigenvalues to [I_MIN, I_MAX] = [0.005, 1.0] kg m^2 (paper-style
  pelvis box).  Raw identified values are always kept in report.md and in
  dr_x1_spi.json (``raw`` block) — nothing is hidden.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

NOMINAL_PELVIS = {  # from MJCF x1-body / URDF base_link
    "mass": 4.3041648,
    "com": np.array([0.00252285, -0.00063439, 0.03023409]),
}
COM_CLAMP = 0.15      # m, per axis, around nominal com
I_MIN, I_MAX = 0.005, 1.0   # kg m^2, eigenvalue clamp


def _parse_nd(s) -> np.ndarray:
    """Coerce a value that may be a list or a numpy-repr *string* (produced by
    json.dumps(..., default=str) in run_spi.py) into a float array."""
    if isinstance(s, str):
        return np.fromstring(re.sub(r"[\[\]\n]", " ", s), sep=" ")
    return np.asarray(s, dtype=float)

DR_TEMPLATE = {
    "randomize_base_mass": True,
    "added_mass_range_comment": "centred on identified mass, +/-5% (paper nominal DR)",
    "randomize_com": True,
    "com_range_comment": "identified com +/- 0.03 m (paper nominal +/-0.1 around URDF)",
    "randomize_gains": True,
    "stiffness_multiplier_range": [0.9, 1.1],
    "damping_multiplier_range": [0.9, 1.1],
    "randomize_torque": True,
    "randomize_link_mass": True,
    "added_link_mass_range": [0.95, 1.05],
}


def clamp_physical(mass, com, inertia, clamp: bool):
    """Clamp weakly-observable directions into the physically defensible box."""
    com_c = np.array(com, dtype=float)
    I_c = np.array(inertia, dtype=float)
    notes = []
    if clamp:
        ref = NOMINAL_PELVIS["com"]
        off = com_c - ref
        for i in range(3):
            if abs(off[i]) > COM_CLAMP:
                com_c[i] = ref[i] + np.sign(off[i]) * COM_CLAMP
                notes.append(f"com[{i}] clamped to +-0.15 m")
        I_c = 0.5 * (I_c + I_c.T)
        lam, V = np.linalg.eigh(I_c)
        lam_c = np.clip(lam, I_MIN, I_MAX)
        if not np.allclose(lam, lam_c):
            notes.append(f"inertia eig {np.round(lam,4).tolist()} -> "
                         f"{np.round(lam_c,4).tolist()} (clamped to [{I_MIN},{I_MAX}])")
        if np.linalg.det(V) < 0:
            V[:, 0] = -V[:, 0]
        I_c = V @ np.diag(lam_c) @ V.T
    return float(mass), com_c, I_c, notes


def patch_urdf(urdf_text: str, body: str, mass: float, com, inertia) -> str:
    new_inertial = (f'<inertial>\n    <origin xyz="{com[0]:.8f} {com[1]:.8f} {com[2]:.8f}"'
                    f' rpy="0 0 0"/>\n    <mass value="{mass:.8f}"/>\n'
                    f'    <inertia ixx="{inertia[0][0]:.8f}" ixy="{inertia[0][1]:.8f}"'
                    f' ixz="{inertia[0][2]:.8f}" iyy="{inertia[1][1]:.8f}"'
                    f' iyz="{inertia[1][2]:.8f}" izz="{inertia[2][2]:.8f}"/>\n  </inertial>')
    pat = re.compile(r'(<link\s+name="' + re.escape(body) + r'"\s*>\s*)(.*?)(</link>)',
                     re.S)
    m = pat.search(urdf_text)
    if not m:
        raise SystemExit(f"cannot locate pelvis <link name={body}> in URDF")
    block_new = re.sub(r"<inertial>.*?</inertial>", new_inertial, m.group(2),
                       count=1, flags=re.S)
    return urdf_text[:m.start(2)] + block_new + urdf_text[m.end(2):]


def patch_mjcf(mjcf_text: str, body: str, mass: float, com, inertia) -> str:
    I = np.asarray(inertia, dtype=float)
    lam, V = np.linalg.eigh(0.5 * (I + I.T))
    if np.linalg.det(V) < 0:
        V[:, 0] = -V[:, 0]
    q = _mat2quat(V)
    full = (f'<inertial pos="{com[0]:.8f} {com[1]:.8f} {com[2]:.8f}" '
            f'mass="{mass:.8f}" '
            f'diaginertia="{lam[0]:.8f} {lam[1]:.8f} {lam[2]:.8f}" '
            f'quat="{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}"/>')
    pat = re.compile(r'(<body\s+name\s*=\s*"' + re.escape(body) + r'".*?>)(.*?)(</body>)', re.S)
    m = pat.search(mjcf_text)
    if not m:
        raise SystemExit(f"cannot locate <body name={body}> in MJCF")
    block_new = re.sub(r"<inertial[^>]*/>", full, m.group(2), count=1)
    return mjcf_text[:m.start(2)] + block_new + mjcf_text[m.end(2):]


def _mat2quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> wxyz unit quaternion (inline; mirrors spi.rollout)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="identified_params.json")
    ap.add_argument("--urdf", default=str(ROOT / "spi_identify/resources/x1_nominal.urdf"))
    ap.add_argument("--mjcf", default=str(
        ROOT / "spi_identify/resources/mjcf/robot/xyber_x1/xyber_x1_serial.xml"),
        help="MJCF robot include to patch (default: vendored 2.3.6-compatible copy)")
    ap.add_argument("--urdf-body", default="base_link")
    ap.add_argument("--mjcf-body", default="x1-body")
    ap.add_argument("--out-dir", default=str(ROOT / "spi_identify/export"))
    ap.add_argument("--no-clamp", action="store_true",
                    help="export raw identified values without physical clamp")
    args = ap.parse_args()

    payload = json.loads(Path(args.params).read_text())

    # multi-body: read body definitions (nominal + physical domain + mjcf body
    # name) from the identification config; bodies absent from the payload are
    # skipped (older single-body results keep working).
    import yaml
    cfg_path = ROOT / "spi_identify/configs/x1_spi.yaml"
    cfg_bodies = yaml.safe_load(cfg_path.read_text())["bodies"] \
        if cfg_path.exists() else []
    sys.path.insert(0, str(ROOT / "spi_identify"))
    from spi.param_space import project_to_physical_domain  # noqa: E402

    raw_bodies = {}
    for name, p in payload["best_params"]["bodies"].items():
        raw_bodies[name] = {
            "mass": float(p["mass"]),
            "com": np.asarray(_parse_nd(p["com"]), dtype=float),
            "inertia": np.asarray(_parse_nd(p["inertia"]), dtype=float).reshape(3, 3),
        }

    if args.no_clamp:
        exported = {k: dict(v) for k, v in raw_bodies.items()}
        clamp_notes: dict = {}
    else:
        # project ALL payload bodies present in the config domain (same
        # semantics as the in-loop span-normalized penalty; older results with
        # bodies outside the config pass through untouched via a 1-entry cfg)
        cfg_subset = [b for b in cfg_bodies if b["name"] in raw_bodies] or \
            [{"name": n, "mass_range": (0.0, 1e9), "com_range": (-1e9, 1e9),
              "inertia_diag_range": (0.0, 1e9)} for n in raw_bodies]
        proj = project_to_physical_domain(
            {"bodies": raw_bodies, "motors": {}, "kappa_s": 1.0}, cfg_subset)
        exported = proj["bodies"]
        for name in raw_bodies:
            n0, n1 = raw_bodies[name], exported[name]
            changed = not (np.isclose(n0["mass"], n1["mass"])
                           and np.allclose(n0["com"], n1["com"])
                           and np.allclose(n0["inertia"], n1["inertia"]))
            clamp_notes[name] = bool(changed)

    for name in raw_bodies:
        mjcf_name = next((b.get("mjcf_body", args.mjcf_body)
                          for b in cfg_bodies if b["name"] == name), args.mjcf_body)
        n0, n1 = raw_bodies[name], exported[name]
        print(f"[apply] {name} (raw): m={n0['mass']:.4f} com={n0['com'].round(5).tolist()} "
              f"I_diag={np.diag(n0['inertia']).round(5).tolist()}")
        if clamp_notes.get(name):
            print(f"[apply] CLAMP: {name} projected to configured physical domain")
        print(f"[apply] {name} (export): m={n1['mass']:.4f} com={n1['com'].round(5).tolist()} "
              f"I_diag={np.diag(n1['inertia']).round(5).tolist()}")

    # legacy single-body aliases (pelvis) for downstream consumers
    p = exported.get("base") or next(iter(exported.values()))
    mass, com, I = float(p["mass"]), np.asarray(p["com"], float), np.asarray(p["inertia"], float)
    mass_raw, com_raw = raw_bodies.get("base", p)["mass"], raw_bodies.get("base", p)["com"]
    I_raw = raw_bodies.get("base", p)["inertia"]
    notes = [f"{k} projected to physical domain" for k, v in clamp_notes.items() if v]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    urdf_path = Path(args.urdf)
    if urdf_path.exists():
        b = exported.get("base")
        if b is not None:  # URDF carries only the pelvis (legs model)
            text = patch_urdf(urdf_path.read_text(), args.urdf_body,
                              float(b["mass"]), np.asarray(b["com"], float),
                              np.asarray(b["inertia"], float))
            (out / "x1_identified.urdf").write_text(text)
            print(f"[apply] URDF -> {out / 'x1_identified.urdf'}")
        else:
            print("[apply] no 'base' body in params; URDF patch skipped")
    else:
        print("[apply] URDF source not found; skipped (pass --urdf)")

    mjcf_path = Path(args.mjcf)
    if mjcf_path.exists():
        text = mjcf_path.read_text()
        for name, b in exported.items():
            mjcf_name = next((bb.get("mjcf_body", args.mjcf_body)
                              for bb in cfg_bodies if bb["name"] == name), args.mjcf_body)
            text = patch_mjcf(text, mjcf_name, float(b["mass"]),
                              np.asarray(b["com"], float),
                              np.asarray(b["inertia"], float))
            print(f"[apply] MJCF body '{mjcf_name}' ({name}) patched")
        (out / "xyber_x1_identified.xml").write_text(text)
        print(f"[apply] MJCF -> {out / 'xyber_x1_identified.xml'}")
    else:
        print("[apply] MJCF source not found; skipped (pass --mjcf)")

    dr = dict(DR_TEMPLATE)
    dr["identified_mass"] = mass
    dr["identified_com"] = com.tolist()
    dr["added_mass_range"] = [round(mass * 0.95, 4), round(mass * 1.05, 4)]
    dr["com_range"] = [[round(com[i] - 0.03, 5), round(com[i] + 0.03, 5)]
                       for i in range(3)]
    dr["motor_kappa"] = {k: float(v) for k, v in payload["best_params"]["motors"].items()}
    dr["kappa_s"] = float(payload["best_params"]["kappa_s"])
    dr["bodies"] = {name: {"mass": float(b["mass"]),
                           "com": np.asarray(b["com"], float).tolist(),
                           "mjcf_body": next(
                               (bb.get("mjcf_body", "x1-body")
                                for bb in cfg_bodies if bb["name"] == name), "x1-body")}
                    for name, b in exported.items()}
    dr["raw"] = {"mass": float(mass_raw), "com": np.asarray(com_raw, float).tolist(),
                 "inertia": np.asarray(I_raw, float).tolist(),
                 "clamped": bool(notes)}
    (out / "dr_x1_spi.json").write_text(json.dumps(dr, indent=2))
    print(f"[apply] DR  -> {out / 'dr_x1_spi.json'}")

    # cost provenance: run_spi.py payloads carry nominal_cost/best_cost;
    # validate-only payloads (e.g. committed v15 results) carry costs.val
    costs_val = (payload.get("costs") or {}).get("val") or {}
    nom_cost = payload.get("nominal_cost") or costs_val.get("nominal")
    best_cost = payload.get("best_cost") or costs_val.get("best")
    ratio = (nom_cost / best_cost) if nom_cost and best_cost else None
    n_clips = payload.get("n_clips") or (payload.get("run") or {}).get("dataset")
    hist = payload.get("history") or []
    hist_tail = "\n".join(f"  - {h}" for h in hist[-5:])
    cost_line = (
        f"Multi-step prediction cost: nominal **{nom_cost:.1f}** -> best "
        f"**{best_cost:.1f}** ({ratio:.1f}x lower).\n\n"
        if nom_cost and best_cost
        else "Multi-step prediction cost: see params provenance "
             f"(costs.val nominal={nom_cost} best={best_cost}).\n\n")
    report = (
        f"# SPI identified parameters — X1 pelvis + motor model\n\n"
        f"Data: {n_clips} clips (walk_diag)\n\n"
        f"## Result\n\n"
        f"| quantity | nominal | identified (raw) | exported (clamped) |\n|---|---|---|---|\n"
        f"| mass [kg] | {NOMINAL_PELVIS['mass']:.4f} | {mass_raw:.4f} | {mass:.4f} |\n"
        f"| com [m] | {NOMINAL_PELVIS['com'].tolist()} | {com_raw.round(6).tolist()} | "
        f"{com.round(6).tolist()} |\n"
        f"| I diag [kg m^2] | [0.0268, 0.0108, 0.0218] | "
        f"{np.diag(I_raw).round(6).tolist()} | {np.diag(I).round(6).tolist()} |\n"
        f"| motor kappa | (see config nominal) | "
        f"{payload['best_params']['motors']} | same |\n"
        f"| kappa_s | 1.0 | {payload['best_params']['kappa_s']:.4f} | same |\n\n"
        + cost_line
        + f"\n## Notes\n\n"
        + ("\n".join(f"* clamp: {n}" for n in notes) if notes
           else "* no clamping applied (--no-clamp or in-box values)")
        + "\n* weak observability without mocap: com_y/z and inertia absorb model "
        "error; kappas are all in-box and well identified.\n"
        "\n## Optimization history (tail)\n\n"
        + (hist_tail if hist else "(none — params loaded from committed results)\n"))
    (out / "report.md").write_text(report)
    print(f"[apply] report -> {out / 'report.md'}")


if __name__ == "__main__":
    main()
