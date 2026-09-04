#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-1 divergence diagnosis: clip-length scan of open-loop replay errors.

Answers the review question (2026-09-04_sysid_review.md §2/§10): within what
time window is the no-mocap open-loop replay still a *prediction* rather than
a *falling simulation*? For each clip length L in {0.1, 0.2, 0.5, 1.0, 2.0} s
the same data is re-segmented and rolled out under {nominal, R9} params x
{zero, stance} initial base linear velocity (P0-2 side-by-side), reporting:

  per length:  mean base attitude error [deg], joint-angle error RMS [rad],
               accel specific-force error RMS [m/s^2]  (same box filter as cost)
  per 0.1 s time bin since clip start: attitude error + accel error growth
               curves (the "valid window" readout)

Diagnostic only — exit code 0 regardless of numbers (NOT a gate).

Usage (remote / local with mujoco):
  python spi_identify/scripts/clip_scan_diag.py \
      --config spi_identify/configs/x1_spi.yaml \
      --params spi_identify/results/r9_indomain_params.json \
      --out-dir logs/clip_scan
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "spi_identify"))

from spi.cost import quat_err, box_filter  # noqa: E402
from spi.dataset import LEG_JOINTS, JIDX, parse_csv, segment_clips  # noqa: E402
from spi.optimizer import build_space  # noqa: E402
from spi.rollout import MuJoCoRollouter  # noqa: E402

LENGTHS = [0.1, 0.2, 0.5, 1.0, 2.0]
TIME_BIN = 0.1          # s
MAX_TRAIN_CLIPS = 24    # per length
MAX_CROSS_CLIPS = 12
ACCEL_WIN = 20          # box filter window, matches cost.accel_filter_win


def quat_angle_deg(sim_q, ref_q):
    """Per-step attitude angle error in degrees (geodesic, sign-invariant)."""
    q = sim_q / np.maximum(np.linalg.norm(sim_q, axis=-1, keepdims=True), 1e-12)
    r = ref_q / np.maximum(np.linalg.norm(ref_q, axis=-1, keepdims=True), 1e-12)
    dot = np.clip(np.abs(np.sum(q * r, axis=-1)), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def scan_length(rollouter, clips, params, kappa_map, cost_mask):
    """Roll out clips under one (params, v0-mode) setting; return per-clip
    metrics + per-time-bin accumulators."""
    acc_q = []
    acc_qd = []          # joint-angle error RMS per clip
    acc_accel = []
    bin_angle = {}       # bin_idx -> [sum, count]
    bin_accel = {}
    for c in clips:
        sim = rollouter.rollout_clip(c, params, kappa_map)
        ang = quat_angle_deg(sim["quat"], c["ref_quat"])
        acc_q.append(float(np.mean(ang)))
        d2 = np.sum((sim["q"][:, cost_mask] - c["ref_q"][:, cost_mask]) ** 2, axis=1)
        acc_qd.append(float(np.sqrt(np.mean(d2))))
        sa = box_filter(sim["accel"], ACCEL_WIN)
        ra = box_filter(c["ref_accel"], ACCEL_WIN)
        a2 = np.sum((sa - ra) ** 2, axis=1)
        acc_accel.append(float(np.sqrt(np.mean(a2))))
        t = np.arange(c["n"]) * c["dt"]
        bins = (t // TIME_BIN).astype(int)
        for b, av, rv in zip(bins, ang, np.sqrt(np.sum((sa - ra) ** 2, axis=1))):
            for store, val in ((bin_angle, av), (bin_accel, rv)):
                s = store.setdefault(int(b), [0.0, 0])
                s[0] += val
                s[1] += 1
    out = {
        "n_clips": len(clips),
        "attitude_err_deg_mean": float(np.mean(acc_q)) if acc_q else None,
        "joint_err_rms_mean": float(np.mean(acc_qd)) if acc_qd else None,
        "accel_err_rms_mean": float(np.mean(acc_accel)) if acc_accel else None,
        "time_bins": {
            str(b): {"t_s": round((b + 0.5) * TIME_BIN, 2),
                     "angle_deg": round(v[0] / v[1], 2),
                     "accel_rms": round(bin_accel.get(b, [0, 1])[0] /
                                        max(bin_accel.get(b, [0, 1])[1], 1), 2),
                     "n": v[1]}
            for b, v in sorted(bin_angle.items())
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="spi_identify/configs/x1_spi.yaml")
    ap.add_argument("--params", default="spi_identify/results/r9_indomain_params.json")
    ap.add_argument("--out-dir", default="logs/clip_scan")
    ap.add_argument("--lengths", default=",".join(str(x) for x in LENGTHS))
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    lengths = [float(x) for x in args.lengths.split(",")]

    # parse each source file once, re-segment per length
    logs = {}
    for src in cfg["data"]["sources"]:
        role = src.get("role", "train")
        group = src.get("group", "src")
        for f in src["files"]:
            path = (ROOT / f).resolve()
            log = parse_csv(path, kp=src["kp"], kd=src["kd"])
            logs[(role, group, f)] = log

    mjcf = (ROOT / cfg["model"]["mjcf"]).resolve()
    kmap = {g["name"]: [JIDX[j] for j in g["joints"]] for g in cfg["motor_groups"]}
    space = build_space(cfg)
    nominal = space.nominal_params()

    sys.path.insert(0, str(ROOT / "spi_identify" / "scripts"))
    from validate_spi import coerce_params
    payload = json.loads((ROOT / args.params).read_text())
    r9 = coerce_params(payload["best_params"])

    cost_mask = np.zeros(29, dtype=bool)
    for j in LEG_JOINTS:
        cost_mask[JIDX[j]] = True

    results = {}
    for mode in ("zero", "stance"):
        rollouter = MuJoCoRollouter(mjcf, base_body=cfg["model"]["base_body"],
                                    foot_bodies=tuple(cfg["model"]["foot_bodies"]),
                                    base_linvel_mode=mode)
        for L in lengths:
            train_clips, cross_clips = [], []
            for (role, group, f), log in logs.items():
                clips = segment_clips(log, h_min_s=L, h_max_s=L,
                                      seed=cfg["data"].get("seed", 0))
                cap = MAX_TRAIN_CLIPS if role == "train" else MAX_CROSS_CLIPS
                clips = clips[:cap]
                (train_clips if role == "train" else cross_clips).extend(clips)
            for bucket, clips in (("train", train_clips), ("cross", cross_clips)):
                if not clips:
                    continue
                for pname, p in (("nominal", nominal), ("r9", r9)):
                    key = f"{mode}/L={L:.1f}s/{bucket}/{pname}"
                    print(f"[scan] {key} ({len(clips)} clips)", flush=True)
                    results[key] = scan_length(rollouter, clips, p, kmap, cost_mask)

    out_dir = Path(args.out_dir)
    gm_dir = out_dir / "gm_play"
    gm_dir.mkdir(parents=True, exist_ok=True)
    (gm_dir / "clip_scan.json").write_text(json.dumps(results, indent=1))
    try:
        import torch
        torch.save(results, gm_dir / "clip_scan.pt")
    except ImportError:
        pass

    # markdown summary: one row per (mode, length) averaging buckets x params
    print("\n| v0-mode | L [s] | n | attitude err [deg] | joint err [rad] | accel err [m/s^2] |")
    print("|---|---|---|---|---|---|")
    for mode in ("zero", "stance"):
        for L in lengths:
            rows = [v for k, v in results.items()
                    if k.startswith(f"{mode}/L={L:.1f}s/")]
            if not rows:
                continue
            n = sum(r["n_clips"] for r in rows)
            ang = np.mean([r["attitude_err_deg_mean"] for r in rows])
            jnt = np.mean([r["joint_err_rms_mean"] for r in rows])
            acc = np.mean([r["accel_err_rms_mean"] for r in rows])
            print(f"| {mode} | {L:.1f} | {n} | {ang:.1f} | {jnt:.3f} | {acc:.2f} |")

    # time-bin table for the 2 s runs (valid-window readout)
    print("\n## time-bin growth (2 s clips, r9 params)")
    for mode in ("zero", "stance"):
        for bucket in ("train", "cross"):
            k = f"{mode}/L=2.0s/{bucket}/r9"
            if k not in results:
                continue
            tb = results[k]["time_bins"]
            curve = " ".join(f"{b['t_s']:.1f}s:{b['angle_deg']:.0f}deg/{b['accel_rms']:.0f}"
                             for b in tb.values())
            print(f"- {mode}/{bucket}: {curve}")
    print(f"\n[scan] results -> {gm_dir / 'clip_scan.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
