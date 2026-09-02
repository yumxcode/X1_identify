#!/usr/bin/env python3
"""Parse real-robot logger CSVs into SPI clip datasets (.npz).

Supports role-split sources (x1_spi.yaml ``data.sources[].role``):
  * role=train (default) -> clips go to the main dataset (--out), which the
    identification + random holdout split operate on;
  * role=cross           -> clips go to the disjoint cross-policy dataset
    (--out-cross) used by completion criterion 5 (CROSS-DATASET): identified
    params must also beat nominal on data recorded under a *different policy
    checkpoint* (same robot, same controller gains).

The per-bucket cap uses *stratified* random subsampling (proportional to each
file's clip count) so every source file keeps representation even when the
cap bites (e.g. the small 7500model file inside a large 7500model_addDR mix).

Usage (remote / any machine with numpy+pyyaml):
  python spi_identify/scripts/prepare_dataset.py \
      --config spi_identify/configs/x1_spi.yaml \
      --out data/derived/x1_clips.npz \
      --out-cross data/derived/x1_cross_clips.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "spi_identify"))

from spi.dataset import parse_csv, save_clips, segment_clips  # noqa: E402


def stratified_cap(items, cap: int, seed: int):
    """Subsample <=cap items, proportional per source group.

    items: list of (group_key, clip). Guarantees >=1 clip per group while
    cap >= n_groups; deterministic under seed.
    """
    if cap <= 0 or len(items) <= cap:
        return [c for _, c in items]
    groups: dict = {}
    for g, c in items:
        groups.setdefault(g, []).append(c)
    quota = {g: max(1, int(round(cap * len(v) / len(items)))) for g, v in groups.items()}
    # trim overflow (rounding) from the largest groups
    overflow = sum(quota.values()) - cap
    for g in sorted(quota, key=lambda k: -len(groups[k])):
        while overflow > 0 and quota[g] > 1:
            quota[g] -= 1
            overflow -= 1
    rng = np.random.default_rng(seed)
    keep = []
    for g, clips in groups.items():
        idx = rng.permutation(len(clips))[: quota[g]]
        keep.extend((g, clips[i]) for i in sorted(idx))
    return [c for _, c in keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="train-role clips npz")
    ap.add_argument("--out-cross", default=None,
                    help="cross-role clips npz (omit to skip cross bucket)")
    ap.add_argument("--h-min", type=float, default=None)
    ap.add_argument("--h-max", type=float, default=None)
    ap.add_argument("--max-clips", type=int, default=None)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())

    train_items, cross_items = [], []
    group_stats = []
    for src in cfg["data"]["sources"]:
        role = src.get("role", "train")
        group = src.get("group", f"src{cfg['data']['sources'].index(src)}")
        kp = src.get("kp", 40.0)
        kd = src.get("kd", 3.0)
        kp = kp if isinstance(kp, dict) else {j: float(kp) for j in
                                              cfg.get("_leg_joints", LEG_JOINTS_FALLBACK)}
        kd = kd if isinstance(kd, dict) else {j: float(kd) for j in
                                              cfg.get("_leg_joints", LEG_JOINTS_FALLBACK)}
        for f in src["files"]:
            path = (ROOT / f).resolve()
            print(f"[prepare] parsing ({role}) {path}")
            log = parse_csv(path, kp=kp, kd=kd)
            clips = segment_clips(log,
                                  h_min_s=args.h_min or cfg["data"]["clip_h_min_s"],
                                  h_max_s=args.h_max or cfg["data"]["clip_h_max_s"],
                                  seed=cfg["data"].get("seed", 0))
            print(f"[prepare]   {len(clips)} clips (dt={log.dt*1e3:.2f} ms)")
            bucket = train_items if role == "train" else cross_items
            bucket.extend((group, c) for c in clips)
            group_stats.append({"role": role, "group": group, "file": f,
                                "n_clips": len(clips), "dur_s": round(len(log.t) * log.dt, 1)})

    seed = cfg["data"].get("seed", 0)
    train_cap = args.max_clips or cfg["data"].get("max_clips", 0)
    train_clips = stratified_cap(train_items, train_cap, seed)
    meta = {"n_clips": len(train_clips),
            "mean_len": float(np.mean([c["n"] for c in train_clips]))
            if train_clips else 0.0,
            "groups": group_stats}
    save_clips(train_clips, meta, args.out)
    print(f"[prepare] saved {len(train_clips)} train clips -> {args.out}")

    if args.out_cross and cross_items:
        cross_cap = int(cfg["data"].get("cross_max_clips", 0))
        cross_clips = stratified_cap(cross_items, cross_cap, seed)
        cmeta = {"n_clips": len(cross_clips),
                 "mean_len": float(np.mean([c["n"] for c in cross_clips])),
                 "groups": group_stats, "bucket": "cross"}
        save_clips(cross_clips, cmeta, args.out_cross)
        print(f"[prepare] saved {len(cross_clips)} cross clips -> {args.out_cross}")


LEG_JOINTS_FALLBACK = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]


if __name__ == "__main__":
    main()
