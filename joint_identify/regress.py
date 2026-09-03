#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure-numpy regression primitives for X1 joint-level system identification.

Design contract
---------------
* numpy-only, no IO, no mujoco, no scipy -> unit-testable locally
  (`.venv/bin/python -m unittest discover -s joint_identify/tests`).
* Units (as logged): tau [Nm], q [rad], qd [rad/s], qdd [rad/s^2], g [Nm],
  J_eff [kg*m^2], tau_c [Nm], tau_v [Nm*s/rad], c0 [Nm], k_t [Nm/A-unit].

Serial-joint suspended dynamics model (hip*/knee; ankles are parallel drive):
    tau_meas - g(q) = J_eff*qdd + tau_c*tanh(qd/eps) + tau_v*qd + c0 [+ G*gyro]
The optional imu_gyro regressors absorb sling/base-motion coupling and apply
to ALL serial joints (knee gyro_rms ~0.29 rad/s is the LARGEST serial value —
hip ~0.19, ankle_roll ~0.009; evidence: data/derived/step_m1_regression_all.json
gyro_rms fields; T10 extended coupling from hip-only to every serial joint).

Actuation-gain (M1) model, must stay compatible with
data/derived/step_m1_regression_all.json (alpha evidence band 0.34..0.71 on
SERIAL joints only; parallel ankles legitimately exceed 1):
    tau_meas ~= alpha*kp*(q_des - q) + beta*kd*qd + c
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "savgol_coeffs", "savgol_deriv", "savgol_ddq",
    "irls_huber_fit", "r_squared",
    "fit_dynamics", "xcorr_delay", "fit_m1", "fit_kt",
    "detect_feedback_semantics", "is_serial",
]

SERIAL_JOINT_HINTS = ("hip_pitch", "hip_roll", "hip_yaw", "knee_pitch")


def is_serial(joint):
    """True for serial-drive joints (hip/knee); False for parallel ankles."""
    return any(h in joint for h in SERIAL_JOINT_HINTS)


def _as_1d_float(x):
    return np.asarray(x, dtype=float).ravel()


# ---------------------------------------------------------------------------
# Savitzky-Golay differentiation (numpy-only)
# ---------------------------------------------------------------------------
def savgol_coeffs(window, poly, deriv, dt):
    """Centered SG differentiation coefficients; w has units 1/s^deriv."""
    window = int(window)
    poly = int(poly)
    if window % 2 == 0:
        window += 1
    if window < poly + 2:
        window = poly + 2 + (1 if (poly + 2) % 2 == 0 else 0)
        if window % 2 == 0:
            window += 1
    half = window // 2
    t = (np.arange(window) - half) * float(dt)
    V = np.vander(t, poly + 1, increasing=True)
    Vinv = np.linalg.pinv(V)
    import math
    return float(math.factorial(int(deriv))) * Vinv[int(deriv), :]


def savgol_deriv(x, dt, window=21, poly=3, deriv=1):
    """SG derivative with edge-replication padding; same length as x."""
    x = _as_1d_float(x)
    n = x.size
    window = int(window)
    if window % 2 == 0:
        window += 1
    if window < poly + 2:
        window = poly + 2
        if window % 2 == 0:
            window += 1
    half = window // 2
    pad = np.concatenate([np.full(half, x[0] if n else 0.0), x,
                          np.full(half, x[-1] if n else 0.0)])
    w = savgol_coeffs(window, poly, deriv, dt)
    return np.convolve(pad, w[::-1], mode="valid")


def savgol_ddq(q, dt, window=21, poly=3):
    """Joint acceleration qdd [rad/s^2] from angle q [rad]."""
    return savgol_deriv(q, dt, window, poly, deriv=2)


# ---------------------------------------------------------------------------
# Robust linear regression
# ---------------------------------------------------------------------------
def irls_huber_fit(X, y, n_iter=3, delta=1.345):
    """IRLS with Huber weights (scale = 1.345*MAD per round).

    Returns (beta (p,), weights (n,)).
    """
    X = np.asarray(X, dtype=float)
    y = _as_1d_float(y)
    if X.ndim == 1:
        X = X[:, None]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    w = np.ones_like(y)
    for _ in range(int(n_iter)):
        r = y - X @ beta
        med = np.median(np.abs(r - np.median(r)))
        s = max(1.4826 * med, 1e-9)
        z = np.abs(r) / s
        w = np.where(z <= delta, 1.0, delta / np.maximum(z, 1e-12))
        Xw = X * w[:, None]
        yw = y * w
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return beta, w


def r_squared(y, yhat):
    y = _as_1d_float(y)
    yhat = _as_1d_float(yhat)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ---------------------------------------------------------------------------
# Joint dynamics / delay / gains
# ---------------------------------------------------------------------------
def fit_dynamics(tau_res, qdd, qd, gyro=None, eps=0.05):
    """Fit  tau_res = J_eff*qdd + tau_c*tanh(qd/eps) + tau_v*qd + c0 [+G*gyro].

    tau_res [Nm] must already have the suspended gravity torque g(q) removed.
    gyro: optional (n,3) base angular velocity to absorb sling coupling
    (apply to serial joints whose step excites base motion — hips AND knees:
    knee gyro_rms 0.29 rad/s is the largest among serial joints).
    Returns dict(J_eff, tau_c, tau_v, c0, gyro_coef, R2, n, se_*) where se_*
    are the IRLS weighted-least-squares standard errors of the coefficients
    (cov = s^2 (X'WX)^-1), used by the J2 symmetry significance screen.
    """
    tau_res = _as_1d_float(tau_res)
    qdd = _as_1d_float(qdd)
    qd = _as_1d_float(qd)
    cols = [qdd, np.tanh(qd / float(eps)), qd, np.ones_like(qd)]
    names = ["J_eff", "tau_c", "tau_v", "c0"]
    if gyro is not None:
        g = np.asarray(gyro, dtype=float)
        if g.ndim == 2 and g.shape[0] == qd.size and g.shape[1] == 3:
            for k in range(3):
                cols.append(g[:, k])
                names.append(f"gyro_{chr(ord('x') + k)}")
    X = np.stack(cols, axis=1)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(tau_res)
    X, y = X[ok], tau_res[ok]
    beta, w = irls_huber_fit(X, y, n_iter=3)
    out = {names[i]: float(beta[i]) for i in range(len(names))}
    # weighted-LS coefficient covariance: cov = s^2 (X'WX)^-1 with
    # s^2 = sum(w*r^2)/(n-p)  (IRLS final-weights sandwich is unnecessary
    # here: the weights are deterministic functions of the residuals).
    try:
        XtWX = (X * w[:, None]).T @ X
        r = y - X @ beta
        s2 = float(np.sum(w * r * r)) / max(len(y) - X.shape[1], 1)
        cov = s2 * np.linalg.inv(XtWX)
        ses = np.sqrt(np.maximum(np.diag(cov), 0.0))
        for i, nm in enumerate(names):
            out[f"se_{nm}"] = float(ses[i])
    except np.linalg.LinAlgError:
        pass
    out["R2"] = float(r_squared(y, X @ beta))
    out["n"] = int(ok.sum())
    out["frac_weighted"] = float(np.mean(w < 0.999))
    return out


def xcorr_delay(u, y, dt, max_lag_s=0.03):
    """Lag [s] maximizing normalized cross-correlation corr(u(t-lag), y(t)).

    Positive lag = y responds AFTER u by lag seconds. Search ±max_lag_s.
    Returns (lag_s, peak_corr).
    """
    u = _as_1d_float(u)
    y = _as_1d_float(y)
    n = min(len(u), len(y))
    u, y = u[:n] - np.mean(u[:n]), y[:n] - np.mean(y[:n])
    denom = np.sqrt(float(np.sum(u * u)) * float(np.sum(y * y)))
    if denom < 1e-12:
        return 0.0, 0.0
    max_lag = int(round(max_lag_s / dt))
    best_lag, best = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            uu, yy = u[:n - lag] if lag else u, y[lag:]
        else:
            uu, yy = u[-lag:], y[:n + lag]
        c = float(np.sum(uu * yy)) / denom
        if c > best:
            best, best_lag = c, lag
    return best_lag * float(dt), float(best)


def fit_m1(tau, q_des, q, qd, kp, kd):
    """M1 actuation-gain regression tau ~ alpha*kp*(q_des-q) + beta*kd*qd + c.

    Returns dict(alpha, beta_d, c, R2, n) — compare against
    data/derived/step_m1_regression_all.json (serial joints alpha 0.34..0.71).
    """
    tau = _as_1d_float(tau)
    q_des = _as_1d_float(q_des)
    q = _as_1d_float(q)
    qd = _as_1d_float(qd)
    e = q_des - q
    X = np.stack([float(kp) * e, float(kd) * qd, np.ones_like(qd)], axis=1)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(tau)
    beta, _ = irls_huber_fit(X[ok], tau[ok], n_iter=2)
    out = {"alpha": float(beta[0]), "beta_d": float(beta[1]), "c": float(beta[2]),
           "R2": float(r_squared(tau[ok], X[ok] @ beta)), "n": int(ok.sum())}
    return out


def fit_kt(tau, current):
    """Torque-current mapping tau ~ kt*i + c0 (per-unit slope, R2, intercept).

    Note: absolute kt magnitude depends on the logger's current scaling
    (observed slope ~190 Nm per A-unit on knee data); gates therefore use
    R2 and left/right symmetry, not an absolute band.
    """
    tau = _as_1d_float(tau)
    i = _as_1d_float(current)
    X = np.stack([i, np.ones_like(i)], axis=1)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(tau) & (np.abs(i) > 1e-6)
    beta, _ = irls_huber_fit(X[ok], tau[ok], n_iter=2)
    return {"kt": float(beta[0]), "c0": float(beta[1]),
            "R2": float(r_squared(tau[ok], X[ok] @ beta)), "n": int(ok.sum())}


# ---------------------------------------------------------------------------
# Column-semantics guard (run-time evidence, not assumption)
# ---------------------------------------------------------------------------
def detect_feedback_semantics(cmd, pos, tau, phase_pre="pre_hold", phases=None):
    """Verify (with printed evidence) that `pos`/`tau` columns are MEASURED
    feedback while `cmd` is the step command.

    E1: pre-hold段 cmd==0 但 pos 有活噪声 (std>0) -> pos 是反馈不是目标
    E2: cmd 跳变样本处 pos 不跳变 (|dpos| 小) -> 命令与反馈解耦
    E3: cmd 跳变后短窗 tau 出现 >=3x 瞬态 -> tau 是响应反馈
    Returns (is_feedback: bool, evidence: dict)。phases: bool mask of pre-hold.
    """
    cmd = _as_1d_float(cmd)
    pos = _as_1d_float(pos)
    tau = _as_1d_float(tau)
    n = len(cmd)
    pre = np.asarray(phases) if phases is not None else np.zeros(n, bool)
    ev = {}
    if pre.any():
        ev["E1_prehold_pos_std"] = float(np.std(pos[pre]))
        ev["E1_prehold_cmd_absmax"] = float(np.max(np.abs(cmd[pre])))
    dcmd = np.abs(np.diff(cmd))
    jumps = np.where(dcmd > 0.02)[0] + 1
    ev["E2_n_cmd_jumps"] = int(len(jumps))
    if len(jumps):
        dpos_at_jump = np.abs(np.diff(pos))[np.clip(jumps - 1, 0, n - 2)]
        ev["E2_dpos_at_jump_max"] = float(np.max(dpos_at_jump)) if len(dpos_at_jump) else None
        tau_base = (float(np.median(np.abs(tau[:-1][np.abs(dcmd) < 1e-9])))
                    if (np.abs(dcmd) < 1e-9).any() else 0.0)
        peaks = []
        for j in jumps[:20]:
            w = tau[j:min(j + 30, n)]
            if len(w):
                peaks.append(float(np.max(np.abs(w))))
        ev["E3_post_jump_tau_peak_med"] = float(np.median(peaks)) if peaks else None
        ev["E3_tau_quiet_median"] = float(tau_base)
    is_fb = (ev.get("E1_prehold_pos_std", 0) > 1e-5
             and ev.get("E2_n_cmd_jumps", 0) > 0
             and (ev.get("E2_dpos_at_jump_max", 1.0) or 1.0) < 0.02
             and (ev.get("E3_post_jump_tau_peak_med", 0) or 0)
             > 2.0 * max(ev.get("E3_tau_quiet_median", 0), 1e-3))
    return bool(is_fb), ev
