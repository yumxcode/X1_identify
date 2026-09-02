import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from joint_identify.regress import (detect_feedback_semantics, fit_dynamics,
                                    fit_kt, fit_m1, irls_huber_fit,
                                    r_squared, savgol_ddq, xcorr_delay)
from joint_identify.scripts.validate_joint import evaluate, is_serial

RNG = np.random.default_rng(7)


def _synth_dynamics(n=6000, dt=1e-3, J=0.35, tau_c=1.2, tau_v=0.25, c0=-0.1,
                    gyro_coef=None, noise=0.05):
    t = np.arange(n) * dt
    qd = 0.8 * np.sin(2 * np.pi * 1.5 * t) + 0.3 * np.sin(2 * np.pi * 0.4 * t + 1.0)
    qdd = np.gradient(qd, dt)
    tau = J * qdd + tau_c * np.tanh(qd / 0.05) + tau_v * qd + c0
    gyro = None
    if gyro_coef is not None:
        gyro = np.stack([np.sin(2 * np.pi * 2.0 * t + k) for k in range(3)], axis=1)
        tau = tau + gyro @ np.asarray(gyro_coef)
    tau = tau + RNG.normal(0, noise, n)
    return tau, qdd, qd, gyro


class TestSavgol(unittest.TestCase):
    def test_recovers_constant_accel(self):
        dt = 1e-3
        t = np.arange(5000) * dt
        a = 3.7
        q = 0.5 * a * t ** 2 + 0.1 * t + 0.02
        qdd = savgol_ddq(q, dt, window=41)
        mid = qdd[200:-200]
        self.assertTrue(np.allclose(mid, a, rtol=1e-6, atol=1e-3),
                        msg=f"max err {np.max(np.abs(mid - a)):.2e}")


class TestIRLS(unittest.TestCase):
    def test_outlier_robustness(self):
        n = 2000
        X = np.stack([np.linspace(-1, 1, n), np.ones(n)], axis=1)
        beta_true = np.array([2.0, -0.5])
        y = X @ beta_true + RNG.normal(0, 0.01, n)
        y[:60] += 25.0  # 3% gross outliers
        beta, _ = irls_huber_fit(X, y, n_iter=3)
        self.assertLess(abs(beta[0] - 2.0), 0.05)
        self.assertLess(abs(beta[1] + 0.5), 0.05)


class TestFitDynamics(unittest.TestCase):
    def test_recovers_params(self):
        tau, qdd, qd, _ = _synth_dynamics()
        fit = fit_dynamics(tau, qdd, qd)
        self.assertLess(abs(fit["J_eff"] - 0.35) / 0.35, 0.08)
        self.assertLess(abs(fit["tau_c"] - 1.2) / 1.2, 0.08)
        self.assertLess(abs(fit["tau_v"] - 0.25) / 0.25, 0.15)
        self.assertLess(abs(fit["c0"] + 0.1), 0.05)
        self.assertGreater(fit["R2"], 0.95)

    def test_gyro_coupling_absorbed(self):
        tau, qdd, qd, gyro = _synth_dynamics(gyro_coef=[0.4, -0.3, 0.2])
        fit_no = fit_dynamics(tau, qdd, qd, gyro=None)
        fit_yes = fit_dynamics(tau, qdd, qd, gyro=gyro)
        self.assertGreater(fit_yes["R2"], fit_no["R2"])
        self.assertLess(abs(fit_yes["J_eff"] - 0.35) / 0.35, 0.08)


class TestDelay(unittest.TestCase):
    def test_recovers_known_lag(self):
        dt = 1e-3
        n = 8000
        u = RNG.normal(0, 1, n)
        kern = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # 7-sample delay
        y = np.convolve(u, kern, mode="full")[:n] + 0.05 * RNG.normal(0, 1, n)
        lag_s, corr = xcorr_delay(u, y, dt, max_lag_s=0.02)
        self.assertAlmostEqual(lag_s * 1e3, 7.0, delta=1.5)
        self.assertGreater(corr, 0.9)

    def test_zero_lag(self):
        dt = 1e-3
        u = RNG.normal(0, 1, 5000)
        lag_s, corr = xcorr_delay(u, u + 0.02 * RNG.normal(0, 1, 5000), dt)
        self.assertAlmostEqual(lag_s, 0.0, delta=2 * dt)


class TestM1AndKt(unittest.TestCase):
    def test_m1_recovers_alpha(self):
        n = 5000
        q_des = np.zeros(n)
        q = 0.1 * RNG.normal(0, 1, n).cumsum() * 0.01
        qd = np.gradient(q, 1e-3)
        kp, kd = 60.0, 2.0
        alpha, beta_d, c = 0.55, -0.13, -0.2
        tau = alpha * kp * (q_des - q) + beta_d * kd * qd + c + RNG.normal(0, 0.05, n)
        fit = fit_m1(tau, q_des, q, qd, kp, kd)
        self.assertLess(abs(fit["alpha"] - alpha), 0.02)
        self.assertGreater(fit["R2"], 0.95)

    def test_kt_recovers_slope(self):
        i = RNG.normal(0, 0.05, 4000)
        kt_true = 190.0
        tau = kt_true * i + 0.3 + RNG.normal(0, 0.4, 4000)
        fit = fit_kt(tau, i)
        self.assertLess(abs(fit["kt"] - kt_true) / kt_true, 0.02)
        self.assertGreater(fit["R2"], 0.99)


class TestSemantics(unittest.TestCase):
    def _make(self, feedback=True):
        n = 3000
        cmd = np.zeros(n)
        cmd[500::700] = 0.2  # repeated steps
        if feedback:
            # delayed smooth response: zeros first (actuation latency), then
            # exponential approach -> q does NOT jump at the command sample
            resp = np.concatenate([np.zeros(8), np.exp(-np.arange(80) / 25.0)])
            q = np.convolve(cmd, resp, mode="full")[:n] * 0.9
            tau = np.convolve(cmd, np.array([0] * 7 + [5.0] + [0] * 40), mode="full")[:n]
            tau += RNG.normal(0, 0.3, n)
            q += RNG.normal(0, 1e-4, n)
        else:  # columns ARE the targets: jump instantly with cmd, silent before
            q = cmd.copy()
            tau = 60.0 * cmd
        phases = np.zeros(n, bool)
        phases[:500] = True
        return cmd, q, tau, phases

    def test_feedback_detected(self):
        cmd, q, tau, pre = self._make(feedback=True)
        ok, ev = detect_feedback_semantics(cmd, q, tau, phases=pre)
        self.assertTrue(ok, msg=str(ev))

    def test_target_columns_rejected(self):
        cmd, q, tau, pre = self._make(feedback=False)
        ok, _ = detect_feedback_semantics(cmd, q, tau, phases=pre)
        self.assertFalse(ok)


def _mk_params(r2=0.8, alpha=0.55, delay=7.0, lr_factor=1.0):
    joints = {}
    for side in ("left_", "right_"):
        f = lr_factor if side == "right_" else 1.0
        for jn in ("hip_pitch_joint", "knee_pitch_joint"):
            joints[side + jn] = {
                "m1": {"alpha": alpha, "R2": 0.9, "n": 1000},
                "kt": {"kt": 190.0 * f, "c0": 0.1, "R2": 0.99, "n": 1000},
                "dynamics": {"J_eff": 0.35 * f, "tau_c": 1.2 * f, "tau_v": 0.25 * f,
                             "c0": -0.1, "R2": r2, "n": 1000},
                "delay_ms": delay,
            }
        for jn in ("ankle_pitch_joint",):
            joints[side + jn] = {
                "m1": {"alpha": 2.8, "R2": 0.5, "n": 1000},
                "kt": {"kt": 190.0 * f, "c0": 0.1, "R2": 0.99, "n": 1000},
                "dynamics": None, "delay_ms": None,
            }
    return joints


class TestGates(unittest.TestCase):
    M1 = {j: 0.55 for j in
          ("left_hip_pitch_joint", "left_knee_pitch_joint",
           "right_hip_pitch_joint", "right_knee_pitch_joint")}

    def test_all_pass(self):
        r = evaluate(_mk_params(), m1_alphas=self.M1)
        self.assertEqual(r["verdict"], "PASS", msg=json.dumps(r, ensure_ascii=False))

    def test_fail_on_low_r2(self):
        r = evaluate(_mk_params(r2=0.3), m1_alphas=self.M1)
        self.assertEqual(r["verdict"], "FAIL")
        self.assertFalse(next(c for c in r["checks"] if c["id"] == "J1_DYNAMICS_R2")["ok"])

    def test_fail_on_alpha_out_of_band(self):
        r = evaluate(_mk_params(alpha=0.85), m1_alphas=self.M1)
        self.assertFalse(next(c for c in r["checks"] if c["id"] == "J3_GAIN_BAND")["ok"])

    def test_fail_on_m1_deviation_now_warns(self):
        m1 = dict(self.M1, left_knee_pitch_joint=0.40)
        r = evaluate(_mk_params(), m1_alphas=m1)
        # M1 per-joint deviation is method-level systematics -> WARN, not FAIL
        self.assertEqual(r["verdict"], "PASS")
        self.assertTrue(any("vs M1" in w for w in r["warnings"]))

    def test_edge_alpha_just_outside_band_passes(self):
        # 0.72 is outside [0.34,0.71] but within the +-0.10 widened hard band
        r = evaluate(_mk_params(alpha=0.72), m1_alphas=self.M1)
        self.assertTrue(next(c for c in r["checks"] if c["id"] == "J3_GAIN_BAND")["ok"])

    def test_fail_on_delay_range(self):
        r = evaluate(_mk_params(delay=45.0), m1_alphas=self.M1)
        self.assertFalse(next(c for c in r["checks"] if c["id"] == "J4_DELAY")["ok"])

    def test_fail_on_asymmetry(self):
        r = evaluate(_mk_params(lr_factor=1.6), m1_alphas=self.M1)
        self.assertFalse(next(c for c in r["checks"] if c["id"] == "J2_LR_SYMMETRY")["ok"])

    def test_knee_below_target_is_warn_not_fail(self):
        r = evaluate(_mk_params(r2=0.62), m1_alphas=self.M1)
        self.assertEqual(r["verdict"], "PASS")
        self.assertTrue(any("below target" in w for w in r["warnings"]))

    def test_is_serial(self):
        self.assertTrue(is_serial("left_knee_pitch_joint"))
        self.assertFalse(is_serial("left_ankle_pitch_joint"))


if __name__ == "__main__":
    unittest.main()
