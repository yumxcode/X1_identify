import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from spi.param_space import (phi_search_box, phi_to_physical, physical_to_phi,
                             phi_to_U, tanh_motor_torque, ParamSpace, BodyParams,
                             MotorGroup, physical_violations, com_bounds,
                             project_to_physical_domain,
                             physical_range_penalty)

# X1 pelvis nominal (from xyber_x1_serial.xml)
M0 = 4.3041648
COM0 = np.array([0.00252285, -0.00063439, 0.03023409])
I0 = np.array([[0.02680559, -5.49e-06, 5.389e-05],
               [-5.49e-06, 0.01083128, -0.00011229],
               [5.389e-05, -0.00011229, 0.02180955]])


class TestLogCholesky(unittest.TestCase):
    def test_roundtrip(self):
        phi = physical_to_phi(M0, COM0, I0)
        p = phi_to_physical(phi)
        self.assertAlmostEqual(p["mass"], M0, places=9)
        np.testing.assert_allclose(p["com"], COM0, atol=1e-9)
        np.testing.assert_allclose(p["inertia"], I0, atol=1e-9)

    def test_any_phi_is_feasible(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            phi = rng.normal(scale=1.0, size=10)
            p = phi_to_physical(phi)
            self.assertGreater(p["mass"], 0.0)
            lam = np.linalg.eigvalsh(p["inertia"])
            self.assertGreater(lam.min(), -1e-12)   # inertia PSD
            # fullinertia triangle inequality on diagonal
            d = np.diag(p["inertia"])
            self.assertTrue(np.all(d.sum() >= 2 * d - 1e-12))

    def test_U_upper_triangular_positive_diag(self):
        rng = np.random.default_rng(1)
        U = phi_to_U(rng.normal(size=10) * 0.5)
        self.assertTrue(np.all(np.diag(U) > 0))
        self.assertTrue(np.allclose(U, np.triu(U)))

    def test_search_box_valid(self):
        box = phi_search_box({"mass": M0}, (2.0, 10.0), (-0.15, 0.15), (0.005, 1.0))
        self.assertEqual(box.shape, (10, 2))
        self.assertTrue(np.all(box[:, 0] < box[:, 1]))
        # mass bound maps exactly: alpha range == 0.5 log mass range
        self.assertAlmostEqual(box[0, 0], 0.5 * np.log(2.0), places=9)
        self.assertAlmostEqual(box[0, 1], 0.5 * np.log(10.0), places=9)
        # a nominal-mass phi with zero com lands inside the com (t) box
        phi0 = physical_to_phi(M0, [0, 0, 0], np.diag([0.02, 0.012, 0.022]))
        self.assertTrue(np.all(phi0[7:10] >= box[7:10, 0] - 1e-12))
        self.assertTrue(np.all(phi0[7:10] <= box[7:10, 1] + 1e-12))


class TestMotorModel(unittest.TestCase):
    def test_small_command_linear(self):
        tau = tanh_motor_torque(np.array([1.0, 5.0]), kappa=100.0)
        np.testing.assert_allclose(tau, [1.0, 5.0], rtol=1e-2)

    def test_saturation(self):
        tau = tanh_motor_torque(np.array([1e4]), kappa=20.0)
        self.assertLess(tau[0], 20.0 * 1.001)
        self.assertGreater(tau[0], 19.0)
        # kappa_s linear gain
        tau2 = tanh_motor_torque(np.array([1.0]), kappa=100.0, kappa_s=1.5)
        self.assertAlmostEqual(tau2[0], 1.5, delta=1e-3)


class TestParamSpace(unittest.TestCase):
    def _space(self):
        body = BodyParams("base", {"mass": M0, "com": COM0, "inertia": I0},
                          (2.0, 10.0), (-0.15, 0.15), (0.005, 1.0))
        mg = MotorGroup("knee", ["left_knee_pitch_joint"], 120.0, (40.0, 160.0))
        return ParamSpace(bodies=[body], motor_groups=[mg])

    def test_regularization_zero_at_nominal(self):
        s = self._space()
        self.assertAlmostEqual(s.regularization(s.nominal_params()), 0.0, places=9)

    def test_regularization_grows(self):
        s = self._space()
        p = s.nominal_params()
        p["bodies"]["base"] = {"mass": M0 + 1.0, "com": COM0 + 0.01, "inertia": I0 * 1.1}
        self.assertGreater(s.regularization(p), 0.0)

    def test_dim(self):
        s = self._space()
        self.assertEqual(s.dim, 10 + 1 + 1)

    def test_to_json(self):
        import json
        s = self._space()
        d = json.loads(s.to_json(s.nominal_params()))
        self.assertAlmostEqual(d["bodies"]["base"]["mass"], M0, places=6)


class TestPhysicalRangePenalty(unittest.TestCase):
    def _cfg_body(self):
        return {"name": "base", "mass_range": (3.0, 5.5),
                "com_range": (-0.06, 0.06), "inertia_diag_range": (0.005, 0.15)}

    def _params(self, mass=M0, com=None, inertia=None):
        return {"bodies": {"base": {"mass": mass,
                                    "com": np.asarray(com if com is not None else COM0),
                                    "inertia": np.asarray(inertia if inertia is not None else I0)}}}

    def test_nominal_no_violation(self):
        cfg = [self._cfg_body()]
        self.assertEqual(physical_violations(self._params(), cfg), {})
        self.assertEqual(physical_range_penalty(self._params(), cfg), 0.0)

    def test_out_of_range_detected(self):
        cfg = [self._cfg_body()]
        # 旧首轮辨识结果：质量 6.97 kg、com_y/z ±0.19/-0.20、惯量 1.5-2.0 —— 必须被标记
        bad = self._params(mass=6.97,
                           com=[0.06, 0.19, -0.20],
                           inertia=np.diag([1.5, 1.2, 2.0]))
        viol = physical_violations(bad, cfg)
        self.assertIn("base", viol)
        v = viol["base"]
        self.assertGreater(v["mass"], 1.4)
        self.assertIn("com_y", v)
        self.assertIn("inertia_z", v)
        self.assertGreater(physical_range_penalty(bad, cfg), 1e4)

    def test_inertia_boundary_in_range(self):
        cfg = [self._cfg_body()]
        p = self._params(inertia=np.diag([0.13, 0.02, 0.14]))
        self.assertEqual(physical_violations(p, cfg), {})

    def test_offdiag_violation_detected(self):
        # v13 漏洞复现：特征值在域内但非对角 -0.082（主轴旋转 ~40°）必须被标记
        cfg = [dict(self._cfg_body(), inertia_offdiag_max=0.03)]
        bad_I = np.array([[0.233, 0.043, 0.020],
                          [0.043, 0.093, -0.082],
                          [0.020, -0.082, 0.253]])
        bad = self._params(inertia=bad_I)
        viol = physical_violations(bad, cfg)
        self.assertIn("base", viol)
        self.assertIn("inertia_yz", viol["base"])
        self.assertAlmostEqual(viol["base"]["inertia_yz"], 0.052, places=3)
        self.assertGreater(physical_range_penalty(bad, cfg), 1e2)
        # 无该配置键时不检查（向后兼容；矩阵特征值须在 diag 域内以隔离该效应）
        odd = self._params(inertia=np.array([[0.06, 0.04, 0.0],
                                             [0.04, 0.06, 0.0],
                                             [0.0, 0.0, 0.05]]))
        self.assertEqual(physical_violations(odd, [self._cfg_body()]), {})
        # 小非对角（<= 上界）不违规
        ok = self._params(inertia=np.array([[0.02, 0.01, 0.0],
                                            [0.01, 0.02, 0.0],
                                            [0.0, 0.0, 0.02]]))
        self.assertEqual(physical_violations(ok, cfg), {})


class TestProjectToPhysicalDomain(unittest.TestCase):
    """辨识后物理一致投影（Oaki 2026 SDP-projection 同型操作）。"""

    def _cfg_body(self):
        return {"name": "base", "mass_range": (3.0, 5.5),
                "com_range": (-0.06, 0.06), "inertia_diag_range": (0.005, 0.38),
                "inertia_offdiag_max": 0.03}

    def test_r8_scenario_offdiag_projected(self):
        # R8 实测场景：xy 惯量积 -0.0470 超域 0.017，其余项域内
        I = np.array([[0.07653347, -0.04703821, -0.01323961],
                      [-0.04703821, 0.10682137, -0.00102668],
                      [-0.01323961, -0.00102668, 0.15859107]])
        params = {"bodies": {"base": {"mass": 4.0312, "com": np.array([0.0, 0.055, 0.006]),
                                      "inertia": I}}}
        proj = project_to_physical_domain(params, [self._cfg_body()])
        I2 = proj["bodies"]["base"]["inertia"]
        self.assertLessEqual(abs(I2[0, 1]), 0.03 + 1e-12)
        self.assertLessEqual(abs(I2[0, 2]), 0.03 + 1e-12)
        self.assertLessEqual(abs(I2[1, 2]), 0.03 + 1e-12)
        # 投影后无 violation（PHYSICAL 门禁通过条件）
        self.assertEqual(physical_violations(proj, [self._cfg_body()]), {})
        # 域内分量不动（xz/yz 本来就 < 0.03）
        self.assertAlmostEqual(I2[0, 2], I[0, 2], places=12)
        # 特征值保持域内且为正
        lam = np.linalg.eigvalsh(I2)
        self.assertTrue(np.all(lam >= 0.005 - 1e-12))
        self.assertTrue(np.all(lam <= 0.38 + 1e-12))

    def test_in_domain_params_unchanged(self):
        params = {"bodies": {"base": {"mass": M0, "com": COM0.copy(),
                                      "inertia": I0.copy()}}}
        proj = project_to_physical_domain(params, [self._cfg_body()])
        self.assertAlmostEqual(proj["bodies"]["base"]["mass"], M0, places=12)
        self.assertTrue(np.allclose(proj["bodies"]["base"]["com"], COM0))
        self.assertTrue(np.allclose(proj["bodies"]["base"]["inertia"], I0, atol=1e-9))

    def test_all_axes_clipped(self):
        # 三轴全超 + 质量超 + com 超：全部投影回域
        I = np.array([[0.05, -0.08, 0.05],
                      [-0.08, 0.05, -0.06],
                      [0.05, -0.06, 0.40]])
        params = {"bodies": {"base": {"mass": 9.0, "com": np.array([0.1, -0.1, 0.2]),
                                      "inertia": I}}}
        proj = project_to_physical_domain(params, [self._cfg_body()])
        b = proj["bodies"]["base"]
        self.assertTrue(3.0 <= b["mass"] <= 5.5)
        self.assertTrue(np.all(np.abs(b["com"]) <= 0.06))
        self.assertEqual(physical_violations(proj, [self._cfg_body()]), {})

    def test_motors_and_kappa_s_pass_through(self):
        params = {"bodies": {"base": {"mass": M0, "com": COM0.copy(),
                                      "inertia": I0.copy()}},
                  "motors": {"knee": 127.4}, "kappa_s": 0.361}
        proj = project_to_physical_domain(params, [self._cfg_body()])
        self.assertEqual(proj["motors"]["knee"], 127.4)
        self.assertAlmostEqual(proj["kappa_s"], 0.361)


class TestComDeltaAndMultiBody(unittest.TestCase):
    """com_delta 相对域（躯干等标称 com 远离原点的刚体）+ 二刚体空间。"""

    TORSO = {"name": "torso", "mjcf_body": "link_lumbar_pitch",
             "nominal": {"mass": 9.08107,
                         "com": [-0.000617851, 0.206789, -0.00114246],
                         "inertia": [[0.15447, -0.00586, 0.00037],
                                     [-0.00586, 0.0625, -0.00074],
                                     [0.00037, -0.00074, 0.11822]]},
             "mass_range": (6.0, 12.5), "com_delta": 0.06,
             "inertia_diag_range": (0.03, 0.40), "inertia_offdiag_max": 0.02}

    def test_com_bounds_relative(self):
        lo, hi = com_bounds(self.TORSO)
        self.assertTrue(np.allclose(lo, np.array(self.TORSO["nominal"]["com"]) - 0.06))
        self.assertTrue(np.allclose(hi, np.array(self.TORSO["nominal"]["com"]) + 0.06))

    def test_torso_nominal_not_violating(self):
        # 绝对对称域会把标称 com_y=0.207 判越界；com_delta 语义下标称合法
        p = {"bodies": {"torso": dict(self.TORSO["nominal"])}}
        self.assertEqual(physical_violations(p, [self.TORSO]), {})

    def test_torso_com_shift_violates(self):
        n = dict(self.TORSO["nominal"])
        n["com"] = list(np.array(n["com"]) + np.array([0.0, 0.2, 0.0]))
        p = {"bodies": {"torso": n}}
        viol = physical_violations(p, [self.TORSO])
        self.assertIn("torso", viol)
        self.assertIn("com_y", viol["torso"])

    def test_projection_respects_delta(self):
        n = dict(self.TORSO["nominal"])
        n["com"] = list(np.array(n["com"]) + np.array([0.0, 0.5, 0.0]))
        n["mass"] = 20.0
        proj = project_to_physical_domain({"bodies": {"torso": n}}, [self.TORSO])
        t = proj["bodies"]["torso"]
        self.assertLessEqual(t["com"][1], 0.206789 + 0.06 + 1e-12)
        self.assertLessEqual(t["mass"], 12.5)
        self.assertEqual(physical_violations(proj, [self.TORSO]), {})

    def test_phi_box_maps_relative_com(self):
        body = BodyParams(body_name="torso",
                          nominal={"mass": 9.08107,
                                   "com": np.array([-0.000617851, 0.206789, -0.00114246]),
                                   "inertia": np.diag([0.0625, 0.1182, 0.1545])},
                          mass_range=(6.0, 12.5), com_range=(-99, 99),
                          inertia_diag_range=(0.03, 0.40), com_delta=0.06)
        c_lo, c_hi = body.com_box()
        self.assertAlmostEqual(c_lo[1], 0.206789 - 0.06, places=12)
        # per-axis box through phi_search_box: t_i = r_i * exp(alpha_nom)
        box = phi_search_box({"mass": 9.08107}, (6.0, 12.5), (c_lo, c_hi),
                             (0.03, 0.40))
        a_nom = 0.5 * np.log(9.08107)
        self.assertAlmostEqual(box[8, 0], (0.206789 - 0.06) * np.exp(a_nom), places=9)
        self.assertAlmostEqual(box[8, 1], (0.206789 + 0.06) * np.exp(a_nom), places=9)

    def test_two_body_space_builds_and_samples(self):
        import yaml
        cfg = yaml.safe_load(Path(__file__).resolve().parents[1].joinpath(
            "configs", "x1_spi.yaml").read_text())
        self.assertEqual([b["name"] for b in cfg["bodies"]], ["base", "torso"])
        from spi.optimizer import build_space
        space = build_space(cfg)
        self.assertEqual(space.dim, 10 * 2 + len(cfg["motor_groups"]) + 1)
        nom = space.nominal_params()
        self.assertIn("base", nom["bodies"])
        self.assertIn("torso", nom["bodies"])
        self.assertAlmostEqual(nom["bodies"]["torso"]["mass"], 9.08107, places=5)
        # nominal satisfies its own physical domain (com_delta semantics)
        self.assertEqual(physical_violations(nom, cfg["bodies"]), {})
        # optuna FixedTrial sampling within the phi box stays inside domain:
        # mass axis = middle of the box
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        mid = {}
        for b in space.bodies:
            c_lo, c_hi = b.com_box()
            a_nom = 0.5 * np.log(b.nominal["mass"])
            for i, n in enumerate(["alpha", "d1", "d2", "d3"]):
                pass
            mid[f"{b.body_name}.alpha"] = a_nom
        trial = optuna.trial.FixedTrial(mid)
        # full mid-box sampling: use nominal phi (guaranteed in-box)
        for b in space.bodies:
            for n, v in zip(["alpha", "d1", "d2", "d3", "s12", "s13", "s23",
                             "t1", "t2", "t3"], b.phi_nominal):
                mid[f"{b.body_name}.{n}"] = float(v)
        for g in space.motor_groups:
            mid[f"kappa.{g.name}"] = g.kappa_nominal
        mid["kappa_s"] = space.kappa_s_nominal
        trial = optuna.trial.FixedTrial(mid)
        params = space.sample(trial)
        self.assertEqual(physical_violations(params, cfg["bodies"]), {})

    def test_body_map_from_cfg(self):
        import yaml
        from spi.rollout import body_map_from_cfg
        cfg = yaml.safe_load(Path(__file__).resolve().parents[1].joinpath(
            "configs", "x1_spi.yaml").read_text())
        bm = body_map_from_cfg(cfg)
        self.assertEqual(bm["base"], cfg["model"]["base_body"])
        self.assertEqual(bm["torso"], "link_lumbar_pitch")
        # legacy single-body configs keep the default mapping
        legacy = dict(cfg, bodies=[{"name": "base"}])
        self.assertEqual(body_map_from_cfg(legacy), {"base": cfg["model"]["base_body"]})


if __name__ == "__main__":
    unittest.main()
