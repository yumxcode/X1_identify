import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

# rollout module imports cleanly without mujoco (lazy import); the pure
# stance-odometry solver is unit-testable locally (P0-2).
from spi.rollout import STANCE_Z_TOL, solve_stance_base_linvel


class TestStanceBaseLinvel(unittest.TestCase):
    def test_zero_joint_rates_give_zero_base(self):
        Jb = np.eye(3)
        Jj = 0.5 * np.eye(3)
        v = solve_stance_base_linvel(Jb, Jj, np.zeros(3))
        self.assertTrue(np.allclose(v, 0.0, atol=1e-12))

    def test_single_foot_exact_solution(self):
        # J_base = I, J_joints = A: v = -A qd exactly (square, invertible)
        rng = np.random.default_rng(3)
        A = rng.normal(size=(3, 6))
        qd = rng.normal(size=6)
        v = solve_stance_base_linvel(np.eye(3), A, qd)
        self.assertTrue(np.allclose(v, -A @ qd, atol=1e-10))

    def test_double_support_symmetric_cancel(self):
        # two feet with opposite joint contributions -> compromise near zero
        Jb = np.vstack([np.eye(3), np.eye(3)])
        Jj = np.vstack([0.5 * np.eye(3), -0.5 * np.eye(3)])
        qd = np.array([0.2, -0.1, 0.0])
        v = solve_stance_base_linvel(Jb, Jj, qd)
        self.assertLess(np.linalg.norm(v), 1e-8)

    def test_double_support_consistent_solution(self):
        # consistent constraints (both feet imply the same base velocity)
        Jb = np.vstack([np.eye(3), np.eye(3)])
        Aj = np.diag([1.0, 2.0, -0.5])
        Jj = np.vstack([Aj, Aj])
        qd = np.array([0.3, -0.2, 0.1])
        v = solve_stance_base_linvel(Jb, Jj, qd)
        self.assertTrue(np.allclose(v, -Aj @ qd, atol=1e-10))

    def test_forward_command_sign(self):
        # joint motion pushes the stance foot backward at 0.4 m/s relative to
        # the base; pinning the foot means the base moves FORWARD at 0.4 m/s
        # v_foot = v_base + J_j qd = 0  ->  v_base = -J_j qd = +0.4 x
        Jb = np.eye(3)
        Jj = np.diag([1.0, 0.0, 0.0])
        qd = np.array([-0.4, 0.0, 0.0])
        v = solve_stance_base_linvel(Jb, Jj, qd)
        self.assertAlmostEqual(v[0], 0.4, places=10)

    def test_stance_tol_constant(self):
        # documented double-support threshold (m)
        self.assertGreater(STANCE_Z_TOL, 0.0)
        self.assertLess(STANCE_Z_TOL, 0.1)


if __name__ == "__main__":
    unittest.main()
