"""X1 dynamics core for PRIME-style identification.

Physics model (PRIME arXiv:2605.17681 Eq. 1):
    q in SE(3) x R^12 (floating base + 12 leg joints), v in R^18
    semi-implicit Euler:
        M(q)(v+ - v) = dt*(B u - h(q,v) + sum_i J_i(q)^T f_i)

Smoothed contact (PRIME Eq. 17-20, Pang et al. log-barrier Anitescu):
    v+ = argmin_{w} 0.5*||w - v_free||^2_M - (1/kappa)*sum_i log(s_i(w))
    s_i = (phi_i/dt + J_i^n w)^2/mu^2 - ||J_i^t w||^2
    v_free = v + dt*M^-1*(B u - h)
Optimality (impulse form):
    g(w) = M(w - v_free) - sum_i J_i^T lambda_i(w) = 0
    lambda_i^n = 2(phi_i/dt + J_i^n w)/(mu^2 kappa s_i)
    lambda_i^t = -2 J_i^t w/(kappa s_i)

Jacobian of the map is obtained via the implicit-function theorem on g=0
(analytic w.r.t. w; parameter part via pinocchio's joint-torque regressor,
which is exact and linear in the inertial parameters).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pinocchio as pin

from .log_cholesky import (
    THETA_NAMES,
    jacobian_pi_theta,
    theta_to_pi,
)


#: joint order in the URDF == order used by the walk_diag CSV
JOINT_ORDER: List[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

#: two contact points per foot (toe front/back), as in the MJCF toe bodies
CONTACT_FRAMES: List[str] = [
    "leg_l_toe_a_loop",
    "leg_l_toe_b_loop",
    "leg_r_toe_a_loop",
    "leg_r_toe_b_loop",
]

#: per-body inertia parameter ordering used by pinocchio's regressor
PIN_PARAM_SLICES = (0, 1, 4, 5, 6, 7, 8, 9)  # [m, mc, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]


@dataclass
class SymmetryGroups:
    """Bodies whose parameters are tied together (left/right symmetric legs).

    Maps each identified body index (in `bodies`) to a group id. Bodies in
    the same group share one theta.
    """

    groups: List[List[int]]

    def group_of(self, body: int) -> int:
        for gi, g in enumerate(self.groups):
            if body in g:
                return gi
        raise KeyError(body)


#: inertia-6 ordering everywhere inside this package = pinocchio
#: toDynamicParameters: [m, mc_x, mc_y, mc_z, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]
_PC_TO_LC = [0, 1, 2, 3, 4, 6, 9, 5, 8, 7]  # -> [m,h,Ixx,Iyy,Izz,Ixy,Iyz,Ixz]
_LC_TO_PC = [0, 1, 2, 3, 4, 7, 5, 9, 8, 6]


def _vec6_to_inertia(v: np.ndarray) -> np.ndarray:
    """v = [Ixx, Ixy, Iyy, Ixz, Iyz, Izz] (pinocchio ordering)."""
    Ixx, Ixy, Iyy, Ixz, Iyz, Izz = v
    return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])


def _inertia_to_vec6(I: np.ndarray) -> np.ndarray:
    return np.array([I[0, 0], I[0, 1], I[1, 1], I[0, 2], I[1, 2], I[2, 2]])


#: symmetric index pairs for the 6-vector inertia (pinocchio order)
_SYMS = ((0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2))


def make_param_transform(R: np.ndarray) -> np.ndarray:
    """10x10 matrix B(R) mapping pi expressed in frame A to pi expressed in
    frame B, where R rotates vectors from A to B (pi_B = B @ pi_A):
        m' = m,  mc' = R mc,  I_o' = R I_o R^T  (about the origin).
    Left/right leg links are rotated ~180 deg w.r.t. each other in the X1
    URDF, so sharing group parameters REQUIRES this transform.
    """
    B = np.zeros((10, 10))
    B[0, 0] = 1.0
    B[1:4, 1:4] = R
    for i, (a, b) in enumerate(_SYMS):
        for j, (c, d) in enumerate(_SYMS):
            B[4 + i, 4 + j] += R[a, c] * R[b, d]
    return B


#: ankle bodies are EXCLUDED from identification: with only two toe
#: contact points per foot the ankle dofs are weakly constrained in the
#: contact QP (free hinge about the line joining the two points), which
#: injects huge spurious accelerations into the regression.
EXCLUDED_BODIES = (
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)


def pc_min_eig_params(m: float, h: np.ndarray, Io: np.ndarray) -> float:
    """Min eigenvalue of the pseudo-inertia built from ORIGIN-referenced
    parameters (pinocchio toDynamicParameters semantics):

        I4 = [[Sigma, h], [h^T, m]],  Sigma = 0.5*tr(Io)*Id3 - Io

    Physical consistency <=> I4 ≻ 0 (min eig > 0). Pure function (no
    pinocchio validation) so corrupted parameters can be unit-tested.

    NOTE: `log_cholesky.is_physically_consistent` assumes pi[4:10] is the
    COM inertia (its internal convention). Feeding origin-referenced values
    there silently checks a parallel-axis-shifted object — this caused a
    false "13/13 violations" diagnosis earlier. Use THESE functions for
    pinocchio-sourced parameters.
    """
    S = 0.5 * np.trace(Io) * np.eye(3) - Io
    I4 = np.zeros((4, 4))
    I4[:3, :3] = S
    I4[:3, 3] = h
    I4[3, :3] = h
    I4[3, 3] = m
    return float(np.linalg.eigvalsh(I4).min())


def pc_min_eig_origin(model: pin.Model, joint_id: int) -> float:
    """pc_min_eig_params for a pinocchio body (origin-referenced params)."""
    p = model.inertias[joint_id].toDynamicParameters()
    return pc_min_eig_params(p[0], p[1:4], _vec6_to_inertia(p[4:10]))


def _selftest_pc_check():
    """Bidirectional unit test for pc_min_eig_origin."""
    model = pin.buildModelFromUrdf(
        "/Users/yumx/code/robot_x/X1/X1_辨识/X1_train/resources/robots/x1/urdf/x1.urdf",
        pin.JointModelFreeFlyer(),
    )
    # positive case: known-PD body (root_joint, min eig +1.99e-2)
    e_root = pc_min_eig_origin(model, 1)
    assert e_root > 1e-3, f"root_joint should be PD, got {e_root}"
    # negative case: unphysical origin inertia (negative moment). Built via
    # the pure-function API because pin.Inertia itself validates positivity.
    p1 = model.inertias[1].toDynamicParameters()
    Io_bad = _vec6_to_inertia(p1[4:10]) + np.diag([-1.0, -1.0, -1.0])
    e_bad = pc_min_eig_params(p1[0], p1[1:4], Io_bad)
    assert e_bad <= 0, f"corrupted-inertia body should violate PC, got {e_bad}"
    print(f"pc_check selftest OK (root {e_root:+.2e}, degenerate {e_bad:+.2e})")


def default_identified_bodies(model: pin.Model) -> List[str]:
    """Bodies whose inertias we identify: all bodies except the ankles."""
    names = [model.names[j] for j in range(1, model.njoints)]
    return [n for n in names if n not in EXCLUDED_BODIES]


def build_symmetry(model: pin.Model) -> SymmetryGroups:
    """Group left/right leg bodies to share parameters (12 leg bodies -> 6)."""
    groups: List[List[int]] = [[1]]  # base body (root_joint, joint idx 1)
    joint_names = [model.names[j] for j in range(1, model.njoints)]
    for name in JOINT_ORDER:
        if name.startswith("left_"):
            right = "right_" + name[len("left_"):]
            li = joint_names.index(name) + 1
            ri = joint_names.index(right) + 1
            groups.append([li, ri])
    return SymmetryGroups(groups=groups)


class X1Dynamics:
    """18-DoF floating-base dynamics with smoothed contacts.

    Parameter vector layout (identification vector `theta`):
        theta = concat over groups g of theta_g (10, log-Cholesky)
    The inertias of all bodies in a group are set equal.
    """

    def __init__(
        self,
        urdf_path: str,
        mu: float = 0.8,
        kappa: float = 500.0,
        contact_frames: Optional[Sequence[str]] = None,
        identified_bodies: Optional[List[str]] = None,
        use_symmetry: bool = False,
        phi_active_thresh: float = 0.015,
    ):
        """use_symmetry: left/right legs are MIRROR-symmetric in the X1 URDF
        (det(R_rel) = -1), which cannot be represented by a proper rotation
        parameter transform. Symmetric tying is therefore DISABLED by
        default; each of the 13 bodies is identified independently (130
        params) with the prior regularizing weakly-excited directions.

        phi_active_thresh: contact gate. PRIME's log-barrier requires phi
        accurate to ~0.01 mm (mocap). Without mocap we gate contacts at this
        threshold and clamp phi to <= 0 for gated points, so contact forces
        are determined by dynamic balance, not by absolute phi accuracy."""
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.nq, self.nv = self.model.nq, self.model.nv
        self.nj = self.nv - 6
        assert self.nj == 12
        self.mu = mu
        self.kappa = kappa
        self.phi_active_thresh = phi_active_thresh

        joint_names = [self.model.names[j] for j in range(2, self.model.njoints)]
        assert joint_names == JOINT_ORDER, joint_names
        self.q_idx = np.arange(7, 7 + self.nj)  # joint positions in q
        self.v_joint = np.arange(6, 6 + self.nj)  # joint velocities in v

        self.contact_frame_ids = [
            self.model.getFrameId(f) for f in (contact_frames or CONTACT_FRAMES)
        ]
        self.n_contacts = len(self.contact_frame_ids)

        # bodies: joint indices (1..n-1) whose inertia is identified
        all_names = [self.model.names[j] for j in range(1, self.model.njoints)]
        id_names = identified_bodies or default_identified_bodies(self.model)
        self.bodies = [all_names.index(n) + 1 for n in id_names]
        self.n_bodies = len(self.bodies)

        if use_symmetry:
            self.sym = build_symmetry(self.model)
            # keep only groups whose bodies are identified
            self.groups = [
                g for g in self.sym.groups if all(b in self.bodies for b in g)
            ]
        else:
            self.groups = [[b] for b in self.bodies]
        self.n_groups = len(self.groups)
        self.n_theta = 10 * self.n_groups

        # per-body rotation relative to its group's first body (neutral q),
        # needed because left/right leg frames are rotated ~180 deg apart
        import pinocchio as _pin
        _q0 = _pin.neutral(self.model)
        _pin.forwardKinematics(self.model, self.data, _q0)
        _pin.updateFramePlacements(self.model, self.data)
        self.group_B: List[List[np.ndarray]] = []
        for grp in self.groups:
            R0 = self.data.oMi[grp[0]].rotation
            self.group_B.append(
                [make_param_transform(R0.T @ self.data.oMi[b].rotation) for b in grp]
            )

        # nominal (URDF) parameters -> prior theta_hat
        pi_raw = self._read_pi()
        # NOTE: the URDF nominal inertias are NOT physically consistent
        # (negative eigenvalues of the pseudo-inertia, SolidWorks export).
        # Project onto the physically-consistent set for the prior/theta_hat.
        self.pi_nominal = self._project_to_pc(pi_raw)
        self.pi_urdf = pi_raw
        self.theta_hat = self.pi_to_theta(self.pi_nominal)
        self._write_pi(self.pi_nominal)

        # workspace caches for step()
        self._Jc = np.zeros((3 * self.n_contacts, self.nv))
        self._phi = np.zeros(self.n_contacts)

    @staticmethod
    def _project_to_pc(pi: np.ndarray, eps_rel: float = 1e-3) -> np.ndarray:
        """Project per-body pi onto the physically-consistent set by clamping
        pseudo-inertia eigenvalues to eps_rel * max_eig (keeping mass)."""
        out = pi.copy()
        for k in range(pi.shape[0]):
            m = pi[k, 0]
            if m <= 0:
                raise ValueError(f"body {k}: non-positive mass {m}")
            h = pi[k, 1:4]
            Io = _vec6_to_inertia(pi[k, 4:10])
            c = h / m
            C = pin.skew(c)
            I_com = Io + m * (C @ C)
            I_origin = I_com - m * (C @ C)  # identity; Io is about origin
            S = 0.5 * np.trace(Io) * np.eye(3) - Io
            I4 = np.zeros((4, 4))
            I4[:3, :3] = S
            I4[:3, 3] = h
            I4[3, :3] = h
            I4[3, 3] = m
            w_, V = np.linalg.eigh(I4)
            floor = eps_rel * max(w_.max(), 1e-9)
            w_proj = np.maximum(w_, floor)
            I4p = V @ np.diag(w_proj) @ V.T
            mp = I4p[3, 3]
            hp = I4p[:3, 3]
            Sp = I4p[:3, :3]
            Io_p = np.trace(Sp) * np.eye(3) - Sp
            out[k, 0] = mp
            out[k, 1:4] = hp
            out[k, 4:10] = _inertia_to_vec6(Io_p)
        return out

    # ------------------------------------------------------------------
    # parameter plumbing
    # ------------------------------------------------------------------
    def _read_pi(self) -> np.ndarray:
        """pi for each *body* (len n_bodies, 10 each). Groups share values."""
        pi = np.zeros((self.n_bodies, 10))
        for k, b in enumerate(self.bodies):
            pi[k] = self.model.inertias[b].toDynamicParameters()[
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            ]
        return pi

    def _write_pi(self, pi: np.ndarray) -> None:
        """pi is per-body (n_bodies x 10), already expanded from groups."""
        for k, b in enumerate(self.bodies):
            m = pi[k, 0]
            mc = pi[k, 1:4]
            Io = np.array(
                [
                    [pi[k, 4], pi[k, 5], pi[k, 7]],
                    [pi[k, 5], pi[k, 6], pi[k, 8]],
                    [pi[k, 7], pi[k, 8], pi[k, 9]],
                ]
            )
            lever = mc / m
            C = pin.skew(lever)
            I_com = Io + m * (C @ C)  # origin -> COM
            I_com = 0.5 * (I_com + I_com.T)  # enforce exact symmetry
            self.model.inertias[b] = pin.Inertia(m, lever, I_com)

    # group-level theta <-> per-body pi ---------------------------------
    def theta_to_pi_groups(self, theta: np.ndarray) -> np.ndarray:
        """theta (n_theta) -> per-group pi (n_groups x 10, pinocchio order)."""
        lc = np.stack(
            [theta_to_pi(theta[10 * g : 10 * g + 10]) for g in range(self.n_groups)]
        )
        return lc[:, _LC_TO_PC]

    def pi_groups_to_pi_bodies(self, pi_g: np.ndarray) -> np.ndarray:
        """Expand per-group pi (in group-first-body frame) to per-body pi,
        applying the frame rotation for bodies rotated w.r.t. the group head."""
        pi = np.zeros((self.n_bodies, 10))
        for gi, grp in enumerate(self.groups):
            for bi, b in enumerate(grp):
                pi[self.bodies.index(b)] = self.group_B[gi][bi] @ pi_g[gi]
        return pi

    def pi_to_theta(self, pi: np.ndarray, regularize: float = 0.0) -> np.ndarray:
        """per-body pi (pinocchio order) -> theta via group averaging."""
        from .log_cholesky import pi_to_theta as p2t

        theta = np.zeros(self.n_theta)
        for gi, grp in enumerate(self.groups):
            pi_head = [
                np.linalg.solve(self.group_B[gi][bi], pi[self.bodies.index(b)])
                for bi, b in enumerate(grp)
            ]
            pi_avg = np.mean(pi_head, axis=0)[_PC_TO_LC]
            theta[10 * gi : 10 * gi + 10] = p2t(pi_avg)
        return theta

    def set_theta(self, theta: np.ndarray) -> None:
        pi_g = self.theta_to_pi_groups(theta)
        self._write_pi(self.pi_groups_to_pi_bodies(pi_g))

    def total_mass(self) -> float:
        return float(pin.computeTotalMass(self.model))

    def joint_index(self, joint_name: str) -> int:
        """v-index (6-based) of an actuated joint, with or without the
        '_joint' suffix."""
        if not joint_name.endswith("_joint"):
            joint_name = joint_name + "_joint"
        return 6 + JOINT_ORDER.index(joint_name)

    def selfcheck(self, tol: float = 1e-9) -> float:
        """Round-trip self-test: pi_nominal -> theta_hat -> set_theta ->
        read back and compare per-body. MUST pass before any identification.
        Returns the max abs error (must be <= tol)."""
        pi_before = self.pi_nominal.copy()
        self.set_theta(self.theta_hat)
        pi_after = self._read_pi()
        err = float(np.abs(pi_before - pi_after).max())
        self.set_theta(self.theta_hat)
        if err > tol:
            raise AssertionError(
                f"parameter plumbing round-trip failed: max err {err:.3e} > {tol:.0e}"
                " (check group frame transforms)"
            )
        return err

    # ------------------------------------------------------------------
    # kinematics / dynamics terms
    # ------------------------------------------------------------------
    def kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Contact Jacobians (LOCAL_WORLD_ALIGNED, 3nc x nv) and heights phi."""
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        for i, fid in enumerate(self.contact_frame_ids):
            self._Jc[3 * i : 3 * i + 3] = pin.computeFrameJacobian(
                self.model, self.data, q, fid, pin.LOCAL_WORLD_ALIGNED
            )[:3]
            self._phi[i] = self.data.oMf[fid].translation[2]
        return self._Jc.copy(), self._phi.copy()

    def M_and_h(self, q: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        M = pin.crba(self.model, self.data, q).copy()
        h = pin.nonLinearEffects(self.model, self.data, q, v).copy()
        return M, h

    def regressor(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Y (nv x 10*(njoints-1)) with tau = Y @ p, p = concat per-body
        toDynamicParameters() (about link ORIGIN, [m, mc, Ixx,Ixy,Iyy,Ixz,Iyz,Izz])."""
        return pin.computeJointTorqueRegressor(self.model, self.data, q, v, a).copy()

    # ------------------------------------------------------------------
    # smoothed-contact forward step (PRIME Eq. 17-20)
    # ------------------------------------------------------------------
    @staticmethod
    def _split_J(J_row: np.ndarray, mu: float) -> Tuple[np.ndarray, np.ndarray]:
        """For a 3 x nv contact Jacobian in world-aligned frame with z up:
        normal row = z row; tangential = x,y rows."""
        return J_row[2], J_row[:2]

    def solve_contact_step(
        self,
        q: np.ndarray,
        v: np.ndarray,
        u: np.ndarray,
        dt: float,
        max_iter: int = 50,
        tol: float = 1e-10,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """One-step dynamics with contacts.

        Anitescu convex contact QP (the kappa->infty rigid limit of the
        paper's smoothed problem, which we adopt for robustness to phi
        uncertainty; the one-sided log-barrier variant is kept in
        solve_contact_step_barrier for reference/initialization):

            min_w 0.5||w - v_free||^2_M   s.t.  J w + phi/dt in K*_mu
        dual (in impulses lambda in K_mu):
            min_lambda 0.5 lam^T A lam + lam^T (J v_free + g),
            A = J M^-1 J^T,  g_i = [0, 0, phi_i/dt]
            w = v_free + M^-1 J^T lam

        solved by ADMM with exact SOC projections (smooth, unique, and the
        total normal force is set by dynamic balance regardless of phi noise).

        Returns (v_plus, impulses (nc x 3) [fx, fy, fz] = dt*force, converged).
        """
        Jc, phi_raw = self.kinematics(q)
        active = phi_raw < self.phi_active_thresh
        # gated points are treated as on-ground (phi clamped at 0, negative
        # kept: estimated penetration increases the preload)
        phi = np.where(active, np.minimum(phi_raw, 0.0), phi_raw)
        M, h = self.M_and_h(q, v)
        Bu = np.zeros(self.nv)
        Bu[self.v_joint] = u
        Minv_vfree = dt * np.linalg.solve(M, Bu - h)
        v_free = v + Minv_vfree

        nc = self.n_contacts
        mu = self.mu
        # assemble J (3nc x nv) for ACTIVE points only, rows ordered [x y z]
        act_idx = np.where(active)[0]
        J = Jc.copy()
        # zero out inactive rows => their lambda stays 0
        for i in range(nc):
            if not active[i]:
                J[3 * i : 3 * i + 3] = 0.0

        A = J @ np.linalg.solve(M, J.T)  # 3nc x 3nc
        # rank deficiency: two coplanar points per foot admit an internal
        # force along the line joining them (1-dim per foot). Regularize to
        # select the minimum-norm internal force (standard practice).
        eps_reg = 1e-3 * max(np.trace(A) / A.shape[0], 1e-12)
        A = A + eps_reg * np.eye(A.shape[0])
        b = J @ v_free
        for i in range(nc):
            if active[i]:
                b[3 * i + 2] += phi[i] / dt

        lam = self._solve_soc_qp(A, b, mu)
        impulses = lam.reshape(nc, 3)
        v_plus = v_free + np.linalg.solve(M, J.T @ lam)
        converged = bool(getattr(self, "_last_pgs_converged", False))
        return v_plus, impulses, converged

    def _proj_soc(self, z: np.ndarray, mu: float) -> np.ndarray:
        """Projection of stacked impulses z (nc x 3) onto the friction cone
        K = {(ln, lt): ||lt|| <= mu*ln} (second-order cone projection)."""
        out = z.copy()
        nc = z.shape[0]
        for i in range(nc):
            ln = z[i, 2]
            lt = z[i, :2]
            nt = np.linalg.norm(lt)
            if mu * ln >= nt:
                continue  # inside cone
            if nt <= -mu * ln:  # inside polar cone
                out[i] = 0.0
            else:
                s = (nt + mu * ln) / (1 + mu**2)
                out[i, 2] = s * mu
                out[i, :2] = s * lt / max(nt, 1e-300)
        return out

    def _solve_soc_qp(
        self, A: np.ndarray, b: np.ndarray, mu: float, iters: int = 3000,
        lam0: "np.ndarray | None" = None
    ) -> np.ndarray:
        """min 0.5 lam^T A lam + b^T lam  s.t. lam in K_mu.

        Block-projected Gauss-Seidel (Drake-style PGS/TGS contact solver):
        per-block projected gradient with diagonal preconditioning.
        """
        n = A.shape[0]
        nc = n // 3
        lam = np.zeros(n) if lam0 is None else lam0.copy()
        r = np.zeros(nc)
        for i in range(nc):
            dg = A[3 * i : 3 * i + 3, 3 * i : 3 * i + 3]
            r[i] = 1.0 / max(np.trace(dg) / 3.0, 1e-12)
        converged = False
        for _ in range(iters):
            lam_prev = lam.copy()
            for i in range(nc):
                g = A[3 * i : 3 * i + 3] @ lam + b[3 * i : 3 * i + 3]
                li = lam[3 * i : 3 * i + 3] - r[i] * g
                lam[3 * i : 3 * i + 3] = self._proj_soc(li[None, :], mu)[0]
            if np.linalg.norm(lam - lam_prev) < 1e-7 * (1 + np.linalg.norm(lam)):
                converged = True
                break
        self._last_pgs_converged = converged
        return lam

    def step(
        self, q: np.ndarray, v: np.ndarray, u: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Semi-implicit Euler step with smoothed contacts.
        Returns (v_plus, impulses)."""
        v_plus, impulses, _ = self.solve_contact_step(q, v, u, dt)
        # q+ integration (freeflyer: log/exp maps)
        q_plus = pin.integrate(self.model, q, dt * v_plus)
        return v_plus, impulses
