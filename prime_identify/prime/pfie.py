"""PFIE: Parameter Full-Information Estimation for X1 (PRIME-style).

Fast solver = block-coordinate (IRLS) form of the paper's PFIE objective:

  iterate:
    1. contact sweep: for each frame k solve the Anitescu SOC-QP
       (v+_k, lam_k) = F(q_k, v_k, u_k; theta_i)   [uses current params]
    2. parameter regression: with lam_k and MEASURED v+ fixed, the
       one-step dynamics residual is LINEAR in pi (pinocchio regressor):
         r_k = Y(q_k, v_k, a_k) p  -  B u_k  -  J_k^T lam_k / dt ... (joint rows)
       solve weighted LS for p (stacked group params, theta via log-Cholesky
       linearization) with the Gaussian prior -> theta_{i+1}

  This is a Gauss-Seidel / IRLS fixed point of the joint state-parameter MAP.
  The contact forces are latent variables inferred from dynamics (paper's
  core idea), and the parameters stay physically consistent by construction.

Unobserved base linear velocity (no mocap): rows masked in the regression;
its second-order effect on joint dynamics is absorbed by the process noise.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pinocchio as pin

from .dynamics import X1Dynamics, _PC_TO_LC, _LC_TO_PC
from .data import WalkData


@dataclass
class PFIEConfig:
    sigma_v_joint: float = 0.004      # rad/s velocity noise
    sigma_v_gyro: float = 0.005       # rad/s gyro noise
    prior_scale: float = 1.0          # theta prior weight (per unit logvar)
    n_iters: int = 12
    step_stride: int = 1
    holdout: float = 0.3


@dataclass
class PFIEResult:
    theta: np.ndarray
    J_train: float = 0.0
    J_holdout: float = 0.0
    rmse_joint_train: float = 0.0     # Nm torque-residual RMSE
    rmse_joint_holdout: float = 0.0
    Fz_mean: float = 0.0
    elapsed_s: float = 0.0
    iters_used: int = 0
    converged: bool = False
    history: List[float] = field(default_factory=list)


class PFIE:
    """IRLS solver for the PFIE objective on walk data."""

    def __init__(self, dyn: X1Dynamics, wd: WalkData, cfg: Optional[PFIEConfig] = None):
        self.dyn = dyn
        self.wd = wd
        self.cfg = cfg or PFIEConfig()
        n = len(wd.t)
        idx = np.arange(0, n - 1, self.cfg.step_stride)
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(idx))
        n_hold = int(self.cfg.holdout * len(idx))
        self.idx_ho = np.sort(idx[perm[:n_hold]])
        self.idx_tr = np.sort(idx[perm[n_hold:]])
        # per-row regression weights: joint rows sigma_tau, base rows looser
        self.w_joint = 1.0 / 0.35**2        # Nm torque noise scale
        self.w_base = 1.0 / 3.0**2          # base rows (impulse balance) N s
        self.theta_hat = dyn.theta_hat.copy()
        self.prior_w = self.cfg.prior_scale

    # ------------------------------------------------------------------
    def _sweep(self, theta: np.ndarray, frames: np.ndarray):
        """Solve contacts for all frames; return lam (n,nc,3), v+ (n,nv),
        conv mask, J (n, 3nc, nv), and phi info."""
        dyn, wd = self.dyn, self.wd
        dyn.set_theta(theta)
        n = len(frames)
        lam = np.zeros((n, dyn.n_contacts, 3))
        vplus = np.zeros((n, dyn.nv))
        Jc_all = np.zeros((n, 3 * dyn.n_contacts, dyn.nv))
        conv = np.zeros(n, dtype=bool)
        active = np.zeros((n, dyn.n_contacts), dtype=bool)
        lam_ws = None
        for i, k in enumerate(frames):
            Jc, phi_raw = dyn.kinematics(wd.q[k])
            Jc_all[i] = Jc
            active[i] = phi_raw < dyn.phi_active_thresh
            J = Jc.copy()
            for c in range(dyn.n_contacts):
                if not active[i, c]:
                    J[3 * c : 3 * c + 3] = 0.0
            M, h = dyn.M_and_h(wd.q[k], wd.v[k])
            Bu = np.zeros(dyn.nv)
            Bu[dyn.v_joint] = wd.u[k]
            v_free = wd.v[k] + wd.dt * np.linalg.solve(M, Bu - h)
            A = J @ np.linalg.solve(M, J.T)
            eps_reg = 1e-3 * max(np.trace(A) / A.shape[0], 1e-12)
            A = A + eps_reg * np.eye(A.shape[0])
            b = J @ v_free
            for c in range(dyn.n_contacts):
                if active[i, c]:
                    b[3 * c + 2] += min(phi_raw[c], 0.0) / wd.dt
            lk = dyn._solve_soc_qp(A, b, dyn.mu, lam0=lam_ws)
            lam_ws = lk if dyn._last_pgs_converged else None
            lam[i] = lk.reshape(dyn.n_contacts, 3)
            vplus[i] = v_free + np.linalg.solve(M, J.T @ lk)
            conv[i] = dyn._last_pgs_converged
        return lam, vplus, conv, Jc_all, active

    # ------------------------------------------------------------------
    def _regression(self, theta: np.ndarray, frames: np.ndarray, lam, conv, Jc_all, active):
        """Weighted LS for pi (linear in dynamics) with log-Cholesky
        linearization about theta. Returns theta_new, J_measm, resid stats."""
        dyn, wd = self.dyn, self.wd
        dyn.set_theta(theta)
        nG = dyn.n_groups
        # accumulate normal equations in pi-space (nG*10)
        H = np.zeros((10 * nG, 10 * nG))
        g = np.zeros(10 * nG)
        Jmeas = 0.0
        resid2_joint = 0.0
        n_rows = 0
        # per-group body-column mapping into the full 130-dim regressor p
        # regressor p: bodies 1..13 stacked. group gi has bodies list.
        group_cols = []  # (slice in p, group index, body indices)
        for gi, grp in enumerate(dyn.groups):
            cols = []
            for b in grp:
                # body b (joint idx) -> p block (b-1)*10
                cols.append(np.arange((b - 1) * 10, b * 10))
            group_cols.append(np.hstack(cols))
        # pi-space ordering inside group: use pinocchio order [m, mc, Ixx,Ixy,Iyy,Ixz,Iyz,Izz]
        for i, k in enumerate(frames):
            if not conv[i]:
                continue
            a = (wd.v[k + 1] - wd.v[k]) / wd.dt   # measured accel (v+ measured)
            Y = dyn.regressor(wd.q[k], wd.v[k], a)  # nv x 130
            Bu = np.zeros(dyn.nv)
            Bu[dyn.v_joint] = wd.u[k]
            tau_contact = Jc_all[i].T @ lam[i].reshape(-1) / wd.dt  # nv
            r_lin = Y @ self._current_p(dyn) - Bu - tau_contact  # at current theta
            # expand Y to group space: Y_g (nv x 10*nG)
            Yg = np.zeros((dyn.nv, 10 * nG))
            for gi in range(nG):
                # group regressor = sum over bodies of Y_b @ B_b, where B_b
                # maps group-head-frame pi to this body's frame (left/right
                # leg frames are rotated ~180 deg apart in the URDF)
                Yg_blk = np.zeros((dyn.nv, 10))
                grp = dyn.groups[gi]
                for bi, b in enumerate(grp):
                    cols = np.arange((b - 1) * 10, b * 10)
                    Yg_blk += Y[:, cols] @ dyn.group_B[gi][bi]
                Yg[:, 10 * gi : 10 * gi + 10] = Yg_blk
            # weights: joint rows strong, base rows weaker (impulse N s vs Nm)
            w = np.full(dyn.nv, self.w_base)
            w[dyn.v_joint] = self.w_joint
            # mask base linear rows (unobserved a) -> zero weight
            w[:3] = 0.0
            sw = np.sqrt(w)
            H += (Yg * sw[:, None]).T @ (Yg * sw[:, None])
            g += (Yg * sw[:, None]).T @ (sw * (-r_lin))
            resid2_joint += float(np.sum((r_lin[dyn.v_joint]) ** 2 * self.w_joint))
            n_rows += 1
        # prior in theta-space -> linearize: dtheta = J_lc^{-T} ... approximate
        # by adding prior on pi-groups (scaled by curvature of log-Cholesky)
        # simpler: add prior directly as Tikhonov on delta-pi
                # current pi-groups
        pi_g = dyn.theta_to_pi_groups(theta)  # nG x 10
        g += self.prior_w * 1e-1 * max(np.trace(H) / (10 * nG), 1e-9) / self.prior_w * 0  # (prior center via r_lin below)
        # prior center: Tikhonov pulling pi toward nominal pi_hat with
        # weight ~ prior_rel * (mean data curvature), so the prior matters
        # only in weakly-excited directions
        pi_hat_g = dyn.theta_to_pi_groups(self.theta_hat)
        lam_prior = self.prior_w * 1e-3 * max(np.trace(H) / (10 * nG), 1e-9)
        H += lam_prior * np.eye(10 * nG)
        g += lam_prior * (-(pi_g - pi_hat_g)).reshape(-1)
        if not np.all(np.isfinite(H)) or not np.all(np.isfinite(g)):
            raise RuntimeError(f"nonfinite H/g at regression: trace={np.trace(H)}")
        dp = np.linalg.lstsq(H, -g, rcond=1e-8)[0]
        if not np.all(np.isfinite(dp)):
            raise RuntimeError("nonfinite dp")
        pi_scale = np.maximum(np.abs(pi_g.reshape(-1)), 1e-3)
        dp = np.clip(dp, -0.3 * pi_scale, 0.3 * pi_scale)
        pi_new = (pi_g.reshape(-1) + dp).reshape(nG, 10)
        pi_new[:, 0] = np.maximum(pi_new[:, 0], 0.05)
        # convert to theta (log-Cholesky) from pi (pinocchio order)
        from .log_cholesky import pi_to_theta as p2t
        theta_new = np.concatenate([p2t(pi_new[gi][_PC_TO_LC]) for gi in range(nG)])
        return theta_new, Jmeas, resid2_joint / max(n_rows * dyn.nj, 1), n_rows

    def _current_p(self, dyn) -> np.ndarray:
        return np.concatenate(
            [dyn.model.inertias[j].toDynamicParameters() for j in range(1, dyn.model.njoints)]
        )

    # ------------------------------------------------------------------
    def solve(self, verbose: bool = True) -> PFIEResult:
        t0 = time.time()
        theta = self.theta_hat.copy()
        res = PFIEResult(theta=theta)
        for it in range(self.cfg.n_iters):
            lam, vplus, conv, Jc_all, active = self._sweep(theta, self.idx_tr)
            theta_new, _, rmse_tr, n_used = self._regression(
                theta, self.idx_tr, lam, conv, Jc_all, active
            )
            dth = np.linalg.norm(theta_new - theta) / (1 + np.linalg.norm(theta))
            theta = theta_new
            res.history.append(rmse_tr)
            if verbose:
                print(
                    f"  iter {it:2d}: frames={n_used}  joint-resid-RMSE={rmse_tr:.3f} Nm"
                    f"  dtheta={dth:.2e}  mass={self.dyn.total_mass():.3f} kg"
                )
            if dth < 1e-4:
                res.converged = True
                break
        res.iters_used = it + 1
        res.theta = theta
        # evaluate train/holdout torque residual with final theta
        for tag, idxs in (("train", self.idx_tr), ("holdout", self.idx_ho)):
            lam, vplus, conv, Jc_all, active = self._sweep(theta, idxs)
            self.dyn.set_theta(theta)
            r2 = 0.0
            Fz = []
            n_used = 0
            for i, k in enumerate(idxs):
                if not conv[i]:
                    continue
                a = (self.wd.v[k + 1] - self.wd.v[k]) / self.wd.dt
                Y = self.dyn.regressor(self.wd.q[k], self.wd.v[k], a)
                Bu = np.zeros(self.dyn.nv)
                Bu[self.dyn.v_joint] = self.wd.u[k]
                tau_c = Jc_all[i].T @ lam[i].reshape(-1) / self.wd.dt
                r = Y @ self._current_p(self.dyn) - Bu - tau_c
                r2 += float(np.sum(r[self.dyn.v_joint] ** 2))
                Fz.append(lam[i][:, 2].sum() / self.wd.dt)
                n_used += 1
            rmse = np.sqrt(r2 / max(n_used * self.dyn.nj, 1))
            if tag == "train":
                res.rmse_joint_train = rmse
                res.J_train = r2
                res.Fz_mean = float(np.mean(Fz)) if Fz else 0.0
            else:
                res.rmse_joint_holdout = rmse
                res.J_holdout = r2
        res.elapsed_s = time.time() - t0
        return res
