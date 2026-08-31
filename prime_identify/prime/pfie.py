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
        self.row_sigma = np.full(dyn.nv, 3.0)
        self.row_sigma[dyn.v_joint] = 0.35
        self._row_acc = np.zeros(dyn.nv)

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
        """Contact-free subspace regression (external-force-invariant).

        Per frame, project the torque dynamics onto the left null space of
        the contact Jacobian transpose, which ELIMINATES the unknown contact
        forces exactly (classic identification with unmeasured contacts,
        cf. PRIME refs [25,26]; mathematically strict, no coupling to the
        QP-solved lambda):

            Z_k^T [ Y(q, v, a_meas) p - B u ] = 0,   Z_k = null(J_k) (18 x r)

        Single-step weighted LS over all frames + prior; iterated with the
        IRLS sweep only to refresh a_meas smoothing (the equations are
        linear in p, so convergence is fast).
        """
        dyn, wd = self.dyn, self.wd
        dyn.set_theta(theta)
        nG = dyn.n_groups
        H = np.zeros((10 * nG, 10 * nG))
        g = np.zeros(10 * nG)
        n_rows = 0
        resid2 = 0.0
        row_mask = np.ones(dyn.nv, dtype=bool)
        row_mask[:3] = False
        for jn in ("left_ankle_pitch", "left_ankle_roll",
                   "right_ankle_pitch", "right_ankle_roll"):
            row_mask[dyn.joint_index(jn)] = False
        group_cols = []
        for gi, grp in enumerate(dyn.groups):
            cols = [np.arange((b - 1) * 10, b * 10) for b in grp]
            group_cols.append(np.hstack(cols))
        p_cur = self._current_p(dyn)
        w_row = 1.0 / 0.35**2
        w_base_row = 1.0 / 3.0**2
        for i, k in enumerate(frames):
            if not conv[i]:
                continue
            a = (wd.v_out[k] - wd.v[k]) / wd.dt
            Y = dyn.regressor(wd.q[k], wd.v[k], a)
            Bu = np.zeros(dyn.nv)
            Bu[dyn.v_joint] = wd.u[k]
            r_full = Y @ p_cur - Bu
            J = Jc_all[i]
            # mask ankle rows BEFORE null space (they are unmodeled)
            Jm = J.copy()
            keep = row_mask.copy()
            Jm[:, ~keep] = 0.0
            # Z: null space of Jm rows -> project residual to remove contacts
            U_, S_, Vt_ = np.linalg.svd(Jm, full_matrices=True)
            rank = int(np.sum(S_ > 1e-9 * max(S_.max(), 1e-12)))
            Z = Vt_[rank:].T            # (nv x (nv-rank))
            if Z.shape[1] == 0:
                continue
            Zk = Z * row_mask[:, None]  # keep only trusted rows' contribution
            Yz = Zk.T @ Y               # (nv-rank) x 130
            rz = Zk.T @ r_full          # (nv-rank)
            Yg = np.zeros((Zz_dim := Yz.shape[0], 10 * nG))
            for gi in range(nG):
                Yg[:, 10 * gi : 10 * gi + 10] = Yz[:, group_cols[gi]].reshape(
                    Yz.shape[0], -1, 10
                ).mean(axis=1)
            Aw = Yg * np.sqrt(w_row)
            H += Aw.T @ Aw
            g += Aw.T @ (np.sqrt(w_row) * (-rz))
            resid2 += float(rz @ rz)
            n_rows += 1
        if n_rows == 0:
            return theta, 0.0, np.inf, 0
        pi_g = dyn.theta_to_pi_groups(theta)
        pi_hat_g = dyn.theta_to_pi_groups(self.theta_hat)
        lam_prior = self.prior_w * 1e-3 * max(np.trace(H) / (10 * nG), 1e-12)
        H += lam_prior * np.eye(10 * nG)
        g += lam_prior * (-(pi_g - pi_hat_g)).reshape(-1)
        dp = np.linalg.lstsq(H, g, rcond=1e-4)[0]
        if not np.all(np.isfinite(dp)):
            return theta, 0.0, np.inf, n_rows
        pi_scale = np.maximum(np.abs(pi_g.reshape(-1)), 1e-3)
        dp = np.clip(dp, -0.25 * pi_scale, 0.25 * pi_scale)
        pi_new = (pi_g.reshape(-1) + dp).reshape(nG, 10)
        pi_new[:, 0] = np.maximum(pi_new[:, 0], 0.05)
        from .log_cholesky import pi_to_theta as p2t
        from .dynamics import _PC_TO_LC
        theta_new = theta.copy()
        for gi in range(nG):
            if not np.all(np.isfinite(pi_new[gi])):
                continue
            proj = X1Dynamics._project_to_pc(pi_new[gi][None, :])[0]
            try:
                theta_new[10 * gi : 10 * gi + 10] = p2t(proj[_PC_TO_LC])
            except Exception:
                pass
        resid_rms = float(np.sqrt(resid2 / n_rows))
        return theta_new, 0.0, resid_rms, n_rows

    def _accept_cost(self, theta: np.ndarray, frames: np.ndarray) -> float:
        """Acceptance metric (G2.2): full-row contact-consistent torque
        residual with lam re-solved at theta."""
        dyn, wd = self.dyn, self.wd
        dyn.set_theta(theta)
        m = np.ones(dyn.nv, dtype=bool)
        m[:3] = False
        for jn in ("left_ankle_pitch", "left_ankle_roll",
                   "right_ankle_pitch", "right_ankle_roll"):
            m[dyn.joint_index(jn)] = False
        tot, n = 0.0, 0
        for k in frames:
            vp, imp, conv = dyn.solve_contact_step(wd.q[k], wd.v[k], wd.u[k], wd.dt)
            if not conv or not np.all(np.isfinite(imp)):
                continue
            a = (wd.v_out[k] - wd.v[k]) / wd.dt
            Y = dyn.regressor(wd.q[k], wd.v[k], a)
            Bu = np.zeros(dyn.nv)
            Bu[dyn.v_joint] = wd.u[k]
            Jc, _ = dyn.kinematics(wd.q[k])
            r = Y @ self._current_p(dyn) - Bu - Jc.T @ imp.reshape(-1) / wd.dt
            tot += float(np.sum(r[m] ** 2))
            n += 1
        return tot / max(n, 1)

    def _current_p(self, dyn) -> np.ndarray:
        return np.concatenate(
            [dyn.model.inertias[j].toDynamicParameters() for j in range(1, dyn.model.njoints)]
        )

    # ------------------------------------------------------------------
    def solve(self, verbose: bool = True) -> PFIEResult:
        t0 = time.time()
        theta = self.theta_hat.copy()
        res = PFIEResult(theta=theta)
        # acceptance metric: full-row contact-consistent residual on a
        # subsample (same definition as run_real_walk.fie_cost)
        rng = np.random.default_rng(1)
        ls_idx = self.idx_tr[rng.choice(len(self.idx_tr), min(150, len(self.idx_tr)), replace=False)]

        def accept_cost(th):
            return self._accept_cost(th, ls_idx)

        c_cur = accept_cost(theta)
        res.history.append(c_cur)
        for it in range(self.cfg.n_iters):
            lam, vplus, conv, Jc_all, active = self._sweep(theta, self.idx_tr)
            theta_dir, _, rmse_tr, n_used = self._regression(
                theta, self.idx_tr, lam, conv, Jc_all, active
            )
            # line search on the acceptance metric
            best_alpha, best_cost = 0.0, c_cur
            for alpha in (1.0, 0.5, 0.25, 0.1):
                th_try = theta + alpha * (theta_dir - theta)
                c_try = accept_cost(th_try)
                if np.isfinite(c_try) and c_try < best_cost:
                    best_alpha, best_cost = alpha, c_try
                    break
            theta_new = theta + best_alpha * (theta_dir - theta)
            dth = np.linalg.norm(theta_new - theta) / (1 + np.linalg.norm(theta))
            improved = best_cost < c_cur
            c_cur = min(c_cur, best_cost)
            theta = theta_new
            res.history.append(best_cost)
            if verbose:
                print(
                    f"  iter {it:2d}: frames={n_used}  proj-rmse={rmse_tr:.2f}"
                    f"  alpha={best_alpha:.2f}  accept-cost={best_cost:.1f}"
                    f"  ({'improved' if improved else 'stalled'})  mass={self.dyn.total_mass():.3f} kg"
                )
            if not improved and it >= 2:
                res.converged = True
                break
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
                a = (self.wd.v_out[k] - self.wd.v[k]) / self.wd.dt
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
