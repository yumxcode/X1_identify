"""Patch: null-space (contact-free subspace) regression for PFIE."""
import ast

path = "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify/prime/pfie.py"
src = open(path).read()
start = src.index("    def _regression(")
end = src.index("    def _current_p(")
new_reg = '''    def _regression(self, theta: np.ndarray, frames: np.ndarray, lam, conv, Jc_all, active):
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
            a = (wd.v[k + 1] - wd.v[k]) / wd.dt
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

'''
src = src[:start] + new_reg + src[end:]
open(path, "w").write(src)
ast.parse(src)
print("patched OK")
