"""Patch: full-row torque-residual Gauss-Newton regression (replaces
null-space projection). With the v_out semantics fix the noise floor is
~2 Nm and the parameter signal (86% of cost) dominates, so the plain
contact-consistent residual regression works; lambda is re-solved each
sweep (IRLS) and the step is line-searched on the acceptance metric."""
import ast

path = "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify/prime/pfie.py"
src = open(path).read()
start = src.index("    def _regression(")
end = src.index("    def _accept_cost(")
new_reg = '''    def _regression(self, theta: np.ndarray, frames: np.ndarray, lam, conv, Jc_all, active):
        """Gauss-Newton on the contact-consistent torque residual (the
        acceptance metric's own model):

            r_k = Y(q, v, a_k) p - B u - J^T lam_k/dt,   a_k=(v_out[k]-v[k])/dt

        linear in p at fixed lam (re-solved each sweep); trusted rows only.
        Returns the proposed theta (acceptance line search happens outside).
        """
        dyn, wd = self.dyn, self.wd
        dyn.set_theta(theta)
        nG = dyn.n_groups
        H = np.zeros((10 * nG, 10 * nG))
        g = np.zeros(10 * nG)
        n_rows = 0
        m = np.ones(dyn.nv, dtype=bool)
        m[:3] = False
        for jn in ("left_ankle_pitch", "left_ankle_roll",
                   "right_ankle_pitch", "right_ankle_roll"):
            m[dyn.joint_index(jn)] = False
        sw = (m / 0.35).astype(float)  # sqrt weight rows
        group_cols = []
        for gi, grp in enumerate(dyn.groups):
            cols = [np.arange((b - 1) * 10, b * 10) for b in grp]
            group_cols.append(np.hstack(cols))
        p_cur = self._current_p(dyn)
        resid2 = 0.0
        for i, k in enumerate(frames):
            if not conv[i]:
                continue
            a = (wd.v_out[k] - wd.v[k]) / wd.dt
            Y = dyn.regressor(wd.q[k], wd.v[k], a)
            Bu = np.zeros(dyn.nv)
            Bu[dyn.v_joint] = wd.u[k]
            tau_c = Jc_all[i].T @ lam[i].reshape(-1) / wd.dt
            r = Y @ p_cur - Bu - tau_c
            Yg = np.zeros((dyn.nv, 10 * nG))
            for gi in range(nG):
                Yg[:, 10 * gi : 10 * gi + 10] = Y[:, group_cols[gi]].reshape(
                    dyn.nv, -1, 10
                ).mean(axis=1)
            Aw = Yg * sw[:, None]
            H += Aw.T @ Aw
            g += Aw.T @ (sw * (-r))
            resid2 += float(np.sum(r[m] ** 2))
            n_rows += 1
        if n_rows == 0:
            return theta, 0.0, np.inf, 0
        pi_g = dyn.theta_to_pi_groups(theta)
        pi_hat_g = dyn.theta_to_pi_groups(self.theta_hat)
        lam_prior = self.prior_w * 1e-4 * max(np.trace(H) / (10 * nG), 1e-12)
        H += lam_prior * np.eye(10 * nG)
        g += lam_prior * (-(pi_g - pi_hat_g)).reshape(-1)
        dp = np.linalg.lstsq(H, g, rcond=1e-6)[0]
        if not np.all(np.isfinite(dp)):
            return theta, 0.0, np.inf, n_rows
        pi_scale = np.maximum(np.abs(pi_g.reshape(-1)), 1e-3)
        dp = np.clip(dp, -0.5 * pi_scale, 0.5 * pi_scale)
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
        resid_rms = float(np.sqrt(resid2 / n_rows / max(int(m.sum()), 1)))
        return theta_new, 0.0, resid_rms, n_rows

'''
src = src[:start] + new_reg + src[end:]
open(path, "w").write(src)
ast.parse(src)
print("patched OK")
