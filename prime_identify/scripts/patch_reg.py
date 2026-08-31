"""Rewrite of PFIE._regression (torque-residual channel) via patch script."""
import re

path = "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify/prime/pfie.py"
src = open(path).read()
start = src.index("    def _regression(")
end = src.index("    def _current_p(")
new_reg = '''    def _regression(self, theta: np.ndarray, frames: np.ndarray, lam, conv, Jc_all, active):
        """Gauss-Newton step on the TORQUE-consistency residual (the paper's
        joint-space measurement channel):

            r_k = Y(q, v, a_meas) p  -  B u  -  J^T lam_k / dt

        with lam_k from the contact QP at the CURRENT theta (latent forces
        inferred from dynamics, as in PRIME). The residual is exactly linear
        in p (pinocchio regressor), so each IRLS iteration is a weighted LS.
        Rows: swing-leg joints carry pure parameter signal (no contact
        force); stance rows and base rows are down-weighted adaptively.
        """
        dyn, wd = self.dyn, self.wd
        dyn.set_theta(theta)
        nG = dyn.n_groups
        self._row_acc = np.zeros(dyn.nv)
        H = np.zeros((10 * nG, 10 * nG))
        g = np.zeros(10 * nG)
        n_rows = 0
        m = np.ones(dyn.nv, dtype=bool)
        m[:3] = False            # base linear velocity unobserved (placeholder)
        for jn in ("left_ankle_pitch", "left_ankle_roll",
                   "right_ankle_pitch", "right_ankle_roll"):
            m[dyn.joint_index(jn)] = False   # free-hinge, lam inconsistent
        sig = np.maximum(self.row_sigma, 1e-3)
        sw = (m / sig).astype(float)
        group_cols = []
        for gi, grp in enumerate(dyn.groups):
            cols = [np.arange((b - 1) * 10, b * 10) for b in grp]
            group_cols.append(np.hstack(cols))
        p_cur = self._current_p(dyn)
        for i, k in enumerate(frames):
            if not conv[i]:
                continue
            a = (wd.v[k + 1] - wd.v[k]) / wd.dt
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
            g += Aw.T @ (sw * (-r))     # Newton: dp = H^-1 g solves min ||A dp + r||
            self._row_acc += r**2
            n_rows += 1
        if n_rows == 0:
            return theta, 0.0, np.inf, 0
        rms = np.sqrt(self._row_acc / n_rows)
        floor = np.full(dyn.nv, 0.35)
        floor[:3] = 5.0
        self.row_sigma = np.maximum(rms, floor)
        pi_g = dyn.theta_to_pi_groups(theta)
        pi_hat_g = dyn.theta_to_pi_groups(self.theta_hat)
        lam_prior = self.prior_w * 1e-2 * max(np.trace(H) / (10 * nG), 1e-12)
        H += lam_prior * np.eye(10 * nG)
        g += lam_prior * (-(pi_g - pi_hat_g)).reshape(-1)
        dp = np.linalg.lstsq(H, g, rcond=1e-4)[0]
        if not np.all(np.isfinite(dp)):
            return theta, 0.0, np.inf, n_rows
        pi_scale = np.maximum(np.abs(pi_g.reshape(-1)), 1e-3)
        dp = np.clip(dp, -0.2 * pi_scale, 0.2 * pi_scale)
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
        resid_rms = float(np.sqrt(np.mean(self._row_acc[m] / n_rows)))
        return theta_new, 0.0, resid_rms, n_rows

'''
src = src[:start] + new_reg + src[end:]
open(path, "w").write(src)
import ast
ast.parse(src)
print("patched OK")
