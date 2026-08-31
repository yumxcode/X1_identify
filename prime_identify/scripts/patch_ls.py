"""Patch: line-search-aligned IRLS in PFIE.solve.

The regression step direction dp is accepted only at the step size that
minimizes the ACCEPTANCE metric (contact-consistent full-row residual with
lam re-solved at the candidate theta), enforcing G2.1 (objective-metric
alignment) and G2.4 (no diverging chains).
"""
import ast

path = "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify/prime/pfie.py"
src = open(path).read()

old = '''        res = PFIEResult(theta=theta)
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
                break'''

new = '''        res = PFIEResult(theta=theta)
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
                break'''

assert old in src, "solve loop not found"
src = src.replace(old, new)

# add _accept_cost method before _current_p
anchor = "    def _current_p(self, dyn) -> np.ndarray:"
accept_method = '''    def _accept_cost(self, theta: np.ndarray, frames: np.ndarray) -> float:
        """Acceptance metric (G2.2): full-row contact-consistent torque
        residual with lam re-solved at theta. Matches
        run_real_walk.fie_cost."""
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
            a = (wd.v[k + 1] - wd.v[k]) / wd.dt
            Y = dyn.regressor(wd.q[k], wd.v[k], a)
            Bu = np.zeros(dyn.nv)
            Bu[dyn.v_joint] = wd.u[k]
            Jc, _ = dyn.kinematics(wd.q[k])
            r = Y @ self._current_p(dyn) - Bu - Jc.T @ imp.reshape(-1) / wd.dt
            tot += float(np.sum(r[m] ** 2))
            n += 1
        return tot / max(n, 1)

''' + anchor
assert anchor in src
src = src.replace(anchor, accept_method)
open(path, "w").write(src)
ast.parse(src)
print("patched OK")
