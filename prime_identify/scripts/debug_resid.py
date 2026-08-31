import numpy as np
import sys
sys.path.insert(0, "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify")
from prime.dynamics import X1Dynamics
from prime.data import load_walk_diag

dyn = X1Dynamics("/Users/yumx/code/robot_x/X1/X1_辨识/X1_train/resources/robots/x1/urdf/x1.urdf")
wd = load_walk_diag("/Users/yumx/code/robot_x/X1/X1_辨识/x1_data/walk_diag_20260824_103222.csv", dyn)
k = 100
dyn.set_theta(dyn.theta_hat)
vplus, lam, conv = dyn.solve_contact_step(wd.q[k], wd.v[k], wd.u[k], wd.dt)
a = (wd.v[k + 1] - wd.v[k]) / wd.dt
Y = dyn.regressor(wd.q[k], wd.v[k], a)
p = np.concatenate([dyn.model.inertias[j].toDynamicParameters() for j in range(1, dyn.model.njoints)])
Yp = Y @ p
M, hh = dyn.M_and_h(wd.q[k], wd.v[k])
print("Yp == M a + h :", np.allclose(Yp, M @ a + hh, atol=1e-8))
Bu = np.zeros(dyn.nv)
Bu[6:] = wd.u[k]
Jc, phi = dyn.kinematics(wd.q[k])
tc = Jc.T @ lam.reshape(-1) / wd.dt
r = Yp - Bu - tc
print("conv:", conv)
print("lam in Newtons:")
print(np.round(lam / wd.dt, 1))
print("Yp joints[:6]:", np.round(Yp[6:12], 2))
print("Bu joints[:6]:", np.round(Bu[6:12], 2))
print("tc joints[:6]:", np.round(tc[6:12], 2))
print("r joints:", np.round(r[6:], 1))
print("abs r max joint: %.2f" % np.abs(r[6:]).max())
a_sim = (vplus - wd.v[k]) / wd.dt
Y_sim = dyn.regressor(wd.q[k], wd.v[k], a_sim)
r_sim = Y_sim @ p - Bu - tc
print("self-consistent residual (a from QP vplus) max joint: %.4f" % np.abs(r_sim[6:]).max())
print("a_meas joints[:6]:", np.round(a[6:12], 1))
print("a_sim  joints[:6]:", np.round(a_sim[6:12], 1))
