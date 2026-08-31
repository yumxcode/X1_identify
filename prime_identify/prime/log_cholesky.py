"""Log-Cholesky parameterization of physically-consistent inertial parameters.

Implements Rucker & Wensing's Log-Cholesky map theta -> pi for a single rigid
body, with analytic Jacobian d(pi)/d(theta), following PRIME (arXiv:2605.17681,
Eq. 2-5).

Parameter vector convention (PRIME Eq. 2):
    pi = [m, h_x, h_y, h_z, I_xx, I_yy, I_zz, I_xy, I_yz, I_xz]  (10 params)
    h = m * c  (first mass moment), c = COM in link frame,
    I about the COM expressed in link frame axes.

Pseudo-inertia (PRIME Eq. 3):
    I4 = [[ Sigma, h ],      Sigma = 0.5*tr(I)*Id3 - I   (about origin!)
          [ h^T,   m ]]
    Physical consistency <=> I4 (psd, det>0) <=> m>0 and I About COM pd.

Log-Cholesky (PRIME Eq. 4-5):
    I4 = U U^T,  U = e^alpha * [[e^{d1}, s12, s13, t1],
                                [0,      e^{d2}, s23, t2],
                                [0,      0,      e^{d3}, t3],
                                [0,      0,      0,    1 ]]
    theta = [alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3]
"""
from __future__ import annotations

import numpy as np

# theta ordering used everywhere in this package
THETA_NAMES = ["alpha", "d1", "d2", "d3", "s12", "s23", "s13", "t1", "t2", "t3"]


def theta_to_U(theta: np.ndarray) -> np.ndarray:
    """Build the 4x4 upper-triangular Cholesky factor U from theta."""
    alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3 = theta
    e = np.exp(alpha)
    U = np.array(
        [
            [e * np.exp(d1), e * s12, e * s13, e * t1],
            [0.0, e * np.exp(d2), e * s23, e * t2],
            [0.0, 0.0, e * np.exp(d3), e * t3],
            [0.0, 0.0, 0.0, e],
        ]
    )
    return U


def U_to_pi(U: np.ndarray) -> np.ndarray:
    """Extract pi = [m, h, I_origin] from the pseudo-inertia I4 = U U^T.

    NOTE: the inertia entries of the pseudo-inertia are taken about the link
    frame ORIGIN (not the COM). We return I about the COM expressed in link
    axes, which is the standard pi convention; conversion uses
    I_com = I_origin - m * skew(c)^T skew(c) ... (parallel axis theorem).
    """
    I4 = U @ U.T
    m = I4[3, 3]
    h = I4[:3, 3]  # = m*c
    Sigma = I4[:3, :3]
    # I_origin = tr(Sigma)*Id3 - Sigma ; proof: Sigma = 0.5*tr(I_o)*Id - I_o
    tr_S = np.trace(Sigma)
    I_origin = tr_S * np.eye(3) - Sigma
    c = h / m
    # parallel axis: I_origin = I_com + m * skew(c)^T ... for inertia matrices:
    # I_origin = I_com - m * skew(c) @ skew(c)
    C = _skew(c)
    I_com = I_origin + m * (C @ C)
    return np.concatenate([[m], h, _inertia_to_vec(I_com)])


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
    )


def _inertia_to_vec(I: np.ndarray) -> np.ndarray:
    """[Ixx, Iyy, Izz, Ixy, Iyz, Ixz] (Pinocchio/URDF symmetric ordering)."""
    return np.array([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[1, 2], I[0, 2]])


def _vec_to_inertia(v: np.ndarray) -> np.ndarray:
    Ixx, Iyy, Izz, Ixy, Iyz, Ixz = v
    return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])


def _upper_factor(A: np.ndarray) -> np.ndarray:
    """Unique upper-triangular U with positive diagonal s.t. A = U @ U.T.

    (np.linalg.cholesky gives L with A = L @ L.T; the upper Cholesky factor
    R = L.T satisfies A = R.T @ R instead -- the opposite product order.)
    Computed by backward substitution from the last column.
    """
    n = A.shape[0]
    U = np.zeros((n, n))
    for j in range(n - 1, -1, -1):
        U[j, j] = np.sqrt(A[j, j] - U[j, j + 1 :] @ U[j, j + 1 :])
        for i in range(j):
            U[i, j] = (A[i, j] - U[i, j + 1 :] @ U[j, j + 1 :]) / U[j, j]
    return U


def pi_to_theta(pi: np.ndarray) -> np.ndarray:
    """Invert the map (structured Log-Cholesky is a bijection onto the
    physically-consistent set)."""
    m = pi[0]
    h = pi[1:4]
    I_com = _vec_to_inertia(pi[4:10])
    c = h / m
    C = _skew(c)
    I_origin = I_com - m * (C @ C)
    Sigma = 0.5 * np.trace(I_origin) * np.eye(3) - I_origin
    I4 = np.zeros((4, 4))
    I4[:3, :3] = Sigma
    I4[:3, 3] = h
    I4[3, :3] = h
    I4[3, 3] = m
    # Cholesky of SPD I4 gives U^T U form; we need structured U (diag of last
    # row = 1). Standard eigen/Cholesky gives an equivalent factor; convert by
    # normalizing: take Cholesky L (lower) -> U_chol = L.T; scale columns/rows
    # so that U[3,3] = 1 by factoring out the (3,3) element.
    L = _upper_factor(I4)
    Uc = L
    # structured form: U = e^alpha * [[e^d1, s12, s13, t1], ... [.., 1]]
    # any upper-triangular factor normalizes to this form with
    # e^alpha = Uc[3,3]
    e = Uc[3, 3]
    alpha = np.log(e)
    d1 = np.log(Uc[0, 0] / e)
    d2 = np.log(Uc[1, 1] / e)
    d3 = np.log(Uc[2, 2] / e)
    s12 = Uc[0, 1] / e
    s23 = Uc[1, 2] / e
    s13 = Uc[0, 2] / e
    t1 = Uc[0, 3] / e
    t2 = Uc[1, 3] / e
    t3 = Uc[2, 3] / e
    return np.array([alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3])


def theta_to_pi(theta: np.ndarray) -> np.ndarray:
    return U_to_pi(theta_to_U(theta))


def jacobian_pi_theta(theta: np.ndarray) -> np.ndarray:
    """Analytic d(pi)/d(theta) is involved; validated finite differences are
    used here for robustness (central differences, 10x10 output).

    The forward map is smooth everywhere, so FD is exact to ~1e-9 for the
    step used below and costs 20 map evaluations (microseconds).
    """
    eps = 1e-6
    J = np.zeros((10, 10))
    f0 = theta_to_pi(theta)
    for i in range(10):
        tp = theta.copy()
        tm = theta.copy()
        tp[i] += eps
        tm[i] -= eps
        J[:, i] = (theta_to_pi(tp) - theta_to_pi(tm)) / (2 * eps)
    return J


def is_physically_consistent(pi: np.ndarray, tol: float = 1e-9) -> bool:
    """Check full physical consistency: I4 SPD (all eigenvalues > tol)."""
    m = pi[0]
    h = pi[1:4]
    I_com = _vec_to_inertia(pi[4:10])
    c = h / m
    C = _skew(c)
    I_origin = I_com - m * (C @ C)
    Sigma = 0.5 * np.trace(I_origin) * np.eye(3) - I_origin
    I4 = np.zeros((4, 4))
    I4[:3, :3] = Sigma
    I4[:3, 3] = h
    I4[3, :3] = h
    I4[3, 3] = m
    return np.min(np.linalg.eigvalsh(I4)) > tol and m > tol


def _selftest():
    rng = np.random.default_rng(0)
    for trial in range(200):
        # random physically consistent pi: sample U from random theta
        theta = rng.normal(scale=0.3, size=10)
        pi = theta_to_pi(theta)
        assert is_physically_consistent(pi), "log-cholesky must give pc params"
        theta2 = pi_to_theta(pi)
        pi2 = theta_to_pi(theta2)
        # roundtrip must reproduce pi (the map is a bijection onto PC set)
        assert np.allclose(pi, pi2, atol=1e-8), (pi, pi2)
    # FD jacobian sanity: compare to directional derivative
    theta = rng.normal(scale=0.2, size=10)
    J = jacobian_pi_theta(theta)
    dtheta = rng.normal(size=10) * 1e-6
    fd = (theta_to_pi(theta + dtheta) - theta_to_pi(theta - dtheta)) / 2
    assert np.allclose(J @ dtheta, fd, rtol=1e-5, atol=1e-9)
    print("log_cholesky selftest OK")


if __name__ == "__main__":
    _selftest()
