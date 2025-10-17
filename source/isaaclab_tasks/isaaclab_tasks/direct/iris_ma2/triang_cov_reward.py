
# triang_cov_reward.py
from dataclasses import dataclass
import numpy as np

def skew(v):
    v = np.asarray(v).reshape(3,)
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]], dtype=float)

@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

@dataclass
class View:
    R_wc: np.ndarray
    t_wc: np.ndarray
    Sigma_pix: np.ndarray
    Sigma_twb: np.ndarray = None
    Sigma_phiwb: np.ndarray = None
    Sigma_alpha: np.ndarray = None
    Sigma_beta: np.ndarray = None
    Sigma_K: np.ndarray = None
    e_alpha: np.ndarray = None
    e_beta: np.ndarray = None
    K: Intrinsics = None
    include_pose: bool = True
    include_gimbal: bool = True
    include_intrinsics: bool = False

def proj_jacobian_wrt_Xc(Xc, K):
    Xc = np.asarray(Xc).reshape(3,)
    fx, fy = K.fx, K.fy
    X, Y, Z = Xc
    Z = float(Z)
    if abs(Z) < 1e-12:
        Z = 1e-12
    Z2 = Z * Z
    Ju = np.array([[fx / Z, 0.0, -fx * X / Z2],
                   [0.0, fy / Z, -fy * Y / Z2]], dtype=float)
    return Ju

def make_block_diag(blocks):
    if (blocks is None) or (len(blocks) == 0):
        return None
    sizes = [b.shape[0] for b in blocks]
    n = int(sum(sizes))
    M = np.zeros((n, n), dtype=float)
    i = 0
    for b in blocks:
        k = b.shape[0]
        M[i:i+k, i:i+k] = b
        i += k
    return M

def block_stack_rows(mats):
    rows = [m for m in mats if m is not None]
    if len(rows) == 0:
        return None
    return np.vstack(rows)

def block_diag(mats):
    mats = [m for m in mats if m is not None]
    if len(mats) == 0:
        return None
    sizes = [m.shape[0] for m in mats]
    n = int(sum(sizes))
    M = np.zeros((n, n), dtype=float)
    i = 0
    for m in mats:
        k = m.shape[0]
        M[i:i+k, i:i+k] = m
        i += k
    return M

def jacs_for_view(X_w, view):
    R = np.asarray(view.R_wc).reshape(3,3)
    t = np.asarray(view.t_wc).reshape(3,)
    K = view.K
    Xc = R.T @ (np.asarray(X_w).reshape(3,) - t)
    Ju_Xc = proj_jacobian_wrt_Xc(Xc, K)  # 2-by-3
    JX = Ju_Xc @ R.T  # 2-by-3

    Jblocks = []
    Sig_blocks = []

    if view.include_pose:
        if view.Sigma_twb is not None:
            J_t = -JX
            Jblocks.append(J_t)
            Sig_blocks.append(np.asarray(view.Sigma_twb).reshape(3,3))
        if view.Sigma_phiwb is not None:
            J_phi = Ju_Xc @ (-skew(Xc))
            Jblocks.append(J_phi)
            Sig_blocks.append(np.asarray(view.Sigma_phiwb).reshape(3,3))

    if view.include_gimbal:
        if (view.Sigma_alpha is not None) and (view.e_alpha is not None):
            e_a = np.asarray(view.e_alpha).reshape(3,)
            J_alpha = Ju_Xc @ (-skew(Xc)) @ e_a.reshape(3,1)  # 2-by-1
            Jblocks.append(J_alpha)
            Sig_blocks.append(np.asarray(view.Sigma_alpha).reshape(1,1))
        if (view.Sigma_beta is not None) and (view.e_beta is not None):
            e_b = np.asarray(view.e_beta).reshape(3,)
            J_beta = Ju_Xc @ (-skew(Xc)) @ e_b.reshape(3,1)   # 2-by-1
            Jblocks.append(J_beta)
            Sig_blocks.append(np.asarray(view.Sigma_beta).reshape(1,1))

    if view.include_intrinsics and (view.Sigma_K is not None):
        X, Y, Z = Xc
        if abs(Z) < 1e-12:
            Z = 1e-12
        x_n = X / Z
        y_n = Y / Z
        J_fx = np.array([[x_n],[0.0]])
        J_fy = np.array([[0.0],[y_n]])
        J_cx = np.array([[1.0],[0.0]])
        J_cy = np.array([[0.0],[1.0]])
        J_K = np.concatenate([J_fx, J_fy, J_cx, J_cy], axis=1)  # 2-by-4
        Jblocks.append(J_K)
        Sig_blocks.append(np.asarray(view.Sigma_K).reshape(4,4))

    W_i = np.linalg.inv(np.asarray(view.Sigma_pix).reshape(2,2))

    if len(Jblocks) > 0:
        Jtheta_i = np.concatenate(Jblocks, axis=1)
        Sigtheta_i = make_block_diag(Sig_blocks)
    else:
        Jtheta_i = None
        Sigtheta_i = None

    return JX, Jtheta_i, W_i, Sigtheta_i

def triangulation_covariance(X_w, views, use_schur=True):
    JX_list, Jth_list, W_list, Sigth_list = [], [], [], []
    for v in views:
        JX_i, Jth_i, W_i, Sigth_i = jacs_for_view(X_w, v)
        JX_list.append(JX_i)
        W_list.append(W_i)
        if Jth_i is not None:
            Jth_list.append(Jth_i)
            Sigth_list.append(Sigth_i)

    JX = block_stack_rows(JX_list)
    W = block_diag(W_list)

    if (not use_schur) or (len(Jth_list) == 0):
        Hxx = JX.T @ W @ JX
        S = Hxx
        SigmaX = np.linalg.inv(S + 1e-12*np.eye(3))
        return SigmaX, S

    Jth = block_diag(Jth_list)
    Sigth_prior = block_diag(Sigth_list)
    Hxx = JX.T @ W @ JX
    Hxth = JX.T @ W @ Jth
    Hthth = Jth.T @ W @ Jth + np.linalg.inv(Sigth_prior)
    S = Hxx - Hxth @ np.linalg.solve(Hthth, Hxth.T)
    SigmaX = np.linalg.inv(S + 1e-12*np.eye(3))
    return SigmaX, S

def reward_from_covariance(SigmaX, mode="trace"):
    if mode == "trace":
        r = -float(np.sqrt(np.trace(SigmaX)))
    elif mode == "logdet":
        s, logdet = np.linalg.slogdet(SigmaX)
        r = -0.5 * float(logdet)
    elif mode == "eigmax":
        lm = float(np.linalg.eigvalsh(SigmaX).max())
        r = -lm
    else:
        raise ValueError("Unknown mode")
    return r

def example_usage():
    X_w = np.array([5.0, 0.0, 30.0])
    K = Intrinsics(800, 800, 640, 360)
    R = np.eye(3)
    t0 = np.array([-0.5, 0.0, 0.0])
    t1 = np.array([ 0.5, 0.0, 0.0])
    Sigma_pix = np.diag([1.0, 1.0])
    Sig_t = np.diag([0.01, 0.01, 0.02])**2
    Sig_phi = (np.deg2rad(0.1)**2) * np.eye(3)
    Sig_a = np.array([[np.deg2rad(0.03)**2]])
    Sig_b = np.array([[np.deg2rad(0.03)**2]])
    e_alpha = np.array([0.0, 0.0, 1.0])
    e_beta  = np.array([1.0, 0.0, 0.0])

    v0 = View(R, t0, Sigma_pix, Sig_t, Sig_phi, Sig_a, Sig_b, None, e_alpha, e_beta, K, True, True, False)
    v1 = View(R, t1, Sigma_pix, Sig_t, Sig_phi, Sig_a, Sig_b, None, e_alpha, e_beta, K, True, True, False)

    SigmaX, S = triangulation_covariance(X_w, [v0, v1], use_schur=True)
    r = reward_from_covariance(SigmaX, mode="trace")
    return SigmaX, r

if __name__ == "__main__":
    SigmaX, r = example_usage()
    print("SigmaX:\n", SigmaX)
    print("Reward:", r)
