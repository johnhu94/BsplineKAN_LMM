import numpy as np
def rts_smooth_column(data_col, dt, noise_std):
    # 配置参数
    n_iter = len(data_col)
    Q_var = 5.0 # 过程噪声，调大允许快速变化
    Q = np.array([[0.25*dt**4, 0.5*dt**3], [0.5*dt**3, dt**2]]) * Q_var
    R = noise_std**2
    F = np.array([[1, dt], [0, 1]])
    H = np.array([[1, 0]])
    
    # 容器
    x_priors = np.zeros((n_iter, 2, 1)); P_priors = np.zeros((n_iter, 2, 2))
    x_posteriors = np.zeros((n_iter, 2, 1)); P_posteriors = np.zeros((n_iter, 2, 2))
    
    # 1. 前向滤波
    xhat = np.array([[data_col[0]], [0]])
    P = np.eye(2)
    for k in range(n_iter):
        # 预测
        x_minus = F @ xhat
        P_minus = F @ P @ F.T + Q
        x_priors[k] = x_minus; P_priors[k] = P_minus
        # 更新
        K = P_minus @ H.T * (1 / (H @ P_minus @ H.T + R))
        xhat = x_minus + K * (data_col[k] - H @ x_minus)
        P = (np.eye(2) - K @ H) @ P_minus
        x_posteriors[k] = xhat; P_posteriors[k] = P
        
    # 2. 后向平滑 (RTS)
    x_smooth = np.copy(x_posteriors)
    for k in range(n_iter - 2, -1, -1):
        C_k = P_posteriors[k] @ F.T @ np.linalg.pinv(P_priors[k+1])
        x_smooth[k] = x_posteriors[k] + C_k @ (x_smooth[k+1] - x_priors[k+1])
        
    return x_smooth[:, 0, 0]