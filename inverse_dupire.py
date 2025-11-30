import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter
from black_scholes import black_scholes

# -----------------------------
# 1. Численные производные
# -----------------------------

def compute_derivatives(K, T, C):
    dK = K[1] - K[0]
    dT = T[1] - T[0]

    M, N = C.shape

    dC_dT = np.zeros_like(C)
    dC_dK = np.zeros_like(C)
    d2C_dK2 = np.zeros_like(C)

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            dC_dT[i, j] = (C[i + 1, j] - C[i - 1, j]) / (2 * dT)
            dC_dK[i, j] = (C[i, j + 1] - C[i, j - 1]) / (2 * dK)
            d2C_dK2[i, j] = (C[i, j + 1] - 2 * C[i, j] + C[i, j - 1]) / (dK**2)

    return dC_dT, dC_dK, d2C_dK2


# -----------------------------
# 2. Инверсная формула Дюпира
# -----------------------------

def invert_dupire(K, T, C, r, smoothing=0.8):
    dC_dT, dC_dK, d2C_dK2 = compute_derivatives(K, T, C)

    sigma_lv = np.full_like(C, np.nan)

    for i in range(1, len(T) - 1):
        for j in range(1, len(K) - 1):

            numerator = 2 * (dC_dT[i, j] + r * K[j] * dC_dK[i, j])
            denominator = (K[j]**2) * d2C_dK2[i, j]

            if denominator > 1e-8 and numerator > 0:
                sigma_lv[i, j] = np.sqrt(numerator / denominator)

    # сглаживание для устойчивости
    sigma_lv = gaussian_filter(sigma_lv, sigma=smoothing)

    return sigma_lv


# ------------------------------------------
# 3. Решение PDE Дюпира с найденной волатильностью
# ------------------------------------------

def solve_forward_dupire(K, T, sigma_lv, r, S0):
    N = len(K)
    M = len(T)
    dK = K[1] - K[0]
    dT = T[1] - T[0]

    C = np.zeros((M, N))
    C[0, :] = np.maximum(K - S0, 0)

    for i in range(M - 1):
        A = np.zeros((N, N))
        B = np.zeros((N, N))

        for j in range(1, N - 1):
            alpha = 0.25 * dT * sigma_lv[i, j]**2 * (K[j]**2) / dK**2
            beta = 0.25 * dT * r * K[j] / dK

            A[j, j - 1] = -alpha + beta
            A[j, j] = 1 + 2 * alpha
            A[j, j + 1] = -alpha - beta

            B[j, j - 1] = alpha - beta
            B[j, j] = 1 - 2 * alpha
            B[j, j + 1] = alpha + beta

        A[0, 0] = A[-1, -1] = 1
        B[0, 0] = B[-1, -1] = 1

        rhs = B @ C[i, :]

        C[i + 1, 1:-1] = np.linalg.solve(A[1:-1, 1:-1], rhs[1:-1])

    return C
