import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.stats import norm

def solve_dupire_pde(S0, r, initial_vol, K_min, K_max, T_max, N, M, option_type='call'):
    """
    Решение уравнения Дюпира методом Кранка-Николсона
    
    Уравнение Дюпира (forward PDE):
    ∂C/∂T = (1/2)σ²(K,T)K²(∂²C/∂K²) - rK(∂C/∂K)
    
    Parameters:
    S0 - текущая цена спот
    r - безрисковая ставка
    initial_vol - начальная волатильность
    K_min, K_max - диапазон страйков
    T_max - максимальное время
    N - количество шагов по страйку
    M - количество шагов по времени
    """
    
    # Сетка
    K = np.linspace(K_min, K_max, N)
    T = np.linspace(0, T_max, M)
    
    dK = K[1] - K[0]
    dT = T[1] - T[0]
    
    # Матрица цен опционов
    C = np.zeros((M, N))
    
    # Начальное условие (при T=0)
    if option_type == 'call':
        C[0, :] = np.maximum(K - S0, 0)  # Payoff call опциона
    else:
        C[0, :] = np.maximum(S0 - K, 0)  # Payoff put опциона
    
    # Предположим постоянную волатильность для простоты
    # В реальности здесь может быть поверхность σ(K,T)
    sigma_grid = np.full((M, N), initial_vol)
    
    # Основной цикл по времени (forward in time - уравнение Дюпира)
    for m in range(M-1):
        # Построение матриц для Кранка-Николсона
        A = np.zeros((N, N))
        B = np.zeros((N, N))
        
        for i in range(1, N-1):
            K_val = K[i]
            sigma = sigma_grid[m, i]
            
            # Коэффициенты уравнения Дюпира
            alpha = 0.25 * dT * sigma**2 * K_val**2 / dK**2
            beta = 0.25 * dT * r * K_val / dK
            
            # Матрица A (неявная часть - время m+1)
            A[i, i-1] = -alpha + beta
            A[i, i]   = 1 + 2 * alpha
            A[i, i+1] = -alpha - beta
            
            # Матрица B (явная часть - время m)
            B[i, i-1] = alpha - beta
            B[i, i]   = 1 - 2 * alpha
            B[i, i+1] = alpha + beta
        
        # Граничные условия
        # При K = K_min
        if option_type == 'call':
            A[0, 0] = 1
            B[0, 0] = 1
            C[m+1, 0] = 0  # Call: при K=0, цена = 0
            
            A[N-1, N-1] = 1
            B[N-1, N-1] = 1
            C[m+1, N-1] = K_max - S0 * np.exp(-r * T[m+1])  # При большом K
        else:
            A[0, 0] = 1
            B[0, 0] = 1
            C[m+1, 0] = S0 * np.exp(-r * T[m+1])  # Put: при K=0, цена = S0
            
            A[N-1, N-1] = 1
            B[N-1, N-1] = 1
            C[m+1, N-1] = 0  # При большом K
        
        # Правая часть системы
        rhs = B @ C[m, :]
        
        # Решение системы уравнений
        C[m+1, 1:N-1] = la.solve(A[1:N-1, 1:N-1], rhs[1:N-1])
    
    return K, T, C

def calibrate_dupire_volatility(market_prices, K, T, r, S0):
    """
    Калибровка локальной волатильности по формуле Дюпира
    используя рыночные цены опционов
    """
    M, N = market_prices.shape
    local_vol = np.zeros((M, N))
    
    dT = T[1] - T[0] if len(T) > 1 else 0.1
    dK = K[1] - K[0]
    
    for m in range(1, M-1):
        for n in range(1, N-1):
            # Численные производные
            dC_dT = (market_prices[m+1, n] - market_prices[m-1, n]) / (2 * dT)
            dC_dK = (market_prices[m, n+1] - market_prices[m, n-1]) / (2 * dK)
            d2C_dK2 = (market_prices[m, n+1] - 2*market_prices[m, n] + market_prices[m, n-1]) / (dK**2)
            
            # Формула Дюпира для локальной волатильности
            if d2C_dK2 > 1e-8:
                numerator = 2 * (dC_dT + r * K[n] * dC_dK)
                denominator = (K[n]**2) * d2C_dK2
                
                if numerator > 0 and denominator > 0:
                    local_vol[m, n] = np.sqrt(numerator / denominator)
                else:
                    local_vol[m, n] = np.nan
            else:
                local_vol[m, n] = np.nan
    
    return local_vol

def plot_dupire_cn_solution(K, T, C, local_vol=None):
    """
    Визуализация решения уравнения Дюпира
    """
    T_grid, K_grid = np.meshgrid(T, K)
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Поверхность цен опционов
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(K_grid, T_grid, C.T, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Strike Price (K)')
    ax1.set_ylabel('Time (T)')
    ax1.set_zlabel('Option Price')
    ax1.set_title('Решение уравнения Дюпира\n(Метод Кранка-Николсона)')
    plt.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # 2. Срезы по времени
    ax2 = fig.add_subplot(2, 2, 2)
    time_indices = [int(0.1*len(T)), int(0.5*len(T)), int(0.9*len(T))]
    colors = ['red', 'blue', 'green']
    
    for idx, color in zip(time_indices, colors):
        ax2.plot(K, C[idx, :], color=color, 
                label=f'T = {T[idx]:.2f}', linewidth=2)
    
    ax2.set_xlabel('Strike Price (K)')
    ax2.set_ylabel('Option Price')
    ax2.set_title('Срезы по времени')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Локальная волатильность (если есть)
    if local_vol is not None:
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        clean_vol = np.nan_to_num(local_vol, nan=0.0)
        surf3 = ax3.plot_surface(K_grid, T_grid, clean_vol.T, 
                                cmap='plasma', alpha=0.8)
        ax3.set_xlabel('Strike Price (K)')
        ax3.set_ylabel('Time (T)')
        ax3.set_zlabel('Local Volatility')
        ax3.set_title('Локальная волатильность Дюпира')
        plt.colorbar(surf3, ax=ax3, shrink=0.5)
    
    # 4. Payoff сравнение
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(K, C[0, :], 'k--', label='Начальное условие (T=0)', linewidth=2)
    ax4.plot(K, C[-1, :], 'r-', label=f'Решение (T={T[-1]:.2f})', linewidth=2)
    ax4.set_xlabel('Strike Price (K)')
    ax4.set_ylabel('Option Price')
    ax4.set_title('Эволюция цен во времени')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_with_black_scholes(S0, K_bs, T_bs, r, sigma, C_dupire, K_dupire, T_dupire):
    """
    Сравнение с аналитическим решением Блэка-Шоулза
    """
    # Аналитическое решение
    d1 = (np.log(S0/K_bs) + (r + 0.5*sigma**2)*T_bs) / (sigma*np.sqrt(T_bs))
    d2 = d1 - sigma*np.sqrt(T_bs)
    C_bs = S0 * norm.cdf(d1) - K_bs * np.exp(-r*T_bs) * norm.cdf(d2)
    
    # Находим ближайший страйк и время в сетке Дюпира
    K_idx = np.argmin(np.abs(K_dupire - K_bs))
    T_idx = np.argmin(np.abs(T_dupire - T_bs))
    
    # Численное решение Дюпира
    C_dupire_val = C_dupire[T_idx, K_idx]
    
    print(f"\n--- Сравнение методов при T={T_bs:.2f}, K={K_bs:.1f} ---")
    print(f"Блэка-Шоулз (аналитический): {C_bs:.4f}")
    print(f"Дюпир (Кранк-Николсон): {C_dupire_val:.4f}")
    print(f"Разница: {abs(C_bs - C_dupire_val):.6f}")
    print(f"Относительная ошибка: {abs(C_bs - C_dupire_val)/C_bs*100:.2f}%")
    
    return C_bs, C_dupire_val