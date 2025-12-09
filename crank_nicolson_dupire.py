import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline

def solve_dupire_pde(S0, r, initial_vol, K_min, K_max, T_max, N, M, sigma_grid=None):
    """
    Решение уравнения Дюпира методом Кранка-Николсона
    
    Уравнение Дюпира (forward PDE):
    ∂C/∂T = (1/2)σ²(K,T)K²(∂²C/∂K²) - rK(∂C/∂K) + r·C
    
    Параметры:
    S0 : float
        Текущая цена спот базового актива
    r : float
        Безрисковая процентная ставка (годовая)
    initial_vol : float
        Начальное значение волатильности (используется для всей поверхности)
    K_min : float
        Минимальное значение страйка в сетке
    K_max : float
        Максимальное значение страйка в сетке
    T_max : float
        Максимальное время до экспирации (в годах)
    N : int
        Количество узлов сетки по страйку (пространственная дискретизация)
    M : int
        Количество узлов сетки по времени (временная дискретизация)

    Возвращает:
    K : ndarray
        Вектор страйков размерности N
    T : ndarray
        Вектор времен до экспирации размерности M
    C : ndarray
        Матрица цен опционов размерности M×N, где C[m,i] - цена при времени T[m] и страйке K[i]
    """
    
    # Сетка
    K = np.linspace(K_min, K_max, N)
    T = np.linspace(0, T_max, M)
    
    dK = K[1] - K[0]
    dT = T[1] - T[0]
    
    # Матрица цен опционов
    C = np.zeros((M, N))
    # Начальное условие (при T=0): выплата колла max(S0 - K, 0)
    C[0, :] = np.maximum(S0 - K, 0)

    # Волатильность: если передана sigma_grid — используем её, иначе константа
    if sigma_grid is None:
        sigma_grid = np.full((M, N), initial_vol)
    
    # Основной цикл по времени (forward in time - уравнение Дюпира)
    for m in range(M-1):
        # Построение матриц для Кранка-Николсона
        A = np.zeros((N, N))  # (I - 0.5*dt*L)
        B = np.zeros((N, N))  # (I + 0.5*dt*L)

        for i in range(1, N-1):
            K_val = K[i]
            sigma = sigma_grid[m, i]

            # Коэффициенты оператора L для PDE Дюпира: C_T = L(C)
            alpha = 0.5 * sigma**2 * K_val**2 / (dK**2)
            beta = r * K_val / (2 * dK)
            gamma = r

            left = alpha + beta
            center = -2 * alpha + gamma
            right = alpha - beta

            A[i, i-1] = -0.5 * dT * left
            A[i, i] = 1 - 0.5 * dT * center
            A[i, i+1] = -0.5 * dT * right

            B[i, i-1] = 0.5 * dT * left
            B[i, i] = 1 + 0.5 * dT * center
            B[i, i+1] = 0.5 * dT * right

        # Граничные условия для call: C(T,0)=S0, C(T,K_max)->0
        A[0, 0] = 1
        B[0, 0] = 1
        C[m+1, 0] = S0

        A[N-1, N-1] = 1
        B[N-1, N-1] = 1
        C[m+1, N-1] = 0.0

        # Правая часть системы
        rhs = B @ C[m, :]

        # Решение системы уравнений только по внутренним узлам
        C[m+1, 1:N-1] = la.solve(A[1:N-1, 1:N-1], rhs[1:N-1])
    
    return K, T, C

def calibrate_dupire_volatility(market_prices, K, T, r):
    """
    Калибровка локальной волатильности по формуле Дюпира с использованием рыночных цен опционов.

    Реализует численное вычисление локальной волатильности на основе классической формулы Дюпира:
    σ²(K,T) = [2·(∂C/∂T + rK·∂C/∂K)] / [K²·∂²C/∂K²]
    где C - цена опциона CALL, K - страйк, T - время до экспирации, r - безрисковая ставка.

    Параметры:
    market_prices : ndarray, shape (M, N)
        2D массив рыночных цен опционов CALL, где M - количество временных точек,
        N - количество страйков
    K : ndarray, shape (N,)
        Массив цен исполнения (страйков)
    T : ndarray, shape (M,)
        Массив времен до экспирации (в годах)
    r : float
        Безрисковая процентная ставка (годовая)
    S0 : float
        Текущая цена базового актива (используется для проверок, но не в формуле Дюпира)

    Возвращает:
    local_vol : ndarray, shape (M, N)
        2D массив локальных волатильностей, рассчитанных по формуле Дюпира.
        Граничные точки заполнены нулями, внутренние точки с некорректными значениями - NaN.

    Особенности:
    ------------
    - Использует центральные разности для вычисления производных первого и второго порядка
    - Пропускает граничные точки (индексы 0 и M-1 по времени, 0 и N-1 по страйкам)
    - Проверяет условие d2C_dK2 > 1e-8 для избежания численной нестабильности
    - Автоматически определяет шаги dT и dK из входных массивов
    - В случае нарушения условий устойчивости возвращает NaN
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

def plot_dupire_cn_solution(K, T, C, local_vol=None, save_folder="./"):
    """
    Калибрует локальную волатильность по формуле Дюпира используя рыночные цены опционов.

    Формула Дюпира для локальной волатильности:
    σ²(K,T) = [2·(∂C/∂T + rK·∂C/∂K)] / [K²·∂²C/∂K²]

    Локальная волатильность показывает, какую волатильность должен иметь базовый актив,
    чтобы теоретическая цена опциона совпадала с рыночной для каждого страйка и срока.

    Параметры:
    market_prices : ndarray
        2D массив рыночных цен опционов CALL размером [M, N], где:
        M - количество временных точек (сроков экспирации)
        N - количество страйков
    K : ndarray
        1D массив цен исполнения (страйков) длиной N
    T : ndarray
        1D массив времен до экспирации (в годах) длиной M
    r : float
        Безрисковая процентная ставка (годовая, в десятичной форме)
    S0 : float
        Текущая цена базового актива

     Возвращает:
    local_vol : ndarray
        2D массив локальных волатильностей размером [M, N]
        Граничные точки заполнены NaN из-за невозможности вычисления производных

    Особенности:
    ------------
    - Использует центральные разности для численного дифференцирования
    - Граничные точки не вычисляются из-за отсутствия соседних точек для производных
    """
    import os
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    T_grid, K_grid = np.meshgrid(T, K)

    # 1. Поверхность цен опционов
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(K_grid, T_grid, C.T, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Цена исполнения (K)')
    ax.set_ylabel('Время до экспирации (T)')
    ax.set_zlabel('Option Price')
    ax.set_title('Dupire CN: Option Price Surface')
    plt.colorbar(surf, ax=ax, shrink=0.5)
    fig.savefig(os.path.join(save_folder, "Dupire_CN_prices.png"), dpi=150)
    plt.close(fig)

    # 2. Срезы по времени
    fig, ax = plt.subplots(figsize=(10, 6))
    time_indices = [int(0.1*len(T)), int(0.5*len(T)), int(0.9*len(T))]
    colors = ['red', 'blue', 'green']
    for idx, color in zip(time_indices, colors):
        ax.plot(K, C[idx, :], color=color, label=f'T = {T[idx]:.2f}', linewidth=2)
    ax.set_xlabel('Цена исполнения (K)')
    ax.set_ylabel('Option Price')
    ax.set_title('Dupire CN: Slices over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(save_folder, "Dupire_CN_slices.png"), dpi=150)
    plt.close(fig)

    # 3. Эволюция цен
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(K, C[0, :], 'k--', label='T=0', linewidth=2)
    ax.plot(K, C[-1, :], 'r-', label=f'T={T[-1]:.2f}', linewidth=2)
    ax.set_xlabel('Цена исполнения (K)')
    ax.set_ylabel('Option Price')
    ax.set_title('Dupire CN: Price Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(save_folder, "Dupire_CN_evolution.png"), dpi=150)
    plt.close(fig)


def compare_with_black_scholes(S0, K_bs, T_bs, r, sigma, C_dupire, K_dupire, T_dupire):
    """
    Сравнивает численное решение уравнения Дюпира методом Крэнка-Николсона
    с аналитической формулой Блэка-Шоулза.

    Проверяет точность численного PDE-решателя против эталонного аналитического решения
    для одинаковых параметров опциона (страйк, время, волатильность).

    Параметры:
    S0 : float
        Текущая цена базового актива
    K_bs : float
        Цена исполнения для сравнения
    T_bs : float
        Время до экспирации для сравнения
    r : float
        Безрисковая процентная ставка
    sigma : float
        Постоянная волатильность для модели Блэка-Шоулза
    C_dupire : ndarray
        2D массив цен опционов, рассчитанных методом Крэнка-Николсона
    K_dupire : ndarray
        1D массив страйков из сетки численного метода
    T_dupire : ndarray
        1D массив времен экспирации из сетки численного метода

    Возвращает:
    tuple (C_bs, C_dupire_val)
        C_bs : float
            Цена опциона по аналитической формуле Блэка-Шоулза
        C_dupire_val : float
            Цена опциона из численного решения Крэнка-Николсона
    Особенности:
    ------------
    - Использует ближайшего соседа для поиска соответствия в численной сетке
    - Выводит абсолютную и относительную ошибку для оценки точности
    - Сравнение выполняется для идентичных параметров (K, T, σ, r)
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

def build_dupire_surface_cn(S_t, r, sigma, N=140, M=140):
    """
   Строит поверхность цен опционов методом Крэнка-Николсона для уравнения Дюпира:
    Решает PDE уравнения Дюпира численным методом для получения цен C(K,T)

    Параметры:
    S_t : float
        Текущая цена базового актива
    r : float
        Безрисковая процентная ставка
    sigma : float
        Начальная волатильность для PDE
    Возвращает:
    tuple (K_grid, T_grid, C_dupire, K, T)
        K_grid : ndarray
            2D сетка цен исполнения [T, K]
        T_grid : ndarray
            2D сетка времени до экспирации [T, K]
        C_dupire : ndarray
            2D сетка цен опционов CALL [T, K]
        K : ndarray
            1D массив страйков
        T : ndarray
            1D массив времен экспирации
    """
    # Параметры для численного решения
    K_min = S_t * 0.6
    K_max = S_t * 1.4
    T_max = 1.0

    print("Решение уравнения Дюпира методом Кранка-Николсона...")

    # Решаем уравнение Дюпира
    K, T, C_dupire = solve_dupire_pde(
        S0=S_t, r=r, initial_vol=sigma,
        K_min=K_min, K_max=K_max, T_max=T_max,
        N=N, M=M
    )

    # Создаем сетку для совместимости с существующим кодом
    K_grid, T_grid = np.meshgrid(K, T)

    return K_grid, T_grid, C_dupire, K, T

def calculate_dupire_volatility_improved(K_grid, T_grid, C_grid, r):
    """
    Вычисляет локальную волатильность по улучшенной формуле Дюпира:
    σ²(K,T) = [2·(∂C/∂T + rK·∂C/∂K)] / [K²·∂²C/∂K²]

    Улучшенная версия с центральными разностями и проверкой устойчивости.

    Параметры:
    K_grid : ndarray
        2D сетка цен исполнения [T, K]
    T_grid : ndarray
        2D сетка времени до экспирации [T, K]
    C_grid : ndarray
        2D сетка цен опционов CALL [T, K]
    r : float
        Безрисковая процентная ставка

    Возвращает:
    local_vol_grid : ndarray
        2D сетка локальных волатильностей [T, K]
    Особенности:
    ------------
    - Использует центральные разности для более точных производных
    - Проверяет условие d2C_dK2 > 1e-10 для избежания деления на ноль
    - Проверяет положительность числителя и знаменателя перед вычислением
    - Заполняет граничные точки значением NaN (не вычисляются)
    - Более устойчива к численным ошибкам по сравнению с базовой версией
    - Автоматически определяет шаги сетки dT и dK
    """
    C_smooth = gaussian_filter(C_grid, sigma=1.0)
    local_vol_grid = np.full_like(C_grid, np.nan)
    M, N = C_smooth.shape

    dT = T_grid[1, 0] - T_grid[0, 0] if M > 1 else 0.1
    dK = K_grid[0, 1] - K_grid[0, 0]

    # Пропускаем два крайних слоя, чтобы производные были устойчивее
    for i in range(2, M-2):
        for j in range(2, N-2):
            K_val = K_grid[i, j]

            dC_dT = (C_smooth[i+1, j] - C_smooth[i-1, j]) / (2 * dT)
            dC_dK = (C_smooth[i, j+1] - C_smooth[i, j-1]) / (2 * dK)
            d2C_dK2 = (C_smooth[i, j+1] - 2*C_smooth[i, j] + C_smooth[i, j-1]) / (dK**2)

            if d2C_dK2 > 1e-8:
                numerator = 2 * (dC_dT + r * K_val * dC_dK)
                denominator = (K_val**2) * d2C_dK2
                if numerator > 0 and denominator > 0:
                    local_vol_grid[i, j] = np.sqrt(numerator / denominator)

    finite_vals = local_vol_grid[np.isfinite(local_vol_grid)]
    if finite_vals.size:
        cap = 3.0 * np.nanmedian(finite_vals)
        local_vol_grid = np.clip(local_vol_grid, 0, cap)

    T_vals = T_grid[:, 0]
    K_vals = K_grid[0, :]

    valid_mask = ~np.isnan(local_vol_grid)
    points_T = T_grid[valid_mask]
    points_K = K_grid[valid_mask]
    values = local_vol_grid[valid_mask]


    values_smooth = gaussian_filter(values, sigma=1)

    spline = RectBivariateSpline(
        T_vals,
        K_vals,
        np.nan_to_num(local_vol_grid, nan=np.nanmedian(finite_vals)),
        kx=3, ky=3
    )

    interpolated = spline(T_vals, K_vals)

    filled = local_vol_grid.copy()
    filled[np.isnan(filled)] = interpolated[np.isnan(filled)]

    filled = gaussian_filter(filled, sigma=1.0)

  
    return filled