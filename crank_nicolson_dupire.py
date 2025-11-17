import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.stats import norm

def solve_dupire_pde(S0, r, initial_vol, K_min, K_max, T_max, N, M):
    """
    Решение уравнения Дюпира методом Кранка-Николсона
    
    Уравнение Дюпира (forward PDE):
    ∂C/∂T = (1/2)σ²(K,T)K²(∂²C/∂K²) - rK(∂C/∂K)
    
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
    # Начальное условие (при T=0)
    C[0, :] = np.maximum(K - S0, 0)  # Payoff call опциона

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
        A[0, 0] = 1
        B[0, 0] = 1
        C[m+1, 0] = 0  # Call: при K=0, цена = 0

        A[N-1, N-1] = 1
        B[N-1, N-1] = 1
        C[m+1, N-1] = K_max - S0 * np.exp(-r * T[m+1])  # При большом K
        
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
    T_grid, K_grid = np.meshgrid(T, K)
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Поверхность цен опционов
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(K_grid, T_grid, C.T, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Цена исполнения (K)')
    ax1.set_ylabel('Время до экспирации (T)')
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
    
    ax2.set_xlabel('Цена исполнения (K)')
    ax2.set_ylabel('Option Price')
    ax2.set_title('Срезы по времени')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 4. Сравнение начальных и конечных выплат
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(K, C[0, :], 'k--', label='Начальное условие (T=0)', linewidth=2)
    ax4.plot(K, C[-1, :], 'r-', label=f'Решение (T={T[-1]:.2f})', linewidth=2)
    ax4.set_xlabel('Цена исполнения (K)')
    ax4.set_ylabel('Option Price')
    ax4.set_title('Эволюция цен во времени')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

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

def build_dupire_surface_cn(data, S_t, r, sigma, tau):
    """
   Строит поверхность цен опционов методом Крэнка-Николсона для уравнения Дюпира:
    Решает PDE уравнения Дюпира численным методом для получения цен C(K,T)

    Параметры:
    data : pandas.DataFrame
        DataFrame с историческими данными (используется для контекста)
    S_t : float
        Текущая цена базового актива
    r : float
        Безрисковая процентная ставка
    sigma : float
        Начальная волатильность для PDE
    tau : float
        Время до экспирации (используется для контекста)
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
    K_min = S_t * 0.5
    K_max = S_t * 1.5
    T_max = 1.0
    N = 100  # шаги по страйку
    M = 100  # шаги по времени

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
    local_vol_grid = np.zeros_like(C_grid)
    M, N = C_grid.shape

    dT = T_grid[1, 0] - T_grid[0, 0] if M > 1 else 0.1
    dK = K_grid[0, 1] - K_grid[0, 0]

    for i in range(1, M-1):
        for j in range(1, N-1):
            K_val = K_grid[i, j]

            # Численные производные с центральными разностями
            dC_dT = (C_grid[i+1, j] - C_grid[i-1, j]) / (2 * dT)
            dC_dK = (C_grid[i, j+1] - C_grid[i, j-1]) / (2 * dK)
            d2C_dK2 = (C_grid[i, j+1] - 2*C_grid[i, j] + C_grid[i, j-1]) / (dK**2)

            # Формула Дюпира
            if d2C_dK2 > 1e-10:  # избегаем деления на 0
                numerator = 2 * (dC_dT + r * K_val * dC_dK)
                denominator = (K_val**2) * d2C_dK2

                if numerator > 0 and denominator > 0:
                    local_vol_grid[i, j] = np.sqrt(numerator / denominator)
                else:
                    local_vol_grid[i, j] = np.nan
            else:
                local_vol_grid[i, j] = np.nan

    return local_vol_grid