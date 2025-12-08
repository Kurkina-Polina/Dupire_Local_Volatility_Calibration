import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

from crank_nicolson_dupire import solve_dupire_pde
import yfinance as yf

def _laplacian_penalty(alpha_grid):
    """
    Вычисляет штраф за гладкость на основе дискретного лапласиана для сетки значений дисперсии.

    Реализует регуляризационный штраф, основанный на аппроксимации лапласиана второго порядка,
    который измеряет локальные изменения значений на сетке. Штраф вычисляется как сумма
    квадратов разностей между центральной точкой и её четырьмя ближайшими соседями
    (по вертикали и горизонтали).
    
    Формула: penalty = Σ [ (α(i,j) - α(i-1,j))² + (α(i,j) - α(i+1,j))² +
                         (α(i,j) - α(i,j-1))² + (α(i,j) - α(i,j+1))² ] / ((m-2)*(n-2))

    Параметры:
    alpha_grid : ndarray, shape (m, n)
        2D сетка значений дисперсии или волатильности в логарифмической шкале.
        Обычно представляет собой логарифмы узлов локальной волатильности.
    
    Возвращает:
    float
        Нормализованное значение штрафа за гладкость. Усредняется по всем внутренним точкам
        для независимости от размера сетки.
    """
    pen = 0.0
    m, n = alpha_grid.shape
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            center = alpha_grid[i, j]
            pen += (center - alpha_grid[i - 1, j]) ** 2
            pen += (center - alpha_grid[i + 1, j]) ** 2
            pen += (center - alpha_grid[i, j - 1]) ** 2
            pen += (center - alpha_grid[i, j + 1]) ** 2
    return pen / max((m - 2) * (n - 2), 1)


def _build_sigma_grid(alpha, K_nodes, T_nodes, K_full, T_full):
    """
    Построение сетки волатильности методом билинейной интерполяции из разреженных узлов дисперсии.

    Параметры:
    alpha : ndarray, shape (len(T_nodes)*len(K_nodes),)
        Вектор параметров в логарифмической шкале, представляющий значения дисперсии
        в узлах (T_nodes, K_nodes). Упорядочен как [α(T0,K0), α(T0,K1), ..., α(T1,K0), ...]
    K_nodes : ndarray, shape (M,)
        Разреженная сетка цен исполнения (страйков) для узлов дисперсии
    T_nodes : ndarray, shape (N,)
        Разреженная сетка времен до экспирации для узлов дисперсии
    K_full : ndarray, shape (P,)
        Полная плотная сетка цен исполнения для решателя PDE
    T_full : ndarray, shape (Q,)
        Полная плотная сетка времен до экспирации для решателя PDE

    Возвращает:
    sigma_grid : ndarray, shape (Q, P)
        2D сетка локальных волатильностей на полной сетке (T_full × K_full)

    Особенности:
    ------------
    - Использует билинейную интерполяцию для плавного перехода между узлами
    """
    alpha_grid = alpha.reshape(len(T_nodes), len(K_nodes))
    interp = RegularGridInterpolator((T_nodes, K_nodes), alpha_grid, bounds_error=False, fill_value=None)
    TT, KK = np.meshgrid(T_full, K_full, indexing="ij")
    pts = np.stack([TT.ravel(), KK.ravel()], axis=1)
    sigma_grid = np.sqrt(np.clip(interp(pts).reshape(TT.shape), 1e-8, None))
    return sigma_grid


def calibrate_local_vol(market_prices, K_full, T_full, K_nodes, T_nodes, sigma_init, r, S0, lam=1e-2, maxiter=50, verbose=True):
    """
    Решение обратной задачи: калибровка поверхности локальной волатильности σ(K,T) по рыночным ценам опционов.

    Параметры:
    market_prices : ndarray, shape (M_full, N_full)
        Матрица целевых цен опционов CALL на полной сетке K_full × T_full
    K_full : ndarray, shape (N_full,)
        Полная сетка цен исполнения, используемая в решателе PDE
    T_full : ndarray, shape (M_full,)
        Полная сетка времен до экспирации, используемая в решателе PDE
    K_nodes : ndarray, shape (M_nodes,)
        Разреженная сетка страйков для параметров дисперсии (M_nodes << N_full)
    T_nodes : ndarray, shape (N_nodes,)
        Разреженная сетка времен для параметров дисперсии (N_nodes << M_full)
    sigma_init : float
        Начальное предположение о волатильности (постоянная по всей поверхности)
    r : float
        Безрисковая процентная ставка
    S0 : float
        Текущая цена базового актива
    lam : float, default=1e-2
        Вес Тихоновской регуляризации, контролирующий гладкость поверхности
    maxiter : int, default=50
        Максимальное количество итераций оптимизатора
    verbose : bool, default=True
        Флаг вывода информации о процессе оптимизации

    Возвращает:
    sigma_calibrated : ndarray, shape (M_full, N_full)
        Откалиброванная поверхность локальной волатильности на полной сетке PDE
    res : OptimizeResult
        Результат работы оптимизатора с информацией о сходимости и истории оптимизации

    """
    M_full, N_full = market_prices.shape
    alpha0 = np.full((len(T_nodes), len(K_nodes)), sigma_init**2)
    alpha0_vec = alpha0.ravel()
    
    iteration = [0]
    best_loss = [float('inf')]

    def loss(alpha_vec):
        """
        Целевая функция для оптимизации поверхности локальной волатильности.

        Вычисляет комбинированную функцию потерь, состоящую из двух компонентов:
        1. Несогласие (misfit) - квадратичная ошибка между модельными и рыночными ценами опционов
        2. Регуляризация (regularization) - штраф за негладкость поверхности дисперсии

        Полная потеря: L(α) = MSE(C_model, C_market) + λ × Laplacian(exp(α))

        alpha_vec : ndarray, shape (len(T_nodes)*len(K_nodes),)
            Вектор параметров в логарифмической шкале, представляющий значения дисперсии
            на разреженной сетке узлов (T_nodes, K_nodes)

        Возвращает:
        float
            Общее значение функции потерь, включающее несогласие с рыночными данными
            и штраф за негладкость поверхности
        """
        iteration[0] += 1
        alpha_grid = alpha_vec.reshape(len(T_nodes), len(K_nodes))
        sigma_grid = _build_sigma_grid(alpha_vec, K_nodes, T_nodes, K_full, T_full)
        sigma_grid = np.clip(sigma_grid, 0.01, 2.0)
        
        _, _, C_model = solve_dupire_pde(
            S0=S0,
            r=r,
            initial_vol=sigma_init,
            K_min=K_full[0],
            K_max=K_full[-1],
            T_max=T_full[-1],
            N=len(K_full),
            M=len(T_full),
            sigma_grid=sigma_grid,
        )
        misfit = np.mean((C_model - market_prices) ** 2)
        reg = _laplacian_penalty(alpha_grid)
        total_loss = misfit + lam * reg
        
        if total_loss < best_loss[0]:
            best_loss[0] = total_loss
            if verbose and iteration[0] % 2 == 0:
                with open('Iterations_calibrate_local.txt', 'a') as f:
                    f.write(f"    Итерация {iteration[0]:3d}: потеря={total_loss:.6f}, ошибка={misfit:.6f}, регуляризация={reg:.6f}\n")
        
        return total_loss

    bounds = [(1e-6, 4.0)] * alpha0_vec.size
    res = minimize(loss, alpha0_vec, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "disp": False, "ftol": 1e-6, "gtol": 1e-5})
    sigma_calibrated = _build_sigma_grid(res.x, K_nodes, T_nodes, K_full, T_full)
    return sigma_calibrated, res


def inverse_problem(K_full, T_full, S0, r,
                    K_nodes, T_nodes, K_min, K_max,
                    sigma_init, N, M, stock, evaluation_date, T_max = 1.0):
    """
    Решение обратной задачи калибровки локальной волатильности.

    Параметры:
    ----------
    K_full : ndarray
        Полная сетка цен исполнения
    T_full : ndarray
        Полная сетка времен до экспирации
    S0 : float
        Текущая цена базового актива
    r : float
        Безрисковая процентная ставка
    K_nodes : ndarray, optional
        Разреженная сетка страйков для параметризации
    T_nodes : ndarray, optional
        Разреженная сетка времен для параметризации
    K_min : float, default=60
        Минимальный страйк
    K_max : float, default=140
        Максимальный страйк
    sigma_init : float, default=0.2
        Начальное предположение о волатильности
    N : int, default=100
        Количество точек по страйкам в PDE сетке
    M : int, default=60
        Количество точек по времени в PDE сетке

    Возвращает:
    -----------
    sigma_calibrated : ndarray
        Откалиброванная поверхность волатильности
    res : OptimizeResult
        Результат оптимизации
    """
    print(f"  Базовая цена актива (S0): {S0:.2f}")
    print(f"  Безрисковая ставка (r): {r:.4f} ({r*100:.2f}%)")
    print(f"  Начальное предположение (sigma_init): {sigma_init:.4f}")
    print(f"  Диапазон страйков: {K_min:.1f} - {K_max:.1f}")
    print(f"  Размерность PDE сетки: {N} страйков × {M} временных точек")
    print(f"  Длина T_full: {len(T_full)} точек, диапазон: [{T_full.min():.3f}, {T_full.max():.3f}]")
    print(f"  Длина K_full: {len(K_full)} точек, диапазон: [{K_full.min():.2f}, {K_full.max():.2f}]")

    sigma_grid = np.full((M, N), sigma_init)
    expirations = stock.options
    try:
        # Загружаем исторические данные для получения цены на нужную дату
        hist = stock.history(start=evaluation_date, end=evaluation_date)
        if len(hist) == 0:
            # Если нет данных на точную дату, берем ближайшую предыдущую
            hist = stock.history(period="5d", end=evaluation_date)
        S0 = float(hist['Close'].iloc[-1])
        print(f"Цена акции на {evaluation_date}: ${S0:.2f}")
    except Exception as e:
        print(f"Не удалось получить цену акции: {e}")
        S0 = 100.0

    # Собираем данные по опционам CALL для каждой экспирации
    all_options_data = []

    _, _, market_prices = solve_dupire_pde(
        S0=S0,
        r=r,
        initial_vol=sigma_grid.mean(),
        K_min=K_min,
        K_max=K_max,
        T_max=T_max,
        N=N,
        M=M,
    )

    sigma_calibrated, res = calibrate_local_vol(
        market_prices=market_prices,
        K_full=K_full,
        T_full=T_full,
        K_nodes=K_nodes,
        T_nodes=T_nodes,
        sigma_init=sigma_init,
        r=r,
        S0=S0,
        lam=2e-3,  # Smaller regularization for better fit
        maxiter=50,
        verbose=True,
    )

    stats = {
        "min": float(np.min(sigma_calibrated)),
        "max": float(np.max(sigma_calibrated)),
        "mean": float(np.mean(sigma_calibrated)),
        "median": float(np.median(sigma_calibrated)),
    }

    print(f"\nКалибровка завершена!")

    print(f"Статистика восстановленной волатильности:")
    print(f"мин={stats['min']:.4f}, макс={stats['max']:.4f}")
    print(f"среднее={stats['mean']:.4f}, медиана={stats['median']:.4f}")


    # Визуализация: сохраняем heatmap оцененной волатильности
    fig, ax = plt.subplots(figsize=(10, 4))
    im1 = ax.imshow(sigma_calibrated, extent=[K_min, K_max, T_max, 0], aspect="auto", cmap="viridis")
    ax.set_title("Calibrated sigma")
    ax.set_xlabel("K")
    ax.set_ylabel("T")
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("inverse_demo_sigma.png", dpi=120)
    plt.close(fig)

    print("График сохранен в  inverse_demo_sigma.png")
    return sigma_calibrated, res

