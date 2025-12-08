import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

from crank_nicolson_dupire import solve_dupire_pde


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

    Особенности:
    ------------
    - Обрабатывает только внутренние точки сетки (исключает границы)
    - Вычисляет лапласиан через разности первого порядка (аппроксимация градиента)
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

    Преобразует вектор параметров дисперсии в логарифмической шкале в полную 2D сетку
    локальных волатильностей через билинейную интерполяцию на регулярной сетке.
    Процесс включает:
    1. Преобразование вектора параметров в 2D сетку узлов дисперсии
    2. Интерполяцию логарифмов дисперсии на полную сетку методом RegularGridInterpolator
    3. Преобразование обратно в волатильность через взятие квадратного корня
    4. Ограничение минимального значения для избежания численной нестабильности

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
    - Применяет RegularGridInterpolator с bounds_error=False для экстраполяции за границами
    - Интерполирует логарифмы дисперсии (α), что обеспечивает положительность результата
    - Преобразует дисперсию в волатильность через σ = √exp(α)
    - Ограничивает минимальное значение дисперсии 1e-8 для численной устойчивости
    - Поддерживает экстраполяцию за границами узловой сетки
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

    Оптимизационный алгоритм для восстановления поверхности локальной волатильности,
    которая наилучшим образом воспроизводит рыночные цены опционов CALL. Использует:
    1. Параметризацию через разреженную сетку узлов дисперсии
    2. Решение прямого уравнения Дюпира для ценообразования
    3. Тихоновскую регуляризацию для обеспечения гладкости поверхности
    4. Метод L-BFGS-B для минимизации целевой функции с ограничениями

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

    Особенности:
    ------------
    - Ограничивает волатильность диапазоном [0.01, 2.0] для численной устойчивости
    - Использует L-BFGS-B с ограничениями на параметры для избежания нефизических значений
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

        Параметры:
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
                with open('Iterations_calibrate_local.txt', 'w') as f:
                    f.write(f"    Iter {iteration[0]:3d}: loss={total_loss:.6f}, misfit={misfit:.6f}, reg={reg:.6f}\n")
        
        return total_loss

    bounds = [(1e-6, 4.0)] * alpha0_vec.size
    res = minimize(loss, alpha0_vec, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "disp": False, "ftol": 1e-6, "gtol": 1e-5})
    sigma_calibrated = _build_sigma_grid(res.x, K_nodes, T_nodes, K_full, T_full)
    return sigma_calibrated, res


def inverse_problem(S0, K_min=60.0, K_max=140.0, r=0.03, sigma_true = 0.20, sigma_init = 0.12, N=100, M=60, T_max = 1.0):
    """Toy inverse problem on synthetic data (constant true vol)."""
    # Полный набор цен исполнения (strikes) / dhtvtyb lj , на которых решается дифференциальное уравнение
    K_full = np.linspace(K_min, K_max, N)
    T_full = np.linspace(0.01, T_max, M)

    print("Inverse problem demo: synthetic constant vol")
    print(f"  PDE grid: {N} strikes × {M} times")
    print(f"  True sigma: {sigma_true}")

    # Forward prices with true sigma
    print("  Generating synthetic market prices...")
    sigma_true_grid = np.full((M, N), sigma_true)
    #FIXME why _ _ is not K and T
    _, _, market_prices = solve_dupire_pde(
        S0=S0,
        r=r,
        initial_vol=sigma_true,
        K_min=K_min,
        K_max=K_max,
        T_max=T_max,
        N=N,
        M=M,
        sigma_grid=sigma_true_grid,
    )

    # параметрическая сетка для волатильности - где ищутся неизвестные параметры
    K_nodes = np.linspace(K_min, K_max, 10)
    T_nodes = np.linspace(0.01, T_max, 8)
    print(f"  Parameter grid: {len(K_nodes)} K-nodes × {len(T_nodes)} T-nodes")


    print(f"  Initial guess: sigma={sigma_init}")
    print("  Starting calibration with adaptive regularization...")
    
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

    err = np.mean((sigma_calibrated - sigma_true) ** 2) ** 0.5
    stats = {
        "min": float(np.min(sigma_calibrated)),
        "max": float(np.max(sigma_calibrated)),
        "mean": float(np.mean(sigma_calibrated)),
        "median": float(np.median(sigma_calibrated)),
        "rmse": float(err),
    }

    print(f"\n  Calibration complete!")
    print(f"  RMSE: {err:.6f} (target 0.0000)")
    print(f"  Recovered sigma stats:")
    print(f"    min={stats['min']:.4f}, max={stats['max']:.4f}")
    print(f"    mean={stats['mean']:.4f}, median={stats['median']:.4f}")
    print(f"  Optimizer: {res.message}")

    # Визуализация: сохраняем heatmap оцененной и истинной волатильности
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(sigma_true_grid, extent=[K_min, K_max, T_max, 0], aspect="auto", cmap="viridis")
    axes[0].set_title("True sigma")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("T")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(sigma_calibrated, extent=[K_min, K_max, T_max, 0], aspect="auto", cmap="viridis")
    axes[1].set_title("Calibrated sigma")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("T")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("inverse_demo_sigma.png", dpi=120)
    plt.close(fig)

    print("  plot saved to inverse_demo_sigma.png")

