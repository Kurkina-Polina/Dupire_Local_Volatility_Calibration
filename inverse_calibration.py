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


def inverse_problem(S0, K_min, K_max, r, sigma_true, sigma_init, N, M, T_max = 1.0):
    """Toy inverse problem on synthetic data (constant true vol)."""
    # Полный набор цен исполнения (strikes) / времени до экспирации, на которых решается дифференциальное уравнение
    K_full = np.linspace(K_min, K_max, N)
    T_full = np.linspace(0.01, T_max, M)

    print(f"Сетка Уравнения в Частных Производных: {N} страйков × {M} временных шагов")
    print(f"Среднее значение истинной волатильности: {np.nanmean(sigma_true):.4f}")

    # Forward prices with true sigma
    print("Получение Рыночных цен опционов... Это займет время")
    sigma_true_grid = np.full((M, N), sigma_true)
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

    # Параметрическая сетка для волатильности - где ищутся неизвестные параметры
    K_nodes = np.linspace(K_min, K_max, 10)
    T_nodes = np.linspace(0.01, T_max, 8)
    print(f"Сетка параметров: {len(K_nodes)} узлов по K × {len(T_nodes)} узлов по T")
    print(f"Начальное приближение: sigma={sigma_init}")
    print("Идет калибровка с адаптивной регуляризацией...")
    
    sigma_calibrated, res = calibrate_local_vol(
        market_prices=market_prices,
        K_full=K_full,
        T_full=T_full,
        K_nodes=K_nodes,
        T_nodes=T_nodes,
        sigma_init=sigma_init,
        r=r,
        S0=S0,
        lam=2e-3,
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

    print(f"\nСКО: {err:.6f} (целевое значение: 0.0000)")
    print(f"Статистика восстановленной волатильности:")
    print(f"мин={stats['min']:.4f}, макс={stats['max']:.4f}")
    print(f"среднее={stats['mean']:.4f}, медиана={stats['median']:.4f}")

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
    plt.savefig("inverse__sigma.png", dpi=120)
    plt.close(fig)
    print("  plot saved to inverse__sigma.png")

    return sigma_calibrated

def plot_3d_price_comparison(K_array, T_array, C1, C2, label1="C_cn", label2="C_calibrated"):
    """
    Сравнение двух поверхностей цен опционов в 3D.

    Параметры:
        K_array : (N,) — страйки
        T_array : (M,) — времена до экспирации
        C1, C2 : (M, N) — сетки цен
        label1, label2 : подписи для легенды
    """
    K_grid, T_grid = np.meshgrid(K_array, T_array)  # (M, N)

    fig = plt.figure(figsize=(12, 6))

    # Первый подграфик: обе поверхности
    ax1 = fig.add_subplot(121, projection='3d')
    # Первая поверхность — синяя, полупрозрачная
    ax1.plot_surface(K_grid, T_grid, C1,
                    color='blue', alpha=0.6, edgecolor='none', label=label1)

    # Вторая поверхность — оранжевая, тоже полупрозрачная
    ax1.plot_surface(K_grid, T_grid, C2,
                    color='orange', alpha=0.6, edgecolor='none', label=label2)

    ax1.set_xlabel('Strike K')
    ax1.set_ylabel('Time T')
    ax1.set_zlabel('Call Price C')
    ax1.set_title('Сравнение двух поверхностей цен (3D)')

    # Второй подграфик: разность
    ax2 = fig.add_subplot(122, projection='3d')
    diff = C2 - C1
    surf = ax2.plot_surface(K_grid, T_grid, diff, cmap='RdBu', edgecolor='none')
    ax2.set_xlabel('Strike K')
    ax2.set_ylabel('Time T')
    ax2.set_zlabel('ΔC = C_cal - C_cn')
    ax2.set_title('Разность цен (3D)')
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.savefig("3d_price_comparison.png", dpi=150)
    plt.show()

