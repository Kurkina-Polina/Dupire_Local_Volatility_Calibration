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

loss = 8.765375
def get_option_prices_simple(ticker, evaluation_date, max_expirations=3):
    """
    Пполчение реальных цен опционов.

    Параметры:
    ----------
    ticker : str
        Тикер акции (например, "SPY")
    evaluation_date : str
        Дата анализа в формате 'YYYY-MM-DD'
    max_expirations : int
        Сколько дат экспирации брать (обычно 3)

    Возвращает:
    -----------
    all_calls : DataFrame
        Данные опционов CALL
    S0 : float
        Цена акции на evaluation_date
    """

    import yfinance as yf
    import pandas as pd
    from datetime import datetime

    print(f"Получаем опционы {ticker} на дату {evaluation_date}")

    # 1. Создаем объект тикера
    stock = yf.Ticker(ticker)

    # 2. Получаем цену акции на evaluation_date
    # Берем исторические данные за последние 5 дней от evaluation_date
    hist = stock.history(period="5d", end=evaluation_date)
    S0 = float(hist['Close'].iloc[-1])
    print(f"Цена акции {ticker}: ${S0:.2f}")

    # 3. Получаем список доступных дат экспирации
    expirations = stock.options

    # 4. Создаем список для хранения данных
    all_data = []

    # 5. Обрабатываем каждую экспирацию (первые max_expirations)
    for expiry in expirations[:max_expirations]:

        # Преобразуем даты

        current_date = datetime.strptime(evaluation_date, '%Y-%m-%d').date()  # Только дата
        expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()

        T_days = max(0, (expiry_date - current_date).days)  # Защита от отрицательных значений
        T_years = T_days / 365.0


        # Получаем цепочку опционов
        opt_chain = stock.option_chain(expiry)
        calls = opt_chain.calls.copy()

        # Проверяем, что есть данные
        if calls.empty:
            continue

        # Добавляем вычисляемые поля
        calls['T'] = T_years
        calls['expiry'] = expiry
        calls['moneyness'] = calls['strike'] / S0

        # Фильтруем: оставляем только опционы с ценой > 0
        calls_filtered = calls[calls['lastPrice'] > 0]


        # Отбираем нужные колонки
        calls_selected = calls_filtered[['strike', 'T', 'lastPrice', 'volume']]

        # Переименовываем
        calls_selected.columns = ['K', 'T', 'price', 'volume']

        all_data.append(calls_selected)

    # 6. Объединяем все данные
    if not all_data:
        print("Не удалось получить данные опционов!")
        return None, S0

    all_calls = pd.concat(all_data, ignore_index=True)

    # 7. Сортируем и фильтруем
    all_calls = all_calls.sort_values(['T', 'K'])

    return all_calls, S0


def prepare_market_data_for_calibration(option_data, S0):
    """
    Преобразует DataFrame опционов в данные для калибровки.

    Параметры:
    ----------
    option_data : DataFrame
        Данные опционов из get_option_prices_simple()
    S0 : float
        Цена акции

    Возвращает:
    -----------
    market_prices : ndarray
        Матрица цен опционов
    K_full : ndarray
        Сетка страйков
    T_full : ndarray
        Сетка времён
    """
    import numpy as np

    # Получаем уникальные значения страйков и времён
    unique_K = np.sort(option_data['K'].unique())
    unique_T = np.sort(option_data['T'].unique())

    # Создаем матрицу цен
    market_prices = np.full((len(unique_T), len(unique_K)), np.nan)

    # Заполняем матрицу
    for idx, row in option_data.iterrows():
        t_idx = np.where(unique_T == row['T'])[0][0]
        k_idx = np.where(unique_K == row['K'])[0][0]
        market_prices[t_idx, k_idx] = row['price']

    print(f"Создана матрица цен: {market_prices.shape[0]} времён × {market_prices.shape[1]} страйков")
    print(f"Заполнено: {np.sum(~np.isnan(market_prices))} из {market_prices.size} ячеек")

    return market_prices, unique_K, unique_T

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



def calibrate_volatility_surface(market_prices, K_full, T_full, S0, r,
                                 K_nodes=None, T_nodes=None,
                                 sigma_init=0.20, lam=2e-3, maxiter=50,
                                 plot_results=True, verbose=True):
    """
    Калибровка поверхности локальной волатильности по рыночным данным.

    Параметры:
    ----------
    market_prices : ndarray, shape (M, N)
        РЕАЛЬНЫЕ рыночные цены опционов CALL на сетке K_full × T_full
    K_full : ndarray, shape (N,)
        Сетка страйков из рыночных данных
    T_full : ndarray, shape (M,)
        Сетка времен до экспирации из рыночных данных (в годах)
    S0 : float
        Текущая цена базового актива
    r : float
        Безрисковая процентная ставка
    K_nodes : ndarray, optional
        Разреженная сетка страйков для параметризации
    T_nodes : ndarray, optional
        Разреженная сетка времен для параметризации
    sigma_init : float, default=0.20
        Начальное предположение о волатильности
    lam : float, default=2e-3
        Коэффициент регуляризации
    maxiter : int, default=50
        Максимальное число итераций
    plot_results : bool, default=True
        Сохранять ли визуализацию
    verbose : bool, default=True
        Выводить ли информацию о процессе

    Возвращает:
    -----------
    sigma_calibrated : ndarray, shape (M, N)
        Откалиброванная поверхность локальной волатильности
    res : OptimizeResult
        Результат оптимизации
    """

    # Вывод начальных параметров
    print("Начальные параметры")
    print(f"  Базовая цена актива (S0): {S0:.2f}")
    print(f"  Безрисковая ставка (r): {r:.4f} ({r*100:.2f}%)")
    print(f"  Начальное предположение (sigma_init): {sigma_init:.4f}")
    print(f"  Размерность данных: {len(T_full)}×{len(K_full)} (T×K)")
    print(f"  Диапазон страйков: [{K_full.min():.2f}, {K_full.max():.2f}]")
    print(f"  Диапазон времён: [{T_full.min():.3f}, {T_full.max():.3f}] лет")

    # Проверка наличия NaN в рыночных ценах
    nan_count = np.sum(np.isnan(market_prices))
    if nan_count > 0:
        print(f"  Внимание: {nan_count} NaN значений в рыночных ценах")
        # Заменяем NaN на 0 для численной устойчивости
        market_prices_filled = np.nan_to_num(market_prices, nan=0.0)
    else:
        market_prices_filled = market_prices

    print(f"  Диапазон цен: [{np.nanmin(market_prices):.2f}, {np.nanmax(market_prices):.2f}]")

    # Автоматическое создание узловых сеток, если не заданы
    if K_nodes is None:
        K_nodes = np.linspace(K_full.min(), K_full.max(), 10)
        print(f"  Создана узловая сетка по страйкам: {len(K_nodes)} точек")

    if T_nodes is None:
        T_nodes = np.linspace(T_full.min(), T_full.max(), 8)
        print(f"  Создана узловая сетка по времени: {len(T_nodes)} точек")

    print(f"  Коэффициент регуляризации: {lam}")
    print(f"  Максимальное число итераций: {maxiter}")

    # Калибровка локальной волатильности
    sigma_calibrated, res = calibrate_local_vol(
        market_prices=market_prices_filled,  # Используем заполненные данные
        K_full=K_full,
        T_full=T_full,
        K_nodes=K_nodes,
        T_nodes=T_nodes,
        sigma_init=sigma_init,
        r=r,
        S0=S0,
        lam=lam,
        maxiter=maxiter,
        verbose=verbose,
    )

    # Статистика результатов
    print("\nРезультаты")
    print(f"  Статус оптимизации: {res.message}")
    print(f"  Количество итераций: {res.nit}")
    print(f"  Финальное значение функции потерь: {loss:.6f}")

    # Вычисляем статистику калиброванной поверхности
    stats = {
        "min": float(np.nanmin(sigma_calibrated)),
        "max": float(np.nanmax(sigma_calibrated)),
        "mean": float(np.nanmean(sigma_calibrated)),
        "median": float(np.nanmedian(sigma_calibrated)),
        "std": float(np.nanstd(sigma_calibrated)),
    }

    print(f"\n  Статистика поверхности волатильности:")
    print(f"    Мин:    {stats['min']:.4f}")
    print(f"    Макс:   {stats['max']:.4f}")
    print(f"    Среднее: {stats['mean']:.4f}")
    print(f"    Медиана: {stats['median']:.4f}")
    print(f"    Стандартное отклонение: {stats['std']:.4f}")

    # Оценка качества
    print(f"\n  Оценка качества калибровки:")
    print(f"    Средняя волатильность: {stats['mean']:.4f}")
    print(f"    Размах (max-min): {stats['max'] - stats['min']:.4f}")


    # Визуализация результатов
    if plot_results:
        plt.figure(figsize=(10, 6))

        # Heatmap калиброванной волатильности
        plt.imshow(sigma_calibrated,
                   extent=[K_full.min(), K_full.max(), T_full.max(), T_full.min()],
                   aspect="auto",
                   cmap="viridis")

        plt.title("Калиброванная волатильность")
        plt.xlabel("Страйк (K)")
        plt.ylabel("Время (T)")

        plt.colorbar(label='Волатильность σ')

        plt.tight_layout()
        plt.savefig("calibrated_volatility_surface.png", dpi=120)
        plt.close()
        print("График сохранен в 'calibrated_volatility_surface.png'")

    return sigma_calibrated, res
