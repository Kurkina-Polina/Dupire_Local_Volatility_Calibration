from crank_nicolson_dupire import *
from dupire import *
from inverse_calibration import *

if __name__ == "__main__":
    # Конфигурация
    TICKER = "SPY"
    START_DATE = "2016-01-01"
    END_DATE = "2017-06-01"
    # Подготовка данных
    params, data = get_option_parameters(TICKER, START_DATE, END_DATE)
    # Извлечение параметров
    S_t = params['S_t']
    K = params['K']
    tau = params['tau']
    r = params['r']
    sigma = params['sigma']

    print("Параметры модели:")
    print(f"   • Актив: {TICKER}")
    print(f"   • Текущая цена (S_t): {S_t:.2f}$")
    print(f"   • Цена исполнения (K): {K:.2f}$")
    print(f"   • Время до экспирации (τ): {tau:.3f} лет")
    print(f"   • Безрисковая ставка (r): {r:.4f} ({r*100:.2f}%)")
    print(f"   • Волатильность (σ): {sigma:.4f} ({sigma*100:.2f}%)")

    print("\n" + "="*60)
    print("МОДЕЛЬ БЛЭКА-ШОУЛЗА")
    print("="*60)

    # 1. Вычисляем цену опциона колл по Блэку-Шоулзу
    call_price = black_scholes(S_t, K, tau, r, sigma)
    print(f"\nЦена опциона CALL: {call_price:.2f}$")

    # 2. построение трехмерного графика C(Si,ti) по Блэку-Шоулзу
    plot_black_scholes_surface(data, K, r, sigma, S_t, tau)

    # 3. Дополнительная визуализация: исторические цены и волатильность
    plot_price_volatility(data, TICKER)

    print("\n" + "="*60)
    print("КЛАССИЧЕСКИЙ ПОДХОД ДЮПИРА")
    print("="*60)
    
    # 1. Классический подход: строим поверхность цен C(K,T) через Блэка-Шоулза
    K_grid, T_grid, C_grid = build_dupire_surface(S_t, r, sigma)

    # 2. Вычисляем локальную волатильность по формуле Дюпира
    local_vol_surface = calculate_dupire_volatility(K_grid, T_grid, C_grid, r)

    # 3. Визуализируем результат
    print("\n--- Формула Дюпира (классический подход) ---")
    print(f"Локальная волатильность (средняя): {np.nanmean(local_vol_surface):.4f}")

    # Построение поверхности Дюпира
    plot_dupire_surface(K_grid, T_grid, local_vol_surface)

    print("\n" + "="*60)
    print("МЕТОД КРАНКА-НИКОЛСОНА ДЛЯ УРАВНЕНИЯ ДЮПИРА")
    print("="*60)
    
    # 1. Решаем уравнение Дюпира численно методом Кранка-Николсона
    K_grid_cn, T_grid_cn, C_grid_cn, K_array, T_array = build_dupire_surface_cn(data, S_t, r, sigma, tau)
    
    # 2. Вычисляем локальную волатильность из численного решения
    local_vol_surface_cn = calculate_dupire_volatility_improved(K_grid_cn, T_grid_cn, C_grid_cn, r)
    
    # 3. Визуализируем решение Дюпира методом Кранка-Николсона
    plot_dupire_cn_solution(K_grid_cn[0, :], T_grid_cn[:, 0], C_grid_cn, local_vol_surface_cn)
    
    # 4. Сравнение с Блэка-Шоулз
    compare_with_black_scholes(S_t, K, tau, r, sigma, C_grid_cn, K_array, T_array)
    
    # 5. Визуализация локальной волатильности из метода Кранка-Николсона
    print("\n--- Визуализация поверхности локальной волатильности (Кранк-Николсон) ---")
    plot_dupire_surface(K_grid_cn, T_grid_cn, local_vol_surface_cn)
    
    print("\n--- Анализ решения Дюпира методом Кранка-Николсона ---")
    print(f"Диапазон страйков: {K_grid_cn[0, 0]:.1f} - {K_grid_cn[0, -1]:.1f}")
    print(f"Диапазон времени: {T_grid_cn[0, 0]:.2f} - {T_grid_cn[-1, 0]:.2f} лет")
    print(f"Средняя локальная волатильность: {np.nanmean(local_vol_surface_cn):.4f}")

    # Демонстрация обратной задачи (калибровка локальной волатильности)
    print("\n" + "="*60)
    print("ОБРАТНАЯ ЗАДАЧА")
    print("="*60)
    # 1. Получаем реальные цены опционов на конечную дату
    evaluation_date = '2025-12-08'

    option_data, current_price = get_option_prices_simple(
        ticker=TICKER,
        evaluation_date=evaluation_date,
        max_expirations=3
    )

    if option_data is  None:
        exit()

    # 3. Подготавливаем данные для калибровки
    market_prices, K_full, T_full = prepare_market_data_for_calibration(option_data, current_price)

    print(f"Матрица цен: {market_prices.shape}")
    print(f"Страйки: {len(K_full)} точек от {K_full.min():.1f} до {K_full.max():.1f}")
    print(f"Времена: {len(T_full)} точек от {T_full.min():.3f} до {T_full.max():.3f} лет")

    # 4. Создаем узловые сетки (разреженные)
    K_nodes = np.linspace(K_full.min(), K_full.max(), 10)  # 10 узлов по страйкам
    T_nodes = np.linspace(T_full.min(), T_full.max(), 8)   # 8 узлов по времени

    # 5. Запускаем калибровку (нужно создать функцию без sigma_true)
    from inverse_calibration import calibrate_volatility_surface

    sigma_calibrated, res = calibrate_volatility_surface(
        market_prices=market_prices,
        K_full=K_full,
        T_full=T_full,
        S0=current_price,
        r=r,
        K_nodes=K_nodes,
        T_nodes=T_nodes,
        sigma_init=sigma,  # используем историческую волатильность как начальное предположение
        lam=2e-3,
        maxiter=50,
        plot_results=True,
        verbose=True
    )
