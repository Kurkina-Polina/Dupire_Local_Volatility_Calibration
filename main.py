from black_scholes import run_black_scholes
from building_charts import *

def build_dupire_surface(data, S_t, r, sigma, tau):
    # Сетка для Дюпира: разные страйки K и времена T
    K_range = np.linspace(S_t * 0.5, S_t * 1.5, 50)  # Страйки от 50% до 150% от спота
    T_range = np.linspace(0.1, 1.0, 30)

    K_grid, T_grid = np.meshgrid(K_range, T_range)
    C_grid = np.zeros_like(K_grid)

    # Цена опциона для каждого (K, T)
    for i in range(len(T_range)):
        for j in range(len(K_range)):
            C_grid[i, j] = black_scholes(S_t, K_range[j], T_range[i], r, sigma, 'call')

    return K_grid, T_grid, C_grid

def calculate_dupire_volatility(K_grid, T_grid, C_grid, r, d=0):
    """
    Вычисляет локальную волатильность по формуле Дюпира
    """
    local_vol_grid = np.zeros_like(C_grid)

    for i in range(1, len(T_grid)-1):  # Избегаем границы по времени
        for j in range(1, len(K_grid[0])-1):  # Избегаем границы по страйкам

            # Численные производные
            dC_dT = (C_grid[i+1, j] - C_grid[i-1, j]) / (T_grid[i+1, j] - T_grid[i-1, j])
            dC_dK = (C_grid[i, j+1] - C_grid[i, j-1]) / (K_grid[i, j+1] - K_grid[i, j-1])
            d2C_dK2 = (C_grid[i, j+1] - 2*C_grid[i, j] + C_grid[i, j-1]) / ((K_grid[i, j+1] - K_grid[i, j])**2)

            # Формула Дюпира (упрощенная, без дивидендов)
            numerator = 2 * (dC_dT + r * K_grid[i,j] * dC_dK)
            denominator = (K_grid[i,j]**2) * d2C_dK2

            # Проверка на корректность
            if denominator > 0 and numerator > 0:
                local_vol_grid[i,j] = np.sqrt(numerator / denominator)
            else:
                local_vol_grid[i,j] = np.nan

    return local_vol_grid

def plot_dupire_surface(K_grid, T_grid, local_vol_surface):
    """
    Визуализация поверхности локальной волатильности по формуле Дюпира
    """
    # Очищаем NaN значения для корректного отображения
    clean_vol = np.nan_to_num(local_vol_surface, nan=0.0)

    # 1. 3D поверхность
    fig1 = plt.figure(figsize=(12, 8))
    ax1 = fig1.add_subplot(111, projection='3d')

    surf = ax1.plot_surface(K_grid, T_grid, clean_vol,
                            cmap='viridis', alpha=0.8,
                            linewidth=0, antialiased=True)

    ax1.set_xlabel('Strike Price (K)')
    ax1.set_ylabel('Time to Expiry (T)')
    ax1.set_zlabel('Local Volatility')
    ax1.set_title('3D Surface: Local Volatility by Dupire Formula')
    fig1.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()

    # 2. Контурный график
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111)
    contour = ax2.contourf(K_grid, T_grid, clean_vol, levels=20, cmap='viridis')
    ax2.set_xlabel('Strike Price (K)')
    ax2.set_ylabel('Time to Expiry (T)')
    ax2.set_title('Contour Plot: Local Volatility')
    plt.colorbar(contour, ax=ax2)
    plt.tight_layout()
    plt.show()

    # 3. Срез по времени (фиксированное T)
    fig3 = plt.figure(figsize=(10, 6))
    ax3 = fig3.add_subplot(111)
    time_indices = [5, 15, 25]  # Индексы разных временных точек
    colors = ['red', 'blue', 'green']

    for idx, color in zip(time_indices, colors):
        if idx < len(T_grid):
            ax3.plot(K_grid[idx, :], clean_vol[idx, :],
                     color=color, linewidth=2,
                     label=f'T = {T_grid[idx, 0]:.2f} years')

    ax3.set_xlabel('Strike Price (K)')
    ax3.set_ylabel('Local Volatility')
    ax3.set_title('Local Volatility vs Strike (fixed T)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 4. Срез по страйкам (фиксированный K)
    fig4 = plt.figure(figsize=(10, 6))
    ax4 = fig4.add_subplot(111)
    strike_indices = [10, 25, 40]  # Индексы разных страйков
    colors = ['orange', 'purple', 'brown']

    for idx, color in zip(strike_indices, colors):
        if idx < len(K_grid[0]):
            ax4.plot(T_grid[:, idx], clean_vol[:, idx],
                     color=color, linewidth=2,
                     label=f'K = {K_grid[0, idx]:.1f}')

    ax4.set_xlabel('Time to Expiry (T)')
    ax4.set_ylabel('Local Volatility')
    ax4.set_title('Local Volatility vs Time (fixed K)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 5. Heatmap (дополнительный график)
    fig5 = plt.figure(figsize=(10, 8))
    ax5 = fig5.add_subplot(111)
    im = ax5.imshow(clean_vol, extent=[K_grid.min(), K_grid.max(), T_grid.min(), T_grid.max()],
                    aspect='auto', cmap='hot', origin='lower')
    ax5.set_xlabel('Strike Price (K)')
    ax5.set_ylabel('Time to Expiry (T)')
    ax5.set_title('Heatmap: Local Volatility')
    plt.colorbar(im, ax=ax5, label='Volatility')
    plt.tight_layout()
    plt.show()

    # Вывод статистики
    print("\n--- Анализ поверхности локальной волатильности ---")
    print(f"Минимальная локальная волатильность: {np.nanmin(local_vol_surface):.4f}")
    print(f"Максимальная локальная волатильность: {np.nanmax(local_vol_surface):.4f}")
    print(f"Средняя локальная волатильность: {np.nanmean(local_vol_surface):.4f}")
    print(f"Медианная локальная волатильность: {np.nanmedian(local_vol_surface):.4f}")
    print(f"Стандартное отклонение: {np.nanstd(local_vol_surface):.4f}")

if __name__ == "__main__":
    # Вычисление значения стоимости опционов по формуле Блэка-Шоулза и построение трехмерного графика C(Si,ti)
    ticker = "SPY"
    data_init = yf.download(ticker, start="2016-01-01", end="2017-06-01")
    S_i = data_init['Close'].values  # Цены актива
    t_j = np.arange(len(S_i))   # Временные точки (в днях)
    # Визуализация исходных данных
    plot_initial_data(t_j, S_i, ticker)

    #Собираем данные для формулыз
    params, data = get_option_parameters(ticker, "2016-01-01", "2017-06-01")

    # Извлекаем параметры
    S_t = params['S_t']
    K = params['K']
    tau = params['tau']
    r = params['r']
    sigma = params['sigma']

    print("Рассчитанные параметры для модели Блэка-Шоулза:")
    print(f"Текущая цена актива (S_t): {S_t:.2f}$")
    print(f"Цена исполнения (K): {K:.2f}$")
    print(f"Время до экспирации (T): {tau:.3f} лет")
    print(f"Безрисковая ставка (r): {r:.4f} ({r*100:.2f}%)")
    print(f"Волатильность (σ): {sigma:.4f} ({sigma*100:.2f}%)")

    # Вычисляем цену опциона колл по Блэку-Шоулзу
    call_price = black_scholes(S_t, K, tau, r, sigma, option_type='call')
    print(f"\nЦена опциона CALL: {call_price:.2f}$")

    # Трехмерный график C(Si,ti)
    build_3Dchart(data, K, r, sigma, S_t, tau)
    # Дополнительная визуализация: исторические цены и волатильность
    plot_price_volatility(data, ticker)

    # 1. Строим поверхность цен C(K,T)
    K_grid, T_grid, C_grid = build_dupire_surface(data, S_t, r, sigma, tau)

    # 2. Вычисляем локальную волатильность по формуле Дюпира
    local_vol_surface = calculate_dupire_volatility(K_grid, T_grid, C_grid, r)

    # 3. Визуализируем результат
    print("\n--- Формула Дюпира ---")
    print(f"Локальная волатильность (средняя): {np.nanmean(local_vol_surface):.4f}")

    # Можно добавить график поверхности локальной волатильности
    plot_dupire_surface(K_grid, T_grid, local_vol_surface)