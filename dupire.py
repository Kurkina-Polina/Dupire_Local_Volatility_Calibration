from black_scholes import *
def build_dupire_surface(S_t, r, sigma):
    """
     Строит поверхность цен опционов C(K,T) для формулы Дюпира:
     Зависимость цены опциона от страйка и времени до экспирации

     Параметры:
      S_t : float
         Фиксированная текущая цена базового актива
     r : float
         Безрисковая процентная ставка
     sigma : float
         Постоянная волатильность для модели Блэка-Шоулза

     Возвращает:
     tuple (K_grid, T_grid, C_grid)
         K_grid : ndarray
             2D сетка цен исполнения (ось X)
         T_grid : ndarray
             2D сетка времени до экспирации (ось Y)
         C_grid : ndarray
             2D сетка цен опционов CALL (ось Z)

     Особенности:
     - Все расчеты используют модель Блэка-Шоулза с постоянной волатильностью
     - Полученная поверхность используется для расчета локальной волатильности Дюпира
     """

    # Сетка для Дюпира: разные страйки K и времена T
    K_range = np.linspace(S_t * 0.5, S_t * 1.5, 50)  # Страйки от 50% до 150% от спота
    T_range =  np.linspace(0.1, 1.0, 30)
    # Создание координатных сеток для построения поверхности.
    K_grid, T_grid = np.meshgrid(K_range, T_range)
    C_grid = np.zeros_like(K_grid)

    # Цена опциона для каждого (K, T)
    for i in range(len(T_range)):
        for j in range(len(K_range)):
            C_grid[i, j] = black_scholes(S_t, K_range[j], T_range[i], r, sigma)

    return K_grid, T_grid, C_grid

def calculate_dupire_volatility(K_grid, T_grid, C_grid, r):
    """
    Вычисляет локальную волатильность по формуле Дюпира:
    σ_L(K,T) = √[ (∂C/∂T + rK·∂C/∂K) / (½·K²·∂²C/∂K²) ]

    Локальная волатильность показывает "рыночное ожидание" волатильности
      для конкретных страйков и сроков
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
    - Использует численные производные (конечные разности)
    - Избегает граничных точек для устойчивости расчетов
    - Возвращает NaN для некорректных значений (отрицательные знаменатели)
    - Упрощенная версия без учета дивидендов
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
    Строит комплексную визуализацию поверхности локальной волатильности Дюпира:
    - 3D поверхность для общего обзора
    - Контурный график для анализа уровней волатильности
    - Срезы по времени и страйкам для детального изучения
    - Тепловую карту для выявления паттернов
    - Статистический анализ поверхности

    Параметры:
    K_grid : ndarray
        2D сетка цен исполнения [T, K]
    T_grid : ndarray
        2D сетка времени до экспирации [T, K]
    local_vol_surface : ndarray
        2D сетка локальных волатильностей, рассчитанных по формуле Дюпира

    Возвращает:
    None
        Функция создает 5 графиков и выводит статистику

    """
    # Очищаем NaN значения для корректного отображения
    clean_vol = np.nan_to_num(local_vol_surface, nan=0.0)

    # ПЕРВАЯ КАРТИНКА: 3 графика
    fig1 = plt.figure(figsize=(16, 12))

    # 1. 3D поверхность (левый верх)
    ax1 = fig1.add_subplot(2, 2, 1, projection='3d')
    surf = ax1.plot_surface(K_grid, T_grid, clean_vol, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Strike Price (K)')
    ax1.set_ylabel('Time to Expiry (T)')
    ax1.set_zlabel('Local Volatility')
    ax1.set_title('3D Surface: Local Volatility')
    fig1.colorbar(surf, ax=ax1, shrink=0.6, aspect=20)

    # 2. Контурный график (правый верх)
    ax2 = fig1.add_subplot(2, 2, 2)
    contour = ax2.contourf(K_grid, T_grid, clean_vol, levels=20, cmap='viridis')
    ax2.set_xlabel('Strike Price (K)')
    ax2.set_ylabel('Time to Expiry (T)')
    ax2.set_title('Contour Plot')
    plt.colorbar(contour, ax=ax2)

    # 3. Срез по времени (нижний ряд - занимает оба столбца)
    ax3 = fig1.add_subplot(2, 1, 2)  # 2 строки, 1 столбец, позиция 2
    time_indices = [5, 15, 25]
    colors = ['red', 'blue', 'green']

    for idx, color in zip(time_indices, colors):
        if idx < len(T_grid):
            ax3.plot(K_grid[idx, :], clean_vol[idx, :], color=color, linewidth=2,
                     label=f'T = {T_grid[idx, 0]:.2f} years')

    ax3.set_xlabel('Strike Price (K)')
    ax3.set_ylabel('Local Volatility')
    ax3.set_title('Fixed Time Slices')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ВТОРАЯ КАРТИНКА: 2 графика
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 4. Heatmap (левая половина)
    im = ax1.imshow(clean_vol,
                    extent=[K_grid.min(), K_grid.max(), T_grid.min(), T_grid.max()],
                    aspect='auto', cmap='hot', origin='lower')
    ax1.set_xlabel('Strike Price (K)')
    ax1.set_ylabel('Time to Expiry (T)')
    ax1.set_title('Heatmap: Local Volatility')
    plt.colorbar(im, ax=ax1, label='Volatility')

    # 5. Срез по страйкам (правая половина)
    strike_indices = [10, 25, 40]
    colors = ['orange', 'purple', 'brown']

    for idx, color in zip(strike_indices, colors):
        if idx < K_grid.shape[1]:  # Более безопасная проверка
            ax2.plot(T_grid[:, idx], clean_vol[:, idx], color=color, linewidth=2,
                     label=f'K = {K_grid[0, idx]:.1f}')

    ax2.set_xlabel('Time to Expiry (T)')
    ax2.set_ylabel('Local Volatility')
    ax2.set_title('Fixed Strike Slices')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Вывод статистики
    print("\n--- Анализ поверхности локальной волатильности ---")
    print(f"Минимальная локальная волатильность: {np.nanmin(local_vol_surface):.4f}")
    print(f"Максимальная локальная волатильность: {np.nanmax(local_vol_surface):.4f}")
    print(f"Средняя локальная волатильность: {np.nanmean(local_vol_surface):.4f}")
    print(f"Медианная локальная волатильность: {np.nanmedian(local_vol_surface):.4f}")
    print(f"Стандартное отклонение: {np.nanstd(local_vol_surface):.4f}")