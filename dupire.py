from black_scholes import *
from scipy.ndimage import gaussian_filter
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
    K_range = np.linspace(S_t * 0.6, S_t * 1.4, 80)  # уже и плотнее, меньше шум в производных
    T_range =  np.linspace(0.1, 1.0, 60)
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
    # Лёгкое сглаживание поверхности, чтобы производные не взрывались
    C_smooth = gaussian_filter(C_grid, sigma=1.0)
    local_vol_grid = np.full_like(C_grid, np.nan)

    # Избегаем двух крайних слоёв для устойчивых разностей
    for i in range(2, T_grid.shape[0]-2):
        for j in range(2, K_grid.shape[1]-2):
            dT = T_grid[i+1, j] - T_grid[i-1, j]
            dK = K_grid[i, j+1] - K_grid[i, j-1]

            dC_dT = (C_smooth[i+1, j] - C_smooth[i-1, j]) / dT
            dC_dK = (C_smooth[i, j+1] - C_smooth[i, j-1]) / dK
            d2C_dK2 = (C_smooth[i, j+1] - 2*C_smooth[i, j] + C_smooth[i, j-1]) / (dK**2)

            if d2C_dK2 > 1e-8:
                numerator = 2 * (dC_dT + r * K_grid[i, j] * dC_dK)
                denominator = (K_grid[i, j]**2) * d2C_dK2
                if numerator > 0 and denominator > 0:
                    local_vol_grid[i, j] = np.sqrt(numerator / denominator)

    # Ограничиваем выбросы разумным пределом: 3×медианная вола
    finite_vals = local_vol_grid[np.isfinite(local_vol_grid)]
    if finite_vals.size:
        cap = 3.0 * np.nanmedian(finite_vals)
        local_vol_grid = np.clip(local_vol_grid, 0, cap)

    return local_vol_grid

def plot_dupire_surface(K_grid, T_grid, local_vol_surface, name):
    """
    Строит комплексную визуализацию поверхности локальной волатильности:
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
        2D сетка локальных волатильностей, рассчитанных по формуле Дюпира или Крэнка-Никольсона

    Возвращает:
    None
        Функция создает 5 графиков и выводит статистику

    """
    # Очищаем NaN значения для корректного отображения
    clean_vol = np.nan_to_num(local_vol_surface, nan=0.0)

    # ПЕРВАЯ КАРТИНКА: 3 графика
    fig1 = plt.figure(figsize=(16, 12))
    fig1.suptitle(name, fontsize=16)

    # 1. 3D поверхность (левый верх)
    ax1 = fig1.add_subplot(2, 2, 1, projection='3d')
    surf = ax1.plot_surface(K_grid, T_grid, clean_vol, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Цена исполнения (K)')
    ax1.set_ylabel('Время до экспирации (T)')
    ax1.set_zlabel('Локальная Волатильность')
    ax1.set_title('Поверхность локальной волатильности')
    fig1.colorbar(surf, ax=ax1, shrink=0.6, aspect=20)

    # 2. Контурный график (правый верх)
    ax2 = fig1.add_subplot(2, 2, 2)
    contour = ax2.contourf(K_grid, T_grid, clean_vol, levels=20, cmap='viridis')
    ax2.set_xlabel('Цена исполнения (K)')
    ax2.set_ylabel('Время до экспирации (T)')
    ax2.set_title('Контурный график локальной волатильности')
    plt.colorbar(contour, ax=ax2)

    # 3. Срез по времени (нижний ряд - занимает оба столбца)
    ax3 = fig1.add_subplot(2, 1, 2)
    time_indices = [5, 15, 25]
    colors = ['red', 'blue', 'green']

    for idx, color in zip(time_indices, colors):
        if idx < len(T_grid):
            ax3.plot(K_grid[idx, :], clean_vol[idx, :], color=color, linewidth=2,
                     label=f'T = {T_grid[idx, 0]:.2f} years')

    ax3.set_xlabel('Цена исполнения (K)')
    ax3.set_ylabel('Локальная волатильность')
    ax3.set_title('Форма волатильности для разных сроков экспирации')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1.savefig('local_volatility_1_'+name+'.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ВТОРАЯ КАРТИНКА: 2 графика
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle(name, fontsize=16)

    # 4. Heatmap (левая половина)
    im = ax1.imshow(clean_vol,
                    extent=[K_grid.min(), K_grid.max(), T_grid.min(), T_grid.max()],
                    aspect='auto', cmap='hot', origin='lower')
    ax1.set_xlabel('Цена исполнения (K)')
    ax1.set_ylabel('Время до экспирации (T)')
    ax1.set_title('Интенсивность волатильности по страйкам и времени')
    plt.colorbar(im, ax=ax1, label='Volatility')

    # 5. Срез по страйкам (правая половина)
    strike_indices = [10, 25, 40]
    colors = ['orange', 'purple', 'brown']

    for idx, color in zip(strike_indices, colors):
        if idx < K_grid.shape[1]:  # Более безопасная проверка
            ax2.plot(T_grid[:, idx], clean_vol[:, idx], color=color, linewidth=2,
                     label=f'K = {K_grid[0, idx]:.1f}')

    ax2.set_xlabel('Время до экспирации (T)')
    ax2.set_ylabel('Локальная волатильность')
    ax2.set_title('Динамика волатильности для разных страйков')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1.savefig('local_volatility_2_'+name+'.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Вывод статистики
    print("\n--- Анализ поверхности локальной волатильности ", name, " ---")
    print(f"Минимальная локальная волатильность: {np.nanmin(local_vol_surface):.4f}")
    print(f"Максимальная локальная волатильность: {np.nanmax(local_vol_surface):.4f}")
    print(f"Средняя локальная волатильность: {np.nanmean(local_vol_surface):.4f}")
    print(f"Медианная локальная волатильность: {np.nanmedian(local_vol_surface):.4f}")
    print(f"Стандартное отклонение: {np.nanstd(local_vol_surface):.4f}")