from  additional_features import *
import numpy as np
import matplotlib.pyplot as plt

def build_3Dchart(data, K, r, sigma, S_t, tau):
    # Подготавливаем данные для 3D графика
    S_i = data['Close'].values  # Исторические цены
    t_j = np.arange(len(S_i))   # Временные точки
    T_max = 1.0  # Максимальное время для графика (1 год)

    # Создаем сетку для 3D графика
    S_range = np.linspace(S_i.min() * 0.8, S_i.max() * 1.2, 50)
    T_range = np.linspace(0.1, T_max, 30)  # От 0.1 до избежать деления на 0

    S_grid, T_grid = np.meshgrid(S_range, T_range)
    C_grid = np.zeros_like(S_grid)

    # Вычисляем цены опционов для каждой точки сетки
    for i in range(len(T_range)):
        for j in range(len(S_range)):
            C_grid[i, j] = black_scholes(S_range[j], K, T_range[i], r, sigma, option_type='call')

    # Создаем 3D график
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(S_grid, T_grid, C_grid, cmap='viridis',
                           alpha=0.8, linewidth=0, antialiased=True)

    ax.set_xlabel('Цена актива S ($)')
    ax.set_ylabel('Время до экспирации T (годы)')
    ax.set_zlabel('Цена опциона C ($)')
    ax.set_title(f'Поверхность цен опционов CALL по Блэку-Шоулзу\n'
                 f'K={K:.1f}, r={r:.3f}, σ={sigma:.3f}')

    # Добавляем цветовую шкалу
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Цена опциона ($)')

    # Добавляем точку для текущих параметров
    current_call_price = black_scholes(S_t, K, tau, r, sigma, option_type='call')
    ax.scatter(S_t, tau, current_call_price, color='red', s=100,
               label=f'Текущая точка: S={S_t:.1f}, T={tau:.2f}')
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_price_volatility(data, ticker):
    # Дополнительная визуализация: исторические цены и волатильность
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # График цен
    ax1.plot(data.index, data['Close'], linewidth=1)
    ax1.set_title(f'Исторические цены {ticker}')
    ax1.set_ylabel('Цена ($)')
    ax1.grid(True)

    # График волатильности
    ax2.plot(data.index, data['Volatility'], linewidth=1, color='orange')
    ax2.set_title('Историческая волатильность (30-дневная годовая)')
    ax2.set_ylabel('Волатильность')
    ax2.set_xlabel('Дата')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_initial_data(t_j, S_i, ticker):
    # Визуализация исходных данных
    plt.figure(figsize=(12, 6))
    plt.plot(t_j, S_i)
    plt.title(f'Цены актива {ticker}')
    plt.xlabel('Время (дни)')
    plt.ylabel('Цена ($)')
    plt.grid(True)
    plt.show()