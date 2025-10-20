import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
import pandas as pd

ticker = "SPY"
data = yf.download(ticker, start="2016-01-01", end="2017-06-01")
# Создаем сетку значений
S_i = data['Close'].values  # Цены актива
t_j = np.arange(len(S_i))   # Временные точки (в днях)

# Визуализация исходных данных
# plt.figure(figsize=(12, 6))
# plt.plot(t_j, S_i)
# plt.title(f'Цены актива {ticker}')
# plt.xlabel('Время (дни)')
# plt.ylabel('Цена ($)')
# plt.grid(True)
# plt.show()


def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    S: текущая цена актива
    K: страйк цена
    T: время до экспирации (в годах)
    r: безрисковая ставка
    sigma: волатильность
    """
    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price
# Параметры для примера
K = 150  # Страйк цена
r = 0.05  # Безрисковая ставка 5%
sigma = 0.2  # Волатильность 20%
T_max = 1.0  # Максимальное время до экспирации (1 год)

# Создаем сетку для 3D графика
S_range = np.linspace(S_i.min() * 0.8, S_i.max() * 1.2, 50)
T_range = np.linspace(0.1, T_max, 30)  # От 0.1 до избежать деления на 0

S_grid, T_grid = np.meshgrid(S_range, T_range)
C_grid = np.zeros_like(S_grid)

# Вычисляем цены опционов для каждой точки сетки
for i in range(len(T_range)):
    for j in range(len(S_range)):
        C_grid[i, j] = black_scholes(S_range[j], K, T_range[i], r, sigma)

# Создаем 3D график
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(S_grid, T_grid, C_grid, cmap='viridis',
                       linewidth=0, antialiased=True)

ax.set_xlabel('Цена актива S')
ax.set_ylabel('Время до экспирации T')
ax.set_zlabel('Цена опциона C')
ax.set_title('Поверхность цен опционов по Блэку-Шоулзу')

# Добавляем цветовую шкалу
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

plt.show()