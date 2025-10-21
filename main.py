import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
import pandas as pd

def get_option_parameters(ticker, start_date, end_date):
    """
    Получает все необходимые параметры для модели Блэка-Шоулза
    """
    # Загружаем данные
    data = yf.download(ticker, start=start_date, end=end_date)

    # 1. Текущая цена (последняя цена закрытия)
    S_t = float(data['Close'].iloc[-1])

    # 2. Волатильность (историческая, 30-дневная)
    data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Volatility'] = data['Returns'].rolling(window=30).std() * np.sqrt(252)
    sigma = float(data['Volatility'].iloc[-1])

    # 3. Безрисковая ставка (из казначейских векселей)
    try:
        tbill = yf.Ticker("^IRX")
        tbill_data = tbill.history(period="1d")
        r = float(tbill_data['Close'].iloc[-1] / 100)
    except:
        r = 0.02  # Fallback значение 2%
        print("Используется fallback ставка 2%")

    # 4. Время (предположим, что мы оцениваем опцион на 3 месяца вперед)
    T_minus_t = 90 / 365.0  # 90 дней в долях года

    # 5. Цена исполнения (можно взять как текущую цену для ATM опциона)
    K = S_t  # At-the-money

    parameters = {
        'S_t': S_t,
        'K': K,
        'T_minus_t': T_minus_t,
        'r': r,
        'sigma': sigma
    }

    return parameters, data

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

if __name__ == "__main__":

    ticker = "SPY"
    # Следующие сроки до вызова get_option_parameters чисто для визуализации исходных данных
    data_init = yf.download(ticker, start="2016-01-01", end="2017-06-01")
    S_i = data_init['Close'].values  # Цены актива
    t_j = np.arange(len(S_i))   # Временные точки (в днях)
    # Визуализация исходных данных
    plt.figure(figsize=(12, 6))
    plt.plot(t_j, S_i)
    plt.title(f'Цены актива {ticker}')
    plt.xlabel('Время (дни)')
    plt.ylabel('Цена ($)')
    plt.grid(True)
    plt.show()

    #Собираем данные для формулы
    params, data = get_option_parameters(ticker, "2016-01-01", "2017-06-01")

    # Извлекаем параметры
    S_t = params['S_t']
    K = params['K']
    T_minus_t = params['T_minus_t']
    r = params['r']
    sigma = params['sigma']

    print("Рассчитанные параметры для модели Блэка-Шоулза:")
    print(f"Текущая цена актива (S_t): {S_t:.2f}$")
    print(f"Цена исполнения (K): {K:.2f}$")
    print(f"Время до экспирации (T): {T_minus_t:.3f} лет")
    print(f"Безрисковая ставка (r): {r:.4f} ({r*100:.2f}%)")
    print(f"Волатильность (σ): {sigma:.4f} ({sigma*100:.2f}%)")

    # Вычисляем цену опциона колл по Блэку-Шоулзу
    call_price = black_scholes(S_t, K, T_minus_t, r, sigma, option_type='call')

    print(f"\nЦена опциона CALL: {call_price:.2f}$")

    #FIXME: building chart move to separated func

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
    current_call_price = black_scholes(S_t, K, T_minus_t, r, sigma, option_type='call')
    ax.scatter(S_t, T_minus_t, current_call_price, color='red', s=100,
               label=f'Текущая точка: S={S_t:.1f}, T={T_minus_t:.2f}')
    ax.legend()

    plt.tight_layout()
    plt.show()

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

