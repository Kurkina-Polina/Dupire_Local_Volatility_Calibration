import yfinance as yf
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def get_option_parameters(ticker, start_date, end_date):
    """
    Получает все необходимые параметры для модели Блэка-Шоулза
    ticker название комании (тикера)
    start_date дата начала в формате "2016-01-01"
    end_date дата конца в формате "2016-01-01" минимум 30 дней разницы
    """
    # Загружаем данные
    data = yf.download(ticker, start=start_date, end=end_date)

    # 1. Время (предположим, что мы оцениваем опцион на 3 месяца вперед)
    tau = 90 / 365.0  # 90 дней в долях года

    # 2. Текущая цена (последняя цена закрытия)
    S_t = float(data['Close'].iloc[-1])

    # 3. Сигма - волатильность (историческая, 30-дневная)
    #сигма is the standard deviation of the stock's returns. This is the square root of the quadratic
    # variation of the stock's log price process, a measure of its volatility.

    # Логарифмическая доходность на основе цен закрытия shift(1) сдвигает данные на одну строку назад
    data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    # rolling(window=30) создает скользящее окно из 30 дней; std() высчитывает стандартное отклонение
    # * np.sqrt(252) чтобы перевести дневную волатильность в годовую (252 торговых дня в году)
    data['Volatility'] = data['Returns'].rolling(window=30).std() * np.sqrt(252)
    # Последняя рассчитанная волатильность
    sigma = float(data['Volatility'].iloc[-1])

    # 4. r - безрисковая ставка (из казначейских векселей) 13 WEEK TREASURY BILL (^IRX)
    # векселя — это, как правило, краткосрочные, внебиржевые долговые бумаги
    try:
        tbill = yf.Ticker("^IRX")
        tbill_data = tbill.history(period="1d")
        r = float(tbill_data['Close'].iloc[-1] / 100)
    except:
        r = 0.02  # Fallback значение 2%
        print("Используется fallback ставка 2%")

    # 5. Цена исполнения (можно взять как текущую цену для ATM опциона)
    K = S_t  # At-the-money

    parameters = {
        'S_t': S_t,
        'K': K,
        'tau': tau,
        'r': r,
        'sigma': sigma
    }

    return parameters, data

def black_scholes(S, K, tau, r, sigma, option_type='call'):
    """
    S: текущая цена актива
    K: страйк цена
    tau: время до экспирации (в годах)
    r: безрисковая ставка
    sigma: волатильность
    """
    if tau <= 0: # если опцион уже истек то модель не работает
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price

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
