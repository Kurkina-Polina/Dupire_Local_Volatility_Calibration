import yfinance as yf
import numpy as np
from scipy.stats import norm
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

#FIXME: вынести в отдельный файл с формулами
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