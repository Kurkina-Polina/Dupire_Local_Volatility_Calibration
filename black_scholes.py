from building_charts import *
def run_black_scholes():
    ticker = "SPY"
    data_init = yf.download(ticker, start="2016-01-01", end="2017-06-01")
    S_i = data_init['Close'].values  # Цены актива
    t_j = np.arange(len(S_i))   # Временные точки (в днях)
    # Визуализация исходных данных
    plot_initial_data(t_j, S_i, ticker)

    #Собираем данные для формулы
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