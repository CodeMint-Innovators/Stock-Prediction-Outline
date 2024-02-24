from flask import Flask
import pandas as pd
import alpaca_trade_api as tradeapi
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.main import check_and_make_directories
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
)

app = Flask(__name__)

api = tradeapi.REST('YOUR_API_KEY', 'YOUR_API_SECRET', base_url='https://paper-api.alpaca.markets')

def fetch_data(symbols, timeframe, start_date, end_date):
    try:
        data = {}
        for symbol in symbols:
            bars = api.get_barset(symbol, timeframe, start=start_date, end=end_date).df
            data[symbol] = bars['close']
        return pd.DataFrame(data)
    except Exception as e:
        print("Error fetching data:", e)
        return None

def rebalance_strategy(data):
    weights = {}
    
    # Checks SPY price vs. 200-day moving average
    spy_price = data['SPY']
    spy_ma200 = spy_price.rolling(window=200).mean()
    if spy_price.iloc[-1] > spy_ma200.iloc[-1]:
        # SPY is above its 200-day moving average and then proceeds to the next step
        pass
    else:
        # Allocates to a low-volatility asset (implementation depends on data source)
        # Allocates to TLT (iShares 20+ Year Treasury Bond ETF)
        weights['TLT'] = 1.0
        return weights
    
    # Checks TQQQ RSI
    tqqq_rsi = calculate_rsi(data['TQQQ'])
    if tqqq_rsi > 79:
        weights['UVXY'] = 1.0
        return weights
    
    # Checks SPXL RSI
    spxl_rsi = calculate_rsi(data['SPXL'])
    if spxl_rsi > 80:
        weights['UVXY'] = 1.0
    else:
        weights['TQQQ'] = 1.0
    
    # Checks TQQQ RSI again
    if tqqq_rsi < 31:
        weights['TECL'] = 1.0
    
    # Checks SPY RSI
    spy_rsi = calculate_rsi(data['SPY'])
    if spy_rsi < 30:
        weights['UPRO'] = 1.0
    
    # Checks TQQQ price vs. 20-day moving average
    tqqq_price = data['TQQQ']
    tqqq_ma20 = tqqq_price.rolling(window=20).mean()
    if tqqq_price.iloc[-1] < tqqq_ma20.iloc[-1]:
        # TQQQ is below its 20-day moving average
        sqqq_rsi = calculate_rsi(data['SQQQ'])
        tlt_rsi = calculate_rsi(data['TLT'])
        if sqqq_rsi < tlt_rsi:
            weights['SQQQ'] = 1.0
        else:
            weights['TLT'] = 1.0
    else:
        weights['TQQQ'] = 1.0
    
    # Checks SQQQ RSI
    if sqqq_rsi < 31:
        weights['SQQQ'] = 1.0
    
    return weights

def calculate_rsi(prices, window=14):
    # Calculates RSI (Relative Strength Index)
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

@app.route('/')
def run_trading_bot():
    symbols = ['SPY', 'TQQQ', 'SPXL', 'UVXY', 'TECL', 'UPRO', 'SQQQ', 'TLT']
    start_date = '2024-01-01'
    end_date = '2024-02-01'
    
    # Fetches data from Alpaca API
    data = fetch_data(symbols, 'day', start_date, end_date)
    if data is not None:
        weights = rebalance_strategy(data)
        print("Portfolio weights:", weights)

    # Sets up the training and testing dates
    TRAIN_START_DATE = '2010-01-01'
    TRAIN_END_DATE = '2021-10-01'
    TEST_START_DATE = '2021-10-01'
    TEST_END_DATE = '2023-03-01'

    # Fetches data for training and testing from Yahoo Finance
    df = YahooDownloader(start_date=TRAIN_START_DATE, end_date=TEST_END_DATE, ticker_list=DOW_30_TICKER).fetch_data()

    # Create necessary directories if they don't exist
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    
    return 'Trading bot executed successfully.'

if __name__ == '__main__':
    app.run()
