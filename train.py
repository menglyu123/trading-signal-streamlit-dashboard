import datetime as dt
from data.import_data import HK_CODE_LIST, download_futu_historical_daily_data
from model.trend_pred_model import TrendPredModel
import time

if __name__ == '__main__':
    # Download training data pool using futu api
    data_path = './data/HK_stocks/daily'
    end = dt.datetime.today().date()
    start = end - dt.timedelta(days=365*5)
    for i, ticker in enumerate(HK_CODE_LIST):
        ticker = 'HK.0'+ticker
        if i%60 == 0:
            tick = time.time()
        print(ticker)
        df = download_futu_historical_daily_data(ticker, start= start, end = end)
        time_cost = time.time()-tick
        if ((i+1)%60==0) & (time_cost<=30):
            time.sleep(31-time_cost)   
        if not df.empty:
            df.to_csv(f'{data_path}/{ticker}.csv', index=False)
        else:
            print(f"fail to download historical data for {ticker}")

    # Train model
    model = TrendPredModel(winlen=120)
    model.train(data_path=data_path)





