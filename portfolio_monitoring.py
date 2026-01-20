import datetime as dt
from dataclasses import dataclass
import json
from data.import_data import download_futu_historical_daily_data, download_alpaca_daily_data
from futu import *
from auto_trade import get_pred_df
from util import send_imessage
import schedule
import time
from model.trend_pred_model import TrendPredModel

@dataclass
class Component:  
    ticker: str
    entry_date: dt.datetime
    entry_position_size: int
    entry_price: float

class BacktestPortfolio():
    def __init__(self, portfolio_name: str):
        self.portfolio_name = portfolio_name

    def create_backtest_portfolio(self, create_date: dt.datetime, components: list[Component]):
        create_position = 0
        for component in components:
            create_position += component.entry_position_size*component.entry_price

        # Export portfolio to a json file
        with open(f'{self.portfolio_name}.json', 'w') as json_file:
            json.dump({'create_date':create_date, 'components': components, 'entry_position': create_position}, json_file, indent=4)
    
    def update_backtest_portfolio(self, date: dt.datetime):
        pass




    
def get_current_futu_portfolio(): # for real trading
    trd_ctx = OpenSecTradeContext(filter_trdmarket=TrdMarket.HK, host='127.0.0.1', port=11111, security_firm=SecurityFirm.FUTUSECURITIES)
    ret, positions_df = trd_ctx.position_list_query()
    if ret == RET_OK:
        if positions_df.shape[0] > 0:  # If the position list is not empty
            portfolio_comp_df = positions_df[['code','stock_name','position_market','qty','cost_price','nominal_price']]
            portfolio_comp_df.rename(columns = {'nominal_price': 'current_price'}, inplace=True)
    else:
        print('position_list_query error: ', positions_df)
    
    ret, history_order = trd_ctx.history_order_list_query()
    if ret == RET_OK:
        current_drawdown_list, pred_list, dist_mvg5_list, dist_mvg10_list, dist_mvg20_list, dist_mvg60_list = [],[],[],[],[],[]
        for code in portfolio_comp_df.code:
            name = portfolio_comp_df[portfolio_comp_df.code==code]['stock_name'].values[0]
            print(f'analyze {name}')
            last_trade_time = history_order[(history_order.code==code)&(history_order.order_status=='FILLED_ALL')].iloc[0]['create_time']
            last_trade_date = dt.datetime.strptime(last_trade_time, '%Y-%m-%d %H:%M:%S.%f').date()
            current_date = dt.datetime.today().date()
            start_date = min(current_date-dt.timedelta(days=120+60+90+40+1), last_trade_date)
            if portfolio_comp_df[portfolio_comp_df.code==code]['position_market'].values[0]=='HK':
                df = download_futu_historical_daily_data(code, start_date, current_date)
            if portfolio_comp_df[portfolio_comp_df.code==code]['position_market'].values[0]=='US':
                df = download_alpaca_daily_data(code.split('.')[-1], start_date, current_date)
            hold_highest_price = df[(df.date>=last_trade_date)&(df.date<=current_date)].close.max()
            current_drawdown = max(1-portfolio_comp_df[portfolio_comp_df.code==code]['current_price'].values[0]/hold_highest_price,0)
            current_drawdown_list.append(current_drawdown)

            pred_df = get_pred_df(df, 120)
            pred_list.append(pred_df.iloc[-1]['prediction'])
            dist_mvg5_list.append(abs(pred_df.iloc[-1]['close']/pred_df.iloc[-1]['EMA_5']-1))
            dist_mvg10_list.append(abs(pred_df.iloc[-1]['close']/pred_df.iloc[-1]['EMA_10']-1))
            dist_mvg20_list.append(abs(pred_df.iloc[-1]['close']/pred_df.iloc[-1]['EMA_20']-1))
            dist_mvg60_list.append(abs(pred_df.iloc[-1]['close']/pred_df.iloc[-1]['EMA_60']-1))
            
        portfolio_comp_df['current_drawdown'] = current_drawdown_list
        portfolio_comp_df['pred'] = pred_list
        portfolio_comp_df['dist_mvg5'] = dist_mvg5_list
        portfolio_comp_df['dist_mvg10'] = dist_mvg10_list
        portfolio_comp_df['dist_mvg20'] = dist_mvg20_list
        portfolio_comp_df['dist_mvg60'] = dist_mvg60_list
        print(portfolio_comp_df[['current_drawdown','pred','dist_mvg5','dist_mvg10','dist_mvg20','dist_mvg60']].round(2))
    else:
        print('history_order_list_query error: ', history_order)
    trd_ctx.close()
    return portfolio_comp_df


def send_portfolio_alert():
    now = dt.datetime.now().time()
    morning_start_time = dt.time(9, 30)
    morning_end_time = dt.time(12, 0)
    afternoon_start_time = dt.time(13,0)
    afternoon_end_time = dt.time(16,0)


    if (morning_start_time <= now <= morning_end_time) | (afternoon_start_time <= now <= afternoon_end_time):
        trd_ctx = OpenSecTradeContext(filter_trdmarket=TrdMarket.HK, host='127.0.0.1', port=11111, security_firm=SecurityFirm.FUTUSECURITIES)
        ret, positions_df = trd_ctx.position_list_query()
        if ret == RET_OK:
            if positions_df.shape[0] > 0:  # If the position list is not empty
                portfolio_comp_df = positions_df[['code','stock_name','position_market','qty','cost_price','nominal_price']]
                portfolio_comp_df.rename(columns = {'nominal_price': 'current_price'}, inplace=True)
        else:
            print('position_list_query error: ', positions_df)
        
        ret, history_order = trd_ctx.history_order_list_query()
        if ret == RET_OK:
            msg = ''
            for code in portfolio_comp_df.code:
                name = portfolio_comp_df[portfolio_comp_df.code==code]['stock_name'].values[0]
                last_trade_time = history_order[(history_order.code==code)&(history_order.order_status=='FILLED_ALL')].iloc[0]['create_time']
                last_trade_date = dt.datetime.strptime(last_trade_time, '%Y-%m-%d %H:%M:%S.%f').date()
                current_date = dt.datetime.today().date()
                if portfolio_comp_df[portfolio_comp_df.code==code]['position_market'].values[0]=='HK':
                    df = download_futu_historical_daily_data(code, last_trade_date, current_date)
                if portfolio_comp_df[portfolio_comp_df.code==code]['position_market'].values[0]=='US':
                    df = download_alpaca_daily_data(code.split('.')[-1], last_trade_date, current_date)
                hold_highest_price = df[(df.date>=last_trade_date)&(df.date<=current_date)].close.max()
                current_drawdown = max(1-portfolio_comp_df[portfolio_comp_df.code==code]['current_price'].values[0]/hold_highest_price,0)
                if current_drawdown >=0.02:
                    # calculate current prediction
                    start_date = current_date-dt.timedelta(days=120+60+90+40+1)
                    if portfolio_comp_df[portfolio_comp_df.code==code]['position_market'].values[0]=='HK':
                        df = download_futu_historical_daily_data(code, start_date, current_date)
                    if portfolio_comp_df[portfolio_comp_df.code==code]['position_market'].values[0]=='US':
                        df = download_alpaca_daily_data(code.split('.')[-1], start_date, current_date)
                    model = TrendPredModel(120)
                    bdf = model.predict(df)
                    msg += f"{str(dt.datetime.now().replace(microsecond=0))}-{code}: {name} drop {round(current_drawdown, 2)*100}% with current prediction {bdf.iloc[-1]['prediction'].astype(float).round(2)}\n"
            # send alert message
            send_imessage('85269926347', msg)       
        trd_ctx.close()
    return



def send_portfolio_pred(): # for real trading
    ## current_drawdown >0.03 or pred <0
    df = get_current_futu_portfolio()
    msg = f'Date: {dt.datetime.today().date()}\n'
    for _, row in df.iterrows():
        record = row.to_dict()
        mvg_names = ['dist_mvg5','dist_mvg10','dist_mvg20','dist_mvg60']
        dist_mvg_values = [record[mvg_names[0]],record[mvg_names[1]],record[mvg_names[2]],record[mvg_names[3]]]
        min_dist_mvg = min(dist_mvg_values)
        min_mvg_name = mvg_names[dist_mvg_values.index(min_dist_mvg)]
        msg += f"{record['code']}: {record['stock_name']}\n\tPrediction: {round(record['pred'],3)}\n\tRecent Drawdown: {round(record['current_drawdown'],3)}\n\tClose to {min_mvg_name[5:].capitalize()}: {round(min_dist_mvg,3)}\n\n"
    send_imessage('85269926347', msg)

if __name__ == '__main__':
    # Schedule send alert
    schedule.every(15).minutes.do(send_portfolio_alert)
    while True:
        schedule.run_pending()
        time.sleep(1)
    
    # # Schedule send predictions for a portfolio
    # schedule.every().day.at("15:50").do(send_portfolio_pred)
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)













