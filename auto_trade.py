from util import cal_trade_performance, cal_drawdown, cal_trade_performance
from data.prepare_data import prepare_data
from model.mylenet import lenet_regression
from model.myencoder import encoder_regression
import numpy as np, pandas as pd
import argparse
import statsmodels.api as sm


class auto_trade():
    def __init__(self, model, market='hk'):
        self.market = market
        if self.market == 'hk':
            with open('./data/code_pool_hk.txt','r') as fp:
                self.code_list = [line.rstrip() for line in fp]
        if self.market == 'us':
            with open('./data/code_pool_us.txt','r') as fp:
                self.code_list = [line.rstrip() for line in fp]
        if self.market == 'crypto':
            with open('./data/code_pool_crypto.txt','r') as fp:
                self.code_list = [line.rstrip() for line in fp]
        
        self.winlen = 120
        self.future = 1
        self.fea_num = 16
         
        if model == 'lenet':
            self.model_folder = f'./model/regression/lenet'  
            # build model
            self.model = lenet_regression(shape= (self.winlen, self.fea_num))
        if model == 'encoder':
            self.model_folder = f'./model/regression/encoder'  
            # build model
            self.model = encoder_regression(shape= (self.winlen, self.fea_num))


    def load_model(self):
        self.model_path = self.model_folder+ '/epoch_sel.h5'
        self.model.load(self.model_path)
        return self.model



    def backtest_single(self, df, init_balance=10000, fee=0.001, pct = 1):
        bdf, test_X = prepare_data(df, self.winlen, self.future, training=False)
        if test_X.shape[0] == 0:
            return
        
        # model prediction
        _ = self.load_model()
        pred_val = self.model.predict(test_X)
        bdf['predict'] = pred_val
        
        close = bdf.close.to_numpy()
        pred = bdf.predict.to_numpy()
        
        # y = np.array(bdf.EMA_10)
        # x = np.array(bdf.EMA_20)
        # bdf['curvature'] = calculate_curvature(x, y) 
    
        price_change = [p/close[0] -1 for p in close]
        period_len = bdf.shape[0]
        buy, sell = [0]*period_len, [0]*period_len
        status = 'empty'
        new_status = status
        status_list = ['empty']
        balance_list = []
        balance = init_balance
        in_amt_list = []
        in_amt = 0
        
        rs = bdf.close.rolling(15).apply(lambda x: -np.mean(x[x>0])/(np.mean(x[x<0])+1e-5))
        bdf['close_rsi'] =  100- 100/(1+rs)  #bdf.predict.rolling(5).mean().fillna(0)
        bdf.close_rsi.fillna(method='bfill', inplace=True)


        for i in range(period_len):  
            signal = bdf.iloc[i]['predict']>0.5 #(sum(bdf.iloc[i-3:i+1]['predict']>0)>=2)
            down_deep = (bdf.iloc[i]['EMA_60']/bdf.iloc[i]['close']>1.1)  #the down deeper, the risk lower 
            up_trend= True
            if i >120:
                _, trend = sm.tsa.filters.hpfilter(bdf.iloc[i-120:i+1]['close'], 50)   #close
                trend_50 = pd.Series(trend, name='trend')
                up_trend = (trend_50.iloc[-1]>trend_50.iloc[-2])          
            
            if  (status == 'empty') &signal&down_deep: 
                balance *= (1-fee)
                buy[i] = 1 
                buy_price = bdf.iloc[i]['close']
                new_status = 'hold'
                
                    
            if status == 'hold':
                balance *= close[i]/close[i-1]
                
            #     if (in_amt > start_in_amt*(1+0.05)) & (remain_amt>0):  # add pos
            #         buy_amt = remain_amt
            #         in_amt += buy_amt* (1-fee)
            #         remain_amt -= buy_amt
            #         start_in_amt = in_amt  
            #         add[i] = 1      
            #     if in_amt > max_in_amt:
            #         max_in_amt = in_amt
                
            #     if (in_amt < max(max_in_amt*0.95, start_in_amt*0.98)) & (in_amt>0):  ## max_in_amt-(max_in_amt-start_in_amt)*0.05 reduce pos when price down or take profit |lose_later
            # #    elif (in_amt > start_in_amt*1.05) &(in_amt>0):  # stop gain
            #         sell_amt = 0.9*in_amt
            #         in_amt -= sell_amt
            #         remain_amt += sell_amt*(1-fee)
            #         start_in_amt = in_amt
            #         max_in_amt = in_amt
            #         cut[i] = 1

                if (not signal)|(bdf.iloc[i]['close']<0.98*buy_price): 
                    sell[i] = 1
                    new_status = 'empty'

            status = new_status
            status_list.append(status)
            balance_list.append(balance)
            in_amt_list.append(in_amt)
        
        
        profit = [b- init_balance for b in balance_list]
        advantage = [b/init_balance-p for b,p in zip(profit, price_change)]
        drawdown = cal_drawdown(balance_list)

        metric = cal_trade_performance(profit, buy, sell)  
        eval_df = pd.DataFrame({"profit": profit, "advantage":advantage, "price_change": price_change, "buy": buy, "sell":sell, 
                     "drawdown": drawdown})
        bdf.reset_index(inplace=True, drop=True)
        result_df = pd.concat([bdf, eval_df], axis=1)
        return result_df, metric

        
    def add_pred_cols(self, df):
     #   try:
        bdf, test_X = prepare_data(df, self.winlen, self.future, training=False)
        print("bdf: ", bdf.shape, bdf.head(), test_X.shape)
        _ = self.load_model()
        print("model prediction: ", self.model.predict(test_X))
        bdf['pred'] = self.model.predict(test_X)
        print("pred", bdf.pred)
        bdf['speed'] = np.exp(bdf.pred)
        bdf['speed'] = bdf.speed.pct_change()
        bdf.speed.fillna(0, inplace=True)
        bdf['last5_freq'] = bdf['speed'].rolling(5).apply(lambda x: sum(x>0)/len(x))
        bdf['last10_freq'] = bdf['speed'].rolling(10).apply(lambda x: sum(x>0)/len(x))
        bdf['last15_freq'] = bdf['speed'].rolling(15).apply(lambda x: sum(x>0)/len(x))
        bdf['btm_strength'] = bdf[['last5_freq','last10_freq','last15_freq']].max(axis=1)
        bdf.drop(columns=['last5_freq','last10_freq','last15_freq'], inplace=True)
        # except:
        #     bdf = pd.DataFrame()
        return bdf

    def signal_pool(self, code_list, pred_df_list, date, save=False):
        fetched_code, close_list, pred_list, speed_list, btm_strength_list =[], [], [], [], []
        for code, pred_df in zip(code_list, pred_df_list):
            if len(pred_df)!=0:
                try:
                    fetched_code.append(code)
                    close_list.append(pred_df.iloc[-1]['close'])
                    pred_list.append(pred_df.iloc[-1]['pred'])
                    speed_list.append(pred_df.iloc[-1]['speed'])
                    btm_strength_list.append(pred_df.iloc[-1]['btm_strength'])
                except Exception as e:
                    print(code, e)
                    continue
        rdf = pd.DataFrame({'code':fetched_code, 'close': close_list, 'pred':pred_list, 'speed':speed_list, 'btm_strength':btm_strength_list})
        rdf.set_index('code', inplace=True)
        if save & (len(rdf)!=0):
            rdf.to_csv(f'./results/{date}_{self.market}_signal_pool.csv')
        return rdf
       


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_data', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--realtime_hk', action='store_true')
    parser.add_argument('--realtime_us', action='store_true')
    parser.add_argument('--realtime_crypto', action='store_true')
    args = parser.parse_args()

    if args.download_data:
        auto_trade_today = auto_trade('lenet')
        auto_trade_today.fetch_train_data()
    if args.train:
        auto_trade_today = auto_trade('lenet')
        auto_trade_today.train(lr=1e-4)   
    if args.backtest:
        auto_trade_today = auto_trade('lenet')
        auto_trade_today.backtest(days=500, record=True) #code_list=['0285']
    
        
      