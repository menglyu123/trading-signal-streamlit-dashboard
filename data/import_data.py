import pandas as pd
import datetime as dt, time
import yfinance as yf
import mysql.connector
import sqlalchemy
from sqlalchemy import text
from futu import *
import schedule

with open('./data/code_pool_hk.txt','r') as fp:
    CODE_LIST = [line.rstrip() for line in fp]



def download_yf_data(code_list, start_date=None, end_date=None):
    """
    Download stock data from Yahoo Finance for a list of stocks
    Args:
        code_list: list of stock codes
        start_date: start date (datetime.date or string YYYY-MM-DD)
        end_date: end date (datetime.date or string YYYY-MM-DD), defaults to today if None
    Returns:
        data: DataFrame with stock data from yfinance
    """
    from curl_cffi import requests
    session = requests.Session(impersonate="chrome")
    # Handle date parameters
    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str):
        end_date = dt.datetime.strptime(end_date, '%Y-%m-%d').date()
    if end_date is None:
        end_date = dt.date.today()
        
    # Batch download data using yfinance
    try:
        data = yf.download(
            tickers=code_list,
            start=start_date,
            end=end_date + dt.timedelta(days=1),
            group_by='ticker',
            progress=False,
            session=session
        )
        return data
    except Exception as e:
        print(f"Error downloading batch data: {e}")
        return None
  
def download_today_capital_flow(tickers):
    data = []
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    for ticker in tickers:
        ret, tmp = quote_ctx.get_capital_flow('HK.0'+ticker, period_type = PeriodType.INTRADAY) # DAY, WEEK, MONTH
        if ret == RET_OK:
            tmp = tmp[['capital_flow_item_time','in_flow','super_in_flow','big_in_flow','mid_in_flow','sml_in_flow']]
            tmp.rename(columns={'capital_flow_item_time':'time'}, inplace=True)
            tmp['time'] = pd.to_datetime(tmp.time)
            data.append(tmp)   
            time.sleep(1)  # Interface limitations: A maximum of 30 requests per 30 seconds
        else:
            print("error: ",tmp)
    quote_ctx.close() # After using the connection, remember to close it to prevent the number of connections from running out
    return data

def download_price(tickers, start, end, interval):
    data = []
    if interval == '1d':
        for ticker in tickers:
            try:
                data.append(yf.download(ticker+'.HK', start=start, end=end, interval='1d').reset_index())
            except Exception:
                print(Exception)
    if interval == '1h':
        for ticker in tickers:
            try:
                df = yf.download(ticker+'.HK', start=start, end=end, interval='1h').reset_index()
                df['Datetime'] = pd.to_datetime(df.Datetime).dt.tz_localize(None)
            except Exception:
                print(Exception)
                continue
            data.append(df)
    return data



class DB_ops():
    def __init__(self, host, user, password):
        self.host = host
        self.user = user
        self.password = password
        self.tickers = CODE_LIST

    def build_mysqlconn(self, database=None):
        if not database:
            self.conn = mysql.connector.connect(host = self.host, user = self.user, password=self.password)
        else:
            self.conn = mysql.connector.connect(host = self.host, user = self.user, password=self.password, database= database)
       
    def close_mysqlconn(self):
        self.conn.close()

    def createdb(self, name):
        self.build_mysqlconn()
        cursor = self.conn.cursor()
        cursor.execute("CREATE database "+ name)
        self.close_mysqlconn()

    def build_engine(self, database):
        self.engine = sqlalchemy.create_engine(f"mysql+pymysql://{self.user}:{self.password}@{self.host}:3306/{database}")

    def import_historical_price(self, interval='1d'):
        if interval == '1d':
            database = 'HK_stocks_daily'
            key = 'Date'
            start = None
            end = None
        if interval == '1h':
            database = 'HK_stocks_hourly'
            key = 'Datetime'
            end = dt.datetime.now()
            start = end- dt.timedelta(days=729)  # maximum 730 past records are available
        df_list = download_price(self.tickers, start=start, end=end, interval=interval)
        if df_list != []:
            self.build_engine(database)
            conn = self.engine.connect()
            for frame, symbol in zip(df_list, self.tickers):
                frame.to_sql(con= self.engine, name = symbol, index=False, if_exists='replace')
                if symbol.isnumeric():
                    symbol_fm = "`"+symbol+"`"
                    conn.execute(text("ALTER TABLE "+ symbol_fm +" ADD PRIMARY KEY ("+ key +")"))               
            print('Successfully imported price')


    def import_capital_flow(self, if_exists='replace'):
        database = 'HK_stocks_capital_flow'
        key = 'time'
        df_list = download_today_capital_flow(self.tickers)
        if df_list != []:
            self.build_engine(database)
            conn = self.engine.connect()
            for frame, symbol in zip(df_list, self.tickers):
                try:
                    frame.to_sql(con= self.engine, name = symbol, if_exists=if_exists, index=False)
                except Exception as e:
                    print(e)
                    continue
                if if_exists=='replace':
                    if symbol.isnumeric():
                        symbol_fm = "`"+symbol+"`"
                        conn.execute(text("ALTER TABLE "+ symbol_fm +" ADD PRIMARY KEY ("+ key +")"))     
            print('Successfully insert todays captital_flow')
            conn.close()
      
    

    def update_today_price(self, interval):
        if interval == '1d':
            database = 'HK_stocks_daily'
            start = dt.date.today()
            df_list = download_price(self.tickers, start= start, end= start+dt.timedelta(days=1), interval='1d')
            if df_list != []:
                self.build_mysqlconn(database)
                cursor = self.conn.cursor()
                for frame, symbol in zip(df_list, self.tickers):
                    if symbol.isnumeric():
                        symbol = "`"+symbol+"`"
                    if len(frame)==1:
                        sql_stmt = "INSERT INTO " + symbol+ "(Date, Open, High, Low, Close, `Adj Close`, Volume) VALUES (%s,%s,%s,%s,%s,%s,%s) \
                            ON DUPLICATE KEY UPDATE\
                                Open = VALUES (Open),\
                                High = VALUES (High),\
                                Low = VALUES(Low),\
                                Close = VALUES(Close),\
                                `Adj Close` = VALUES(`Adj Close`),\
                                Volume = VALUES(Volume)"
                        data = (frame.iloc[0]['Date'],frame.iloc[0]['Open'],frame.iloc[0]['High'],frame.iloc[0]['Low'],frame.iloc[0]['Close'],frame.iloc[0]['Adj Close'],int(frame.iloc[0]['Volume'])) 
                        cursor.execute(sql_stmt, data)  
                        self.conn.commit()    #commit the changes to database      
                print('Successfully insert todays daily price')
                self.close_mysqlconn()
            else:
                print('None of todays data is found')
        if interval == '1h':
            database = 'HK_stocks_hourly'
            start = dt.date.today()
            df_list = download_price(self.tickers, start= start, end= start+dt.timedelta(days=1), interval='1h')
            if df_list != []:
                self.build_engine(database)
                for frame, symbol in zip(df_list, self.tickers):   
                    frame.to_sql(con= self.engine, name = symbol, if_exists='append', index=False)
                print('Successfully insert todays hourly price')
    


    def update_today_capital_flow(self):
        self.import_capital_flow(if_exists='append')



    def fetch_batch_price_from_db(self, ticker, interval, start=None, end=None, limit=None):
        if interval == '1d':
            database = 'HK_stocks_daily'
            key = 'Date'
        if interval == '1h':
            database = 'HK_stocks_hourly'
            key = 'Datetime'
        self.build_mysqlconn(database)
        cursor = self.conn.cursor()
        if ticker.isnumeric():
            ticker = "`"+ticker+"`"
        if end == None:
            end = str(dt.datetime.now())
        end_fm = "'"+end +"'"
        if start != None:
            start_fm = "'"+start +"'"
            cursor.execute(f"SELECT * FROM {ticker} WHERE {key} BETWEEN {start_fm} AND {end_fm}")
        elif limit != None:
            end_fm = "'"+end +"'"
            cursor.execute(f"SELECT * FROM {ticker} WHERE {key} <= {end_fm} ORDER BY {key} DESC LIMIT {limit}")
        else:
            cursor.execute(f"SELECT * FROM {ticker} WHERE {key} <= {end_fm}")
        rows = cursor.fetchall()
        if (start == None) & (limit != None):
            rows = rows[::-1]
        self.close_mysqlconn()
        # if rows[-1][0] != datetime.datetime.strptime(date, '%Y-%m-%d'):
        #     return pd.DataFrame()
        # else:
        if interval=='1d':
            df = pd.DataFrame(rows, columns =['date','open','high','low','close','adj_close','volume'])
        if interval == '1h':
            df = pd.DataFrame(rows, columns =['datetime','open','high','low','close','adj_close','volume'])
        df.drop(columns='adj_close',inplace=True)
        return df
    
    
    def update_price_to_db(self):
        self.update_today_price(interval='1d')

    def auto_update(self):
        self.update_today_price(interval='1d')
        self.update_today_capital_flow()
        


##### CREATE database #####
if __name__ == '__main__':  
    mydb = DB_ops('localhost','root','mlu123456')
    schedule.every().day.at("16:20").do(mydb.auto_update)
    while True:
        schedule.run_pending()
        time.sleep(1)
    #mydb.update_today_price(interval='1d')
    # df = mydb.fetch_batch_price_from_db('0700','1d',limit=10)
    # print(df)


  