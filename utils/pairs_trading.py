import pandas as pd
import numpy as np
import datetime as dt
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import sqlite3
sns.set(style="white", rc={"figure.figsize":(8, 4)})
plt.style.use('ggplot') # fivethirtyeight, ggplot, dark_background, classic,  


from statistics import mean
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import coint
import math 

conn = sqlite3.connect('./data/pairs_trading.db')

stocks_hist_price_df = pd.read_sql_query("SELECT * FROM stocks_hist_price", conn, parse_dates=['date'])
stocks_hist_price_df.date = pd.to_datetime(stocks_hist_price_df.date, format='ISO8601')

# fetch full historical adj close price
full_hist_close_df = stocks_hist_price_df.pivot_table(values='Adj Close', index='date', columns='ticker')
full_hist_close_df.index = full_hist_close_df.index.date
full_hist_close_df.interpolate(method='linear', axis=1, inplace=True)

cluster_df = pd.read_csv('./data/db_clabel_10y_data.csv')
target_cluster_df = cluster_df[cluster_df.cluster!=-1].reset_index(drop=True)

"Pair selection method"
"select a pair with lowest p-value from each cluster"
def cluster_pair_selection(cluster_df: pd.DataFrame, significance=0.05, 
                    start_day=dt.datetime(2010, 1, 1).date(), end_day=dt.datetime(2019, 12, 31).date()):
    final_stock_pairs = []   
    cluster_labels = cluster_df.cluster.value_counts().index.to_list()
    temp_price_df = full_hist_close_df.loc[start_day: end_day]

    for i in range(len(cluster_labels)):
        # loop over clusters to get best coint pair in each cluster
        tickers = target_cluster_df[target_cluster_df.cluster == cluster_labels[i]]['ticker'].values
        p_values = []
        coint_pairs = []
        for m in range(len(tickers)):
            for n in range(m+1,len(tickers)):
                p_val = coint(temp_price_df[tickers[m]], temp_price_df[tickers[n]])[1]
                p_values.append(p_val)
                coint_pairs.append([tickers[m], tickers[n]]) 
    
        if len(coint_pairs) > 0:
            if np.min(p_values) < significance:
                index = np.where(p_values == np.min(p_values))[0][0]
                final_stock_pairs.append(coint_pairs[index])
    
    return final_stock_pairs

"Calculate benchmark return and plot the relevant performance charts"
def plot_benchmark_ret(conn):
    spy_index_df = pd.read_sql_query("SELECT * FROM SPY_hist_price", conn, parse_dates=['date'])
    spy_index_df.date = pd.to_datetime(spy_index_df.date, format='ISO8601')
    spy_index_df.date = spy_index_df.date.apply(lambda x: x.date())
    temp_df = spy_index_df[(spy_index_df.date > dt.datetime(2020, 1, 1).date()) & (spy_index_df.date < dt.datetime(2021, 1, 1).date())]
    temp_df['cum_ret'] = temp_df['simple_return'].cumsum()
    spy_sharpe = np.nanmean(temp_df['simple_return'])/np.nanstd(temp_df['simple_return'])

    # plot benchmark
    plt.title(f"Benchmark return for SPY tracking S&P 500 index\n", 
                fontweight='bold', fontsize=12)
    plt.plot(temp_df['date'], temp_df['simple_return'] * 100, label='Daily returns')
    plt.plot(temp_df['date'], temp_df['cum_ret'] * 100, linestyle = '--', label='Cummulative returns')
    plt.ylabel("Percentage return (%)")
    plt.figtext(0.65, 0.4, f'Sharpe ratio = {spy_sharpe:.4f}', bbox=dict(facecolor='orange', alpha=0.3), fontsize=10)
    plt.legend(loc='best')
    plt.show()


class Pair:
    P_VALUE_THRESHOLD = 0.05 # significance level
    MIN_MEAN_CROSSES = 12 # minimum number of mean crosses in the signal
    TRADING_BOUND = 1 # STD level which signifies the trading bounds
    EXIT_PROFIT = 0 # level at which to exit and book profits (close the position)
    STOP_LOSS = 2 # STD level which signifies the stoploss bounds
    RETURN_ON_BOTH_LEGS = False # method of computing returns (should both long & short legs be considered)
    WAITING_DAYS = 15 # wait time period for a signal hold its state after crossing a level before which to close the trade 
    TRADING_START_DATE = dt.datetime(2020, 1, 1).date() # backtest period
    TRADING_END_DATE = dt.datetime(2021, 1, 1).date()
    TC = 1 # transaction costs (defined as a percentage of capital invested)
    
    def __init__(self, 
                stockX, 
                stockY, 
                trading_start_date = TRADING_START_DATE,
                trading_end_date = TRADING_END_DATE,
                waiting_days = WAITING_DAYS, 
                trading_bound = TRADING_BOUND, 
                exit_profit = EXIT_PROFIT, 
                stop_loss = STOP_LOSS,
                return_on_both_legs = RETURN_ON_BOTH_LEGS,
                tc = TC):
        # constructor
        self.name = f"{stockX.name}, {stockY.name}"
        self.stockX_trading = stockX[trading_start_date : trading_end_date]
        self.stockY_trading = stockY[trading_start_date : trading_end_date]
        self.waiting_days = waiting_days
        self.trading_start_date = trading_start_date
        self.trading_end_date = trading_end_date
        self.exit_profit = exit_profit
        self.trading_bound = trading_bound
        self.stop_loss = stop_loss
        self.return_on_both_legs = return_on_both_legs
        self.tc = tc
        self.error = False

        try:
            beta = OLS(stockY[trading_start_date : trading_end_date], stockX[trading_start_date : trading_end_date]).fit().params[0]
            self.spread = stockY[trading_start_date : trading_end_date] - beta * stockX[trading_start_date : trading_end_date]
            self.normalized_spread_trading = (self.spread - self.spread.mean()) / self.spread.std()
            self.p_value = coint(stockX, stockY)[1] 
            self.generate_trading_signals()
        except Exception as e:
            print(e)
            print(f"Error encountered with pair [{self.stockX_trading.name}, {self.stockY_trading.name}]")
            self.error = True
            
    def eligible(self, p_value_threshold = P_VALUE_THRESHOLD, min_mean_crosses = MIN_MEAN_CROSSES):
        # Check for the pair eligibility (if its p_val < threshold & if it meets the min mean crosses threshold)
        if self.error:
            return False
        elif self.p_value <= p_value_threshold and self.mean_crosses >= min_mean_crosses:
            return True
        return False
    
    def __level_crosses(self, series, level = 2):
        # Identify & record the changes in the signal movements as and when it crosses
        # either above or below a defined level 
        change = []
        for i, el in enumerate(series):
            if i != 0 and el > level and series[i-1] < level:
                # if signal crosses above the level (going upwards), record it as '1'
                change.append(1)
            elif i != 0 and el < level and series[i-1] > level:
                # if signal crosses below the level (going downwards), record it as '-1'
                change.append(-1)
            else:
                # else, record it as no change, '0'
                change.append(0)
        return change
    
    def __average_holding_period(self):
        # compute the average holding period for the pair,
        # based on their opening and closing positions recorded
        holding_periods = []
        for closed_date in self.closed_positions:
            open_date = list(filter(lambda x: x < closed_date, self.open_positions))[-1]
            holding_periods.append(closed_date - open_date)
        return np.mean(np.array(holding_periods))
    
    def __calculate_returns(self, pos_y, pos_x, i):
        # function to compute the total returns for the pair
        
        # first, calculate PnL for the pair either when a stoploss is recorded, 
        # or when the position is closed with profit booking
        if pos_x[2] == 'l':
            # long on X (meaning short Y - short the spread)
            cost_long = pos_x[0] * pos_x[1] # cost of long = opening price * number of stocks bought
            cost_short = pos_y[0] * pos_y[1] # cost of short => opening price * number of stocks (this is a simplified consideration) 
            profit = (self.stockX_trading[i] - self.pos_x[0]) * pos_x[1] + (self.pos_y[0] - self.stockY_trading[i]) * pos_y[1]
        else:
            # long on Y (meaning long spread - short X)
            cost_long = pos_y[0] * pos_y[1]
            cost_short = pos_x[0] * pos_x[1]
            profit = (self.pos_x[0] - self.stockX_trading[i]) * pos_x[1] + (self.stockY_trading[i] - self.pos_y[0]) * pos_y[1]
        
        # second, compute returns considering the transaction costs as well
        if self.return_on_both_legs:
            # (computes the unlevered rets)
            # computing total rets considering both the legs of the trade (long & short) to compute initiation costs
            return_before_tc = (profit / (cost_long + cost_short)) * 100
            return return_before_tc * (1.0 - (self.tc / 100))
        else:
            # calculate the return only on the cost of long position (computes the levered rets)
            return_before_tc = (profit / cost_long) * 100
            return return_before_tc * (1.0 - (self.tc / 100))
    
    def generate_trading_signals(self):
        # generate the trading signals for the pair (basically computes the 1, -1 & 0 values for the trade signal based on level crossing)
        self.upper_trading = self.__level_crosses(self.normalized_spread_trading, level = self.trading_bound)
        self.lower_trading = self.__level_crosses(self.normalized_spread_trading, level = -self.trading_bound)
        self.upper_stop = self.__level_crosses(self.normalized_spread_trading, level = self.stop_loss)
        self.lower_stop = self.__level_crosses(self.normalized_spread_trading, level = -self.stop_loss)
        self.mean = self.__level_crosses(self.normalized_spread_trading, level = self.exit_profit)
        
        # record the count of mean crosses
        self.mean_crosses = self.mean.count(1) + self.mean.count(-1) 
    
        open_position = False # flag to check for position status
        # entry_level = 0 # pos opening entry level
        
        # placeholders
        self.stop_losses = []
        self.open_positions = []
        self.closed_positions = []
        returns = []
        
        i = 0
        while i < len(self.normalized_spread_trading):
            # loop over every data point in the normalised spread signal (each date until the end)
            if open_position:
                # if in an open position, 
                # 1. if signal crosses above upper stoploss or crosses below lower stoploss bounds - book Loss & record rets, update stoplosses & wait for reversal
                # 2. if signal crosses at the mean bounds - close pos & book Profit & record rets
                # 3. else - nothing, so record no change with '0' rets
                if self.upper_stop[i] == 1 or self.lower_stop[i] == -1:
                    # STOP LOSS (CLOSE WITH LOSS & WAIT FOR REVERSAL)
                    open_position = False
                    returns.append(self.__calculate_returns(pos_y= self.pos_y, pos_x=self.pos_x, i=i))
                    self.stop_losses.append(i)
                    i += self.waiting_days # wait for signal to reverse its course and get back within the bounds
                    returns.extend([0] * (self.waiting_days - 1))
                    continue
                elif self.mean[i] != 0:
                    # CLOSING WITH PROFIT
                    open_position = False
                    returns.append(self.__calculate_returns(pos_y= self.pos_y, pos_x=self.pos_x, i=i))
                    self.closed_positions.append(i)
                else:
                    returns.append(0)
            else:
                # if not in an open position,
                # enter into a pos if signal crosses below upper trading bound - short the spread
                # if signal above lower trading bound - go long the spread
                # else, no change, 
                # record returns as '0' (since in any case we are just opening a fresh positiion & not closing to be able to compute rets)
                if self.upper_trading[i] == -1 or self.lower_trading[i] == 1:
                    #ENTERING THE POSITION
                    open_position = True
                    self.open_positions.append(i)
                    #the return will not depend on the amount we fixed here
                    approx_amount = 100000
                    open_price_x = self.stockX_trading[i]
                    open_price_y = self.stockY_trading[i]
                    b = open_price_y / open_price_x
                    a = approx_amount / (open_price_y + b * open_price_x)
                    # we assume that you cannot buy portion of stocks, and the fixed amount is approximated (<1% error assumption!)
                    number_stocks_y = math.ceil(a)
                    number_stocks_x = math.ceil(a*b)
                    total_eff_amount = number_stocks_y * open_price_y + number_stocks_x * open_price_x
                    # placeholders for recording trade details (price & shares) when a position was opened; 'l' for long and 's' for short
                    if self.upper_trading[i] == -1:
                        #LONG X, SHORT Y
                        self.pos_x = (open_price_x, number_stocks_x, 'l')
                        self.pos_y = (open_price_y, number_stocks_y, 's')
                    elif self.lower_trading[i] == 1:
                        #SHORT X, LONG Y
                        self.pos_x = (open_price_x, number_stocks_x, 's')
                        self.pos_y = (open_price_y, number_stocks_y, 'l')
                    # record the spread level for entering into position 
                    # entry_level = self.spread[i]
                # here in this step, we just opened a fresh pos, so record rets as '0'
                returns.append(0)
            # step up to the next date
            i += 1
            
        self.profitable_trades_perc = len(self.closed_positions) / (len(self.closed_positions) + len(self.stop_losses)) * 100
        self.average_holding_period = self.__average_holding_period()
        
        self.returns_series = pd.Series(index = self.stockX_trading.index, data = returns)
        self.cum_returns = self.returns_series.cumsum()
        
    def plot_pair(self):
        # method to plot the price movement charts for the pair
        fig, (ax_stockX, ax_spread) = plt.subplots(2, 1)
        fig.suptitle(t=f'Price movement and trading signal charts for the pair {self.stockX_trading.name, self.stockY_trading.name}', 
                     fontsize=16, fontweight='bold')
        
        ax_stockX.title.set_text("Stocks prices")
        ax_spread.title.set_text("Normalized spread")
        
        plt_X = ax_stockX.plot(self.stockX_trading, color = "b", label = self.stockX_trading.name)
        ax_stockY = ax_stockX.twinx()
        plt_Y = ax_stockY.plot(self.stockY_trading, color = "y", label = self.stockY_trading.name)
        # Solution for having two legends
        leg = plt_X + plt_Y
        labs = [l.get_label() for l in leg]
        ax_stockX.legend(leg, labs, loc=0)

        
        ax_spread.plot(self.normalized_spread_trading[self.trading_start_date : self.trading_end_date])
        ax_spread.axhline(self.trading_bound, linestyle = '--', color = "g", label = "Trading bound")
        ax_spread.axhline(-self.trading_bound, linestyle = '--', color = "g")
        ax_spread.axhline(self.stop_loss, linestyle = '--', color = "r", label = "Stop loss")
        ax_spread.axhline(-self.stop_loss, linestyle = '--', color = "r")
        ax_spread.plot_date([self.normalized_spread_trading.index[i] for i in self.open_positions], [self.normalized_spread_trading[i] for i in self.open_positions], label = 'Open position', marker = '^', markeredgecolor = 'b', markerfacecolor = 'b', markersize = 16)
        ax_spread.plot_date([self.normalized_spread_trading.index[i] for i in self.closed_positions], [self.normalized_spread_trading[i] for i in self.closed_positions], label = 'Closed position', marker = 'P', markeredgecolor = 'g', markerfacecolor = 'g', markersize = 16)
        ax_spread.plot_date([self.normalized_spread_trading.index[i] for i in self.stop_losses], [self.normalized_spread_trading[i] for i in self.stop_losses], label = 'Stop loss', marker = 'X', markeredgecolor = 'r', markerfacecolor = 'r', markersize = 16)
        ax_spread.legend(loc=0)

        fig.set_size_inches(16, 10, forward = True)
        fig.show()
        plt.tight_layout()
        
    def __repr__(self):
        s = f"Pair [{self.stockX_trading.name}, {self.stockY_trading.name}]"
        s += f"\n\tp-value: {self.p_value}"
        s += f"\n\tMean crosses: {self.mean_crosses}"
        s += f"\n\tPair eligible: {self.eligible()}"
        s += f"\n\tProfitable trades (%): {self.profitable_trades_perc}"
        s += f"\n\tAverage holding period (days): {self.average_holding_period}"
        return s
    

class Portfolio:
    def __init__(self, stocks_df: pd.DataFrame, pairs_list: list):
        if not isinstance(stocks_df, pd.core.frame.DataFrame):
            raise Exception("Symbols must be provided in a Pandas DataFrame")
        
        self.time_series_df = stocks_df
        # self.time_series_df.dropna(inplace = True)
        self.symbols_list = self.time_series_df.columns
        # self.all_possible_pairs = list(combinations(self.symbols_list, 2))
        self.defined_pairs = pairs_list
        self.selected_pairs = list()
        
        for i, pair_symbols in enumerate(self.defined_pairs):
            print(f"{i}/{len(self.defined_pairs)}")
            pair = Pair(self.time_series_df[pair_symbols[0]], 
                        self.time_series_df[pair_symbols[1]])
            if pair.eligible():
                self.selected_pairs.append(pair)
        
        self.calculate_portfolio_return()
                
    def calculate_portfolio_return(self):
        # compute portfolio return & cumm return  metrics
        data = dict()
        for pair in self.selected_pairs:
            data[pair.name] = pair.returns_series
        df_return = pd.DataFrame(data = data)
        df_return['Return'] = df_return.mean(axis = 1)
        self.returns = df_return['Return']
        df_return['Cumulative Return'] = df_return['Return'].cumsum()
        self.cum_return = df_return['Cumulative Return']
        self.sharpe_ratio = np.nanmean(self.returns)/np.nanstd(self.returns)
        
    def plot_portfolio(self):
        # plot portfolio
        plt.title(f"Return of a selective portfolio of ({len(self.selected_pairs)}) pairs of cointegrated stocks\n", 
                  fontweight='bold', fontsize=12)
        plt.plot(self.returns, label='Daily returns')
        plt.plot(self.cum_return, linestyle = '--', label='Cummulative returns')
        plt.ylabel("Percentage return (%)")
        plt.legend(loc='best')
        plt.figtext(0.65, 0.4, f'Sharpe ratio = {self.sharpe_ratio:.4f}', bbox=dict(facecolor='orange', alpha=0.3), fontsize=10)
        plt.show()
        
    def plot_pairs(self):
        # plot pairs price movements
        for pair in self.selected_pairs:
            pair.plot_pair()
conn.close()