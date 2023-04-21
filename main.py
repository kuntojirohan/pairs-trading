#!/usr/bin/env python
# coding: utf-8

# In[6]:


from utils.data_generation import fetch_sp500_comp, get_bus_desc_data, get_hist_price_data, get_hist_ratios_data
import sqlite3
import pandas as pd

# Initialise the database
conn = sqlite3.connect('data/pairs_trading.db')

sp500_comp_profile_df = fetch_sp500_comp()

ticker_list = sp500_comp_profile_df['ticker'].to_list()
sp500_comp_profile_df = get_bus_desc_data(tickers=ticker_list, conn=conn)
stocks_hist_price_df = get_hist_price_data(tickers=tickers, start_date="2010-01-01", end_date="2023-03-31", conn=conn)
stocks_hist_ratios_df = get_hist_ratios_data(tickers, start=2010, conn)

print(f'Total stocks in S&P 500 profile data DF: {len(sp500_comp_profile_df)}')

stocks_hist_price_df = pd.read_sql_query("SELECT * FROM stocks_hist_price", conn)
print(f'Total count of unique stocks: {len(stocks_hist_price_df.ticker.unique())}\n') # in total, historical price data was fetched for 502 stocks 

stocks_hist_ratios_df = pd.read_sql_query("SELECT * FROM stocks_hist_ratios", conn)
print(f'Total count of unique stocks: {len(stocks_hist_ratios_df.ticker.unique())}\n') # in total, historical ratios data was fetched for 502 stocks 

profile_tickers = sp500_comp_profile_df.ticker.unique().tolist()
print(f'Tickers in profile dataset ({len(profile_tickers)}):\n{profile_tickers}')
print('--'*20)
price_tickers = stocks_hist_price_df.ticker.unique().tolist()
print(f'Tickers in price dataset ({len(price_tickers)}):\n{price_tickers}')
print('--'*20)
ratio_tickers = stocks_hist_ratios_df.ticker.unique().tolist()
print(f'Tickers in ratios dataset ({len(ratio_tickers)}):\n{ratio_tickers}')
print('--'*20)
common_tickers_universe = [tkr for tkr in profile_tickers if tkr in price_tickers and tkr in ratio_tickers]
print(f'\nCommon universe of tickers ({len(common_tickers_universe)}):\n{common_tickers_universe}')

# Count of companies in each sector
sector_counts = pd.read_sql_query('''SELECT sector, COUNT(*) as num_companies FROM stocks_profile 
                                        GROUP BY sector ORDER BY num_companies DESC limit 10''', conn)

print("Top 10 sectors with the highest number of companies")
print(sector_counts)


companies_by_location = pd.read_sql_query('''
SELECT
 -- extract the state or country from the 'hq' column
  CASE
  
  -- Check if the 'hq' value contains a comma followed by a space (', ')
  
    WHEN hq LIKE '%, %' THEN 
    
    -- If there's a comma followed by a space, extract the string after the comma
    
    SUBSTR(hq, INSTR(hq, ',') + 2)
    
    -- If there's no substring after a comma, then set as Unknown
    ELSE 'Unknown'
  END as location,
  COUNT(*) as num_companies
FROM stocks_profile
GROUP BY location
ORDER BY num_companies DESC 
LIMIT 10
''', conn)

print("\nNumber of companies based on each location:")
print(companies_by_location)

top_10_avg_volume = pd.read_sql_query('''SELECT ticker, printf('%.0f', AVG(Volume)) as avg_volume FROM stocks_hist_price 
                                            GROUP BY ticker ORDER BY avg_volume DESC LIMIT 10''', conn)

print("\nTop 10 tickers with the highest average daily trading volume:")
print(top_10_avg_volume)

top_10_days_total_volume = pd.read_sql_query('''SELECT strftime('%Y-%m-%d', date) as date, printf('%.0f', SUM(Volume)) as total_volume
                                                FROM stocks_hist_price 
                                                GROUP BY date ORDER BY total_volume DESC LIMIT 10''', 
                                             conn)

print("\nTop 10 trading days with the highest total trading volume across all tickers:")
print(top_10_days_total_volume)

top_10_roe_companies = pd.read_sql_query("""SELECT ticker, MAX("Fiscal Date Ending") as latest_fiscal_date,
                                            MAX("Return on equity") as ROE FROM stocks_hist_ratios 
                                            GROUP BY ticker ORDER BY ROE DESC LIMIT 10""", conn)

print("\nTop 10 companies with the highest Return on Equity (ROE) for 2022:")
print(top_10_roe_companies)


top_10_gpm_companies = pd.read_sql_query("""SELECT ticker, MAX("Fiscal Date Ending") as latest_fiscal_date,
                                            MAX("Current ratio") as current_ratio FROM stocks_hist_ratios 
                                            GROUP BY ticker ORDER BY current_ratio DESC LIMIT 10""", conn)

print("\nTop 10 companies with the highest Current Ratio for 2022:")
print(top_10_gpm_companies)

conn.close()


