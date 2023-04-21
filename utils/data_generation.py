#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
import datetime as dt
from collections import Counter
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

from tqdm import tqdm
from openbb_terminal.sdk import openbb
openbb.keys.fred(key=os.getenv('FRED_API_KEY_1'))
openbb.keys.fmp(key=os.getenv('FMP_RSK_KEY_2'))
import sqlite3
import requests
import time

import warnings
warnings.filterwarnings("ignore")



# Pull the S&P 500 constituents profile data from Wiki page (public data)
def fetch_sp500_comp(wiki_url: str=os.getenv('WIKI_SP500_URL')):
    """
    Fetch the S&P 500 constituents profile data from Wiki page (public data)
    """
    wiki_table = pd.read_html(wiki_url)
    sp500_comp_profile = wiki_table[0]
    sp500_comp_profile['Symbol'] = sp500_comp_profile['Symbol'].str.replace('\\.', '-', regex=True)
    sp500_comp_profile.rename(columns={"Symbol": "ticker", "Security": "company_name", "CIK": "cik", "GICS Sector": "sector", 
                                    "GICS Sub-Industry": "sub_industry", "Headquarters Location": "hq", "Founded": "founded", 
                                    "Date added": "date_added"}, inplace=True)
    return sp500_comp_profile

def get_bus_desc_data(tickers, conn):
    """
    Returns a Series of bus descriptions.
    """
    # Initialize an empty dict for the ticker : description data
    bus_desc = {}
    # Loop over each ticker symbol and get the company description
    for tkr in tqdm(tickers):
        # Define the API endpoint with the ticker symbol and API key
        API_KEY = os.getenv('ALPHA_VANT_API_KEY')
        endpoint = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={tkr}&apikey={API_KEY}'
        print(f"Processing ticker: {tkr}")  # Added to show the ticker being processed
            
        # Make the API request and extract the company description from the response
        try:
            key_stats_response = requests.request("GET", endpoint)
            response_data = key_stats_response.json()

            if 'Description' in response_data:
                bus_desc[tkr] = response_data['Description']
            else:
                print(f"No 'Description' field found in response for {tkr}")
                bus_desc[tkr] = None
                
            # print(response_data)
            print(f'{bus_desc}')

            # Wait for 12 seconds before the next API call to stay within the rate limit of 5 calls per minute
            time.sleep(12)

        except Exception as e:
            print(f"Error for {tkr}: {e}")
            break

    bus_desc_df = pd.Series(bus_desc)
    bus_desc_df = bus_desc_df.reset_index(name='business_desc').rename(columns={'index': 'ticker'})
    bus_desc_df.to_sql('stocks_profile', conn, if_exists='replace', index=False)
    return bus_desc_df


def get_hist_price_data(tickers, start_date: str, end_date: str, conn):
    """
    Get historical price data from OpenBB python package - AlphaVantage
    start_date: str="2010-01-01"
    end_date: str="2023-03-31"
    
    """
    stock_dict = {}
    for tkr in tqdm(tickers):
        stock_data = openbb.stocks.load(tkr, start_date=start_date, end_date=end_date, source='AlphaVantage', 
                                        verbose=False).reset_index().rename(columns={"date": "date"})
        if len(stock_data) > 0:
            # calculate simple returns & log returns
            stock_data['simple_return'] = stock_data['Adj Close'].pct_change()
            stock_data['log_return'] = np.log(stock_data['Adj Close']/stock_data['Adj Close'].shift(1))
        else:
            print(f'No data found for {tkr}')

        stock_dict[tkr] = stock_data
        time.sleep(11) # wait for 11 seconds before the next API call to stay within the rate limit of 5 calls per minute

    # create a list of DFs of individual stock historical price data
    dfs = []
    for ticker, data in stock_dict.items():
        data['ticker'] = ticker
        dfs.append(data)

    # concatenate all DFs to form a single large DF
    all_stock_hist_price = pd.concat(dfs, ignore_index=False)
    # all_stock_hist_price = all_stock_hist_price.drop('index', axis=1)
    all_stock_hist_price['date'] = pd.to_datetime(all_stock_hist_price['date'])
    # set the order of columns for better readability
    column_order = ['ticker'] + [col for col in all_stock_hist_price.columns if col != 'ticker']
    all_stock_hist_price = all_stock_hist_price[column_order]

    # Check if any stocks were left out
    tkr_unique = all_stock_hist_price['ticker'].unique()
    tkr_dropped = [ tkr for tkr in tickers if tkr not in tkr_unique ]
    print(f'{len(tkr_dropped)} missed out: {tkr_dropped}') # none were dropped, awesome!

    # Create a separate DF containing historical price data of all the stocks and store it to the database
    all_stock_hist_price.to_sql('stocks_hist_price', conn, if_exists='replace', index=False)
    return all_stock_hist_price

def get_hist_ratios_data(tickers, start: int, conn):
    """
    Get historical financial ratios data from OpenBB python package - Financial Modeling Prep
    start: int=2010
    """
    ratios_dict = {}
    for ticker in tqdm(tickers):
        stock_ratios = openbb.stocks.fa.ratios(ticker, 15)
        if len(stock_ratios) > 0:
            stock_ratios = stock_ratios.T
            stock_ratios = stock_ratios.drop(columns='Period').reset_index()
            stock_ratios['Fiscal Date Ending'] = stock_ratios['Fiscal Date Ending'].astype(int)
            stock_ratios = stock_ratios[stock_ratios['Fiscal Date Ending'] >= 2010]
            stock_ratios = stock_ratios.reindex(index=stock_ratios.index[::-1]).reset_index(drop=True)
            stock_ratios['ticker'] = ticker
        else:
            print(f'No data found for {ticker}')
        ratios_dict[ticker] = stock_ratios

    # create a list of DFs of individual stock historical ratios data
    ratio_dfs = []
    for tkr, data in ratios_dict.items():
        ratio_dfs.append(data)

    # concatenate all DFs to form a single large DF
    all_stock_hist_ratios = pd.concat(ratio_dfs, ignore_index=False)
    # all_stock_hist_ratios = all_stock_hist_price.drop('index', axis=1)
    # set the order of columns for better readability
    column_order = ['ticker'] + [col for col in all_stock_hist_ratios.columns if col != 'ticker']
    all_stock_hist_ratios = all_stock_hist_ratios[column_order]

    # Check if any stocks were left out
    tkr_unique = all_stock_hist_ratios['ticker'].unique()
    tkr_dropped = [ tkr for tkr in tickers if tkr not in tkr_unique ]
    print(f'{len(tkr_dropped)} missed out: {tkr_dropped}') # 1 stock missed out ('PEG')
    # For now, let's not re-fetch the price data for dropped tickers, we'll proceed with what we have  

    # Create a separate DF containing historical price data of all the stocks and store to the database
    all_stock_hist_ratios.to_sql('stocks_hist_ratios', conn, if_exists='replace', index=False)
    return all_stock_hist_ratios

# In[ ]:




