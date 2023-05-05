[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE) [![DVC](https://img.shields.io/badge/-Data_Version_Control-white.svg?logo=data-version-control&style=social)](https://dvc.org/?utm_campaign=badge) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

# Stock cluster based pairs-trading

## Introduction
Statistical arbitrage is a class of trading strategies that profit from exploiting what are believed to be market inefficiencies. These inefficiencies are determined through statistical and econometric techniques. Note that the arbitrage part should by no means suggest a riskless strategy, rather a strategy in which risk is statistically assessed.

One of the earliest and simplest types of statistical arbitrage is pairs trading. It was first introduced in the mid 1980s by a group of top research analysts working at Morgan Stanley. In short, pairs trading is a strategy in which two securities that have been cointegrated over the years are identified and any short-term deviations from their long-term equilibrium create opportunities for generating profits. 

## Problem Statement
To find the cointegrated pairs of stocks from a defined stock universe is very compute intensive (For instance, for a stock universe of S&P 500 with 500 different stocks, the possible number of pairs to evaluate for cointegration would amount to 124750 pairs). A better approach would be to group similar stocks together and then find cointegrated pairs among these individual groups. We could use the GICS sector classification definitions to group stocks based on their GICS sectors. However, a major drawback of this method is that these definitions are susceptible to abrupt changes (eg, recent changes, March 2023, caused both VISA and MASTERCARD to be moved to the Payments sub-industry under the Financial sector, which were initially part of the GICS Tech sector). And more often than not, many major conglomerates are engaged in businesses across multiple sectors and industries, which makes the sector based stock clustering inefficient. 

In this work, we evaluate if grouping stocks based on a combination of their historical financial ratios and company descriptions (textual data derived from Wiki pages), leads to the formation of better stock clusters, and thereby better cointegrated pairs. 

## Dataset used
1. S&P500:
503 stocks | Jan 2010 to Mar 2023 | OHLCV data | company descriptions (static) | annual values of 10 financial metrics were considered | sources: OpenBB, Alphavantage, Financial Modeling Prep

## Project file structure
### experiments folder 
This folder contains all the python notebooks. Specifically, it has 3 EDA notebooks which give insights on the different datasets and clustering based on individual datasets alone. It also contains the following 3 notebooks:

1. #### generate_data.ipynb
It contains the code for collecting novel data from various sources using APIs (OpenBB, AlphaVantage, FRED) and using a SQLite DB for storing the data. It also presents the design of the database schema.

2. #### stock_clustering.ipynb
It contains the code for clustering stocks based on a combination of historical ratios dataset and the company descriptions dataset using various clustering algorithms and discusses our findings and results.

3. #### pairs_trading.ipynb
It contains the code for finding the cointegrated pairs, building a trading strategy, backtesting and comparing its performance with the benchmark (S&P 500).

### utils folder 
This folder the python scripts for the above 3 python notebooks with code modularisation.

## Setup & execution
1. Create a virtual environment using the following command:
    ```bash
    conda env create -f environment.yml
    ```
2. Ensure your virtual environment is activated and run the main.py script 
