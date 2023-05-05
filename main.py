#!/usr/bin/env python
# coding: utf-8
#PATH: ./main.py
#--This is the main file that runs all the functions from 
# data_generation, stock_clustering, pairs_trading
# In[6]:


from utils.data_generation import *
from utils.stock_clustering import *
from utils.pairs_trading import *
import sqlite3
import pandas as pd

## -- Novel Data Set Collection -- ##

# Initialise the database
conn = sqlite3.connect('data/pairs_trading.db')

sp500_comp_profile_df = fetch_sp500_comp()

ticker_list = sp500_comp_profile_df['ticker'].to_list()
sp500_comp_profile_df = get_bus_desc_data(tickers=ticker_list, conn=conn)
stocks_hist_price_df = get_hist_price_data(tickers=ticker_list, start_date="2010-01-01", end_date="2023-03-31", conn=conn)
stocks_hist_ratios_df = get_hist_ratios_data(ticker_list, start=2010, conn=conn)

## -- End of Novel Data Set Collection -- ##


## -- Database Querying and Reporting -- ##

display_analysis(conn)


## -- End of Database Querying and Reporting -- #

## -- Data Preparation -- ##

# clean and preprocess the stock ratios data

stock_ratios_df = pd.read_sql_query("SELECT * FROM stocks_hist_ratios", conn)
sp500_stocks_profile_df = pd.read_sql_query("SELECT * FROM stocks_profile", conn)

# clean and preprocess ratios data
target_ratios = ['Quick ratio', 'Cash ratio', 'Interest coverage', 'Debt equity ratio', 'Asset turnover', 'Receivables turnover', 
                 'Return on assets', 'Operating profit margin', 'Enterprise value multiple', 'Payout ratio']
ratios_pp_df = clean_preprocess_ratios_data(stock_ratios_df)

# clean and preprocess stock business descriptions text
document_topic_df, word_topic_df, sing_topic_df = preprocess_bus_desc_data(sp500_stocks_profile_df[['ticker', 'business_desc']], 
                                                                           n_topics=80)

# Plot top 10 terms within each topic
plot_topic_top10_terms(word_topic_df)

# combined feature space creation
final_features_df = combine_and_normalize_data(ratios_pp_df, document_topic_df)
print(final_features_df.shape)

## -- End of Data Preparation -- ##

## -- Model Building -- ##

# K-Means
km_cluster_df = find_and_fit_optimal_kmeans(final_features_df)

# OPTICS
db_cluster_df = optics_fit(final_features_df, min_samples=7)

# Hierarchical
hc_cluster_df = agg_hc_fit(final_features_df, n_clusters=14, linkage='average')

# Cluster Analysis
compare_clustering_results(km_cluster_df, db_cluster_df, hc_cluster_df, sp500_stocks_profile_df)
final_cluster_df = get_final_clabel_profile_df(db_cluster_df, target_ratios)
boxplot_cluster_fin_ratios(final_cluster_df, target_ratios)
plot_cluster_wordclouds(final_cluster_df)

## -- End of Model Building -- ##

## -- Textual Analysis -- ##

filtered_stock_profiles = filter_stock_profiles(sp500_stocks_profile_df)
filtered_stock_profiles = apply_preprocessing(filtered_stock_profiles)
unique_words_count = count_unique_words(filtered_stock_profiles)
tfidf_sparse_matrix = tfidf_transform(filtered_stock_profiles)
visualize_sparse_matrix(tfidf_sparse_matrix)

## -- End of Textual Analysis -- ##

##-- Pairs Trading --

# find cointegrated pairs and perform pairs selection
hist_final_stock_pairs = cluster_pair_selection(cluster_df=target_cluster_df)
print(f"Number of clusters: {len(target_cluster_df.cluster.value_counts())}")
print(f"Number of cointegrated pairs: {len(hist_final_stock_pairs)}")
print(f"Pairs with lowest p-value from each clusters:\n{hist_final_stock_pairs}")

# create a portfolio of selected pairs based on the set criterion
portfolio = Portfolio(stocks_df=full_hist_close_df, pairs_list=hist_final_stock_pairs)
print(f'Selected pairs: \n {portfolio.selected_pairs}')
portfolio.plot_portfolio()

# plot performance charts for the benchmark for comparison
plot_benchmark_ret(conn)

## -- End of Pairs Trading --
conn.close()

