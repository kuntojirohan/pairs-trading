#!/usr/bin/env python
# coding: utf-8
#PATH: utils/stock_clustering.py
##--This script contains code for data pre-processing, data visualisation, text analytics, and model building--

import pandas as pd
import numpy as np
import datetime as dt
from collections import Counter
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import colorcet as cc
sns.set(style="white", rc={"figure.figsize":(8, 4)})
plt.style.use('ggplot') # fivethirtyeight, ggplot, dark_background, classic,  

from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn import feature_extraction
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans, OPTICS, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from sklearn.impute import KNNImputer
from scipy.spatial.distance import pdist, cdist

from kneed import KneeLocator
import re
import nltk
# nltk.download('stopwords') # required to be downloaded only for the first time
# nltk.download('wordnet') # required to be downloaded only for the first time
# nltk.download('omw-1.4') # required to be downloaded only for the first time
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from wordcloud import WordCloud

import warnings
warnings.filterwarnings("ignore")


def std_normalise_data(data = pd.DataFrame()):
    """
    Standard scaling and Normalisation of input data

    Parameters:
    -----------
    data : pd.DataFrame
            raw data

    Returns: 
    --------
    new_df : pd.DataFrame
            scaled data
    """
    new_data = StandardScaler().fit_transform(data)
    new_data = Normalizer().fit_transform(new_data) 
    new_df = pd.DataFrame(new_data, index=data.index, columns=data.columns)
    return new_df

def knn_imputer(data: pd.DataFrame, n_neighbours: int=30):
    """
    Data imputation with KNNImputer

    Parameters:
    -----------
    data : pd.DataFrame
            raw data that needs to be imputed to handle missing values.

    n_neighbours : int, default value is 30
            Number of neighboring samples to use for imputation.

    Returns: 
    --------
    new_df : pd.DataFrame
            imputed data with zero NaN values.
    """
    # scale the data using minmaxscaler
    scaled_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
    # Define KNN imputer and fill missing values
    knn_fit_data = KNNImputer(n_neighbors=n_neighbours, weights='distance').fit_transform(scaled_data)
    new_df = pd.DataFrame(knn_fit_data, columns=data.columns, index=data.index)

    # check if there are any more missing values after imputation
    print(f'Total count of missing values before imputing with KNNImputer: {data.isna().sum().sum()}')
    print(f'Total count of missing values after imputing with KNNImputer: {new_df.isna().sum().sum()}')
    return new_df

def pca_tuning(data = pd.DataFrame()):
    """
    PCA tuning - dimensionality reduction

    method 1 - Plot the cummulative explained variance against the count of principal components
    
    method 2 - Create a 'Scree' plot which gives the visual representation of eigenvalues 
    that define the magnitude of eigenvectors (principal components)

    Parameters:
    -----------
    data : pd.DataFrame
            data on which to fit the PCA model

    Returns: 
    --------
    pca_base : PCA object
            Fitted PCA model object

    knee_value : int
            computed Knee point from the Scree plot        
    """
    # Start with basic PCA, keeping all components with no reduction   
    pca_base = PCA()
    pca_base.fit(data)

    plt.figure(figsize=(12, 6))
    # Method 1
    # Fetch variance explained by each individual component & compute the cummulative sum
    print('Method 1 - variance explained by components') 
    plt.subplot(1, 2, 1)
    exp_var = pca_base.explained_variance_ratio_ * 100
    cum_exp_var = np.cumsum(exp_var)
    plt.bar(range(0, exp_var.size), exp_var, align='center', label='Individual explained variance (%)')
    plt.step(range(0, cum_exp_var.size), cum_exp_var, where='mid', label='Cumulative explained variance (%)', color='indigo')
    plt.ylabel('Explained variance (%)')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')

    # Method 2 - Scree plot
    print('\nMethod 2 - Scree plot')
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(pca_base.n_components_) + 1, pca_base.explained_variance_, 'ro-', linewidth=2,  color='orange')
    knee = KneeLocator(np.arange(pca_base.n_components_) + 1, pca_base.explained_variance_, S=20, 
                       curve='convex', direction='decreasing', interp_method='interp1d')
    plt.axvline(x=knee.knee, ymax=0.95, ymin=0.05, color='brown', ls='--')
    plt.figtext(0.65, 0.4, f'optimal n_comp = {knee.knee}', bbox=dict(facecolor='violet', alpha=0.3), fontsize=10)
    plt.title("Scree Plot")
    plt.xlabel("Principal component index")
    plt.ylabel("Eigenvalues")
    print(f'Knee point, eigenvalue : {knee.knee, pca_base.explained_variance_[knee.knee]}')
    plt.tight_layout(pad=2.0)
    plt.show()

    return pca_base, knee.knee


    
def clean_preprocess_ratios_data(raw_data: pd.DataFrame, target_date_range: range=range(2010, 2020), copy: bool=True,
                                 target_ratios: list=None):
    """
    Perform data cleaning and preprocessing
    on raw historical financial ratios dataset.

    Parameters:
    -----------
    raw_data : pd.DataFrame
        large single dataframe of historical ratios for all the stocks.

    target_date_range : range
        range of fiscal years to be considered for fetching annual historical ratios measures.
        It's a range of years which includes the first value but not the last value. 

    copy : bool, default value is True
        If set to True, all processing will be done on a copy of the input raw data. 
        Else, the input raw data will be modified.
    
    target_ratios : list, default list of desired targets defined
        The list of specific ratios to be considered as the target feature set for model fitting. 

    Returns: 
    --------
    ratios_final_df : pd.DataFrame
        cleaned and processed dataframe of target historical ratio measures within the specified date range.
    """
    
    if target_ratios is None:
        target_ratios = ['Quick ratio', 'Cash ratio', 'Interest coverage', 'Debt equity ratio', 'Asset turnover', 
                 'Receivables turnover', 'Return on assets', 'Operating profit margin', 
                 'Enterprise value multiple', 'Payout ratio']

    
    if copy:
        raw_data = raw_data.copy()
    # reshuffling DF into a pivot DF
    pivot_df = raw_data[['ticker', 'Fiscal Date Ending'] + target_ratios].pivot(
        index='ticker', columns='Fiscal Date Ending', values=target_ratios)
    # change the dtypes from object to float
    pivot_df = pivot_df.apply(pd.to_numeric, errors='coerce', downcast='float')
    # filter data only within the target date range
    target_df = pivot_df[[col_tup for col_tup in pivot_df.columns if col_tup[1] in target_date_range]]

    # data imputation of missing values
    # From our EDA, we found KNNImputer works the best on this data for n_neighbours=30
    print('-'*50)
    print('Data Imputation\n')
    target_imp_df = knn_imputer(target_df, n_neighbours=30)

    # PCA - dimensionality reduction
    # standard scaling and normalisation before fitting PCA
    print('-'*50)
    print('PCA for Dimensionality Reduction\n')
    scaled_df = std_normalise_data(target_imp_df)
    # Tune a default PCA model and find optimal n_components for fitting a PCA
    # Note: PCA needs the input data to be in (n_features, n_samples) format
    # for it to function properly and reduce the feature dimensionality
    pca_base, knee_value = pca_tuning(scaled_df.T)
    print('-'*50)
    print(f'Fit a new PCA with n_components = {knee_value} as observed from the Scree plot')
    # Fit a new PCA model with the obtained hyperparam value testing
    pca_final = PCA(n_components=knee_value)
    pca_final.fit(scaled_df.T) # Note: we need to input data in the exact required format here for the PCA to work properly

    # Structure the final results in a dataframe
    ratios_final_df = pd.DataFrame(pca_final.components_.T, index=scaled_df.index, columns=[f'PC_{i}' for i in 
                                                                                            range(pca_final.components_.T.shape[1])])
    
    return ratios_final_df

def text_cleaning(text: str, flg_stemm=False, flg_lemm=True):
    """
    Clean & preprocess input string data. (remove stop words from the text, stemming/lemmatisation)

    Parameters:
    -----------
    text : str
        Textual data of string type.

    flg_stemm : bool, default is False
        Flag to indicate whether stemming should be performed on the input text.
        Note: You should not set both flg_stemm & flg_lemm to be True. Only one of them can be True at a time.

    flg_lemm : bool, default value is True
        Flag to indicate whether lemmatisation should be performed on the input text.
        Note: You should not set both flg_stemm & flg_lemm to be True. Only one of them can be True at a time.

    Returns: 
    --------
    text : str
        cleaned and processed string (removed stop words, stemming/lemmatisation).
    """
    # clean (convert to lowercase and remove punctuations and special characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    # Tokenize (convert from string to list)
    tokens = text.split()
    # remove Stopwords
    stop_words.extend(['founded', 'firm', 'company', 'llc', 'inc', 'incorporated', 
                       'multinational', 'corporation', 'commonly', 'headquartered']) # extend the default list by adding more non-important words
    if stop_words is not None:
        tokens = [word for word in tokens if word not in stop_words]
    # Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        tokens = [ps.stem(word) for word in tokens]
    # Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        tokens = [lem.lemmatize(word) for word in tokens]
            
    # back to string from list
    text = ' '.join(tokens)
    return text


def preprocess_bus_desc_data(raw_data: pd.DataFrame, copy: bool=True, n_topics: int=150):
    """
    Perform data cleaning and preprocessing
    on raw historical financial ratios dataset.

    Parameters:
    -----------
    raw_data : pd.DataFrame
        Dataframe containing stock tickers and their respective business descriptions.

    copy : bool, default value is True
        If set to True, all processing will be done on a copy of the input raw data. 
        Else, the input raw data will be modified.

    n_topics : int, default value is 150
        number of topics to be deduced from LSA.

    Returns: 
    --------
    document_topic_df : pd.DataFrame
        DF containing Document-topic matrix values.

    word_topic_df : pd.DataFrame
        DF containing Word-topic matrix values.

    sing_topic_df : pd.DataFrame
        DF containing singular matrix values.  
    """
    
    if copy:
        raw_data = raw_data.copy()
        
    # drop tickers with missing values in business_desc column only
    print('-'*50)
    print('Drop stocks with missing business description data since this forms the bedrock of feature creation\n')
    raw_data.dropna(axis=0, subset=['business_desc'], inplace=True)
    print(f'Total stocks in consideration after dropping the ones with missing business descriptions: {raw_data.shape[0]}') 
    # perform text cleaning to get rid of stopwords, apply stemming/lemmatisation rules
    print('-'*50)
    print('Carry out text cleaning which includes - removing stopwords, stemming & lemmatisation\n')
    raw_data['bd_clean'] = raw_data['business_desc'].apply(lambda txt: text_cleaning(txt))
    print(f'Total count of unique words (unigrams) found in our corpus of business descriptions across all stocks:       {len(set([ele for arr in list(raw_data["bd_clean"].apply(lambda x: str(x).split(" "))) for ele in arr]))}')
    # Vectorise the textual data using TF-IDF method and get a sparse DTM on the cleaned data (tuning the Tf-Idf hyperparams min_df, max_df & ngram_range)
    print('-'*50)
    print('Vectorise cleaned text data using TF-IDF method, get a sparse DTM - improve vectorisation with appropriate hyperparam values\n')
    tfidf = feature_extraction.text.TfidfVectorizer(stop_words='english', max_features=10000, min_df=1, max_df=0.1,
                                                strip_accents='unicode', ngram_range=(2,4))
    tfidf_sparse = tfidf.fit_transform(raw_data['bd_clean'])
    tfidf_features = tfidf.get_feature_names_out().tolist()
    # let's see some features which were extracted by Tf-Idf
    print(f'TF-IDF feature names (a few samples): \n{tfidf_features[-100:]}\n')

    # Apply LSA to reduce dimensionality of TF-IDF sparse matrix
    print('-'*50)
    print('Apply LSA to reduce dimensionality of TF-IDF sparse matrix\n')
    # define a desired n_components (themes/topics) for a Truncated SVD model
    lsa_obj = TruncatedSVD(n_components=n_topics, n_iter=100, random_state=1500)
    # define tfidf sparse matrix as an actual DF
    tfidf_sparse_df = pd.DataFrame(data=tfidf_sparse.toarray(), index=raw_data.ticker, columns=tfidf_features)
    # compute document-topic matrix
    document_topic_m = lsa_obj.fit_transform(tfidf_sparse_df)
    # compute word-topic matrix
    word_topic_m = lsa_obj.components_.T
    # compute singular values topic matrix
    sing_topic_m = lsa_obj.singular_values_
    print(f'Reduced dimensionality from {len(tfidf_features)} to {lsa_obj.components_.shape[0]}\n')
    print(f'Total variance explained by the top {lsa_obj.components_.shape[0]} topics (%):           {np.sum(lsa_obj.explained_variance_ratio_) * 100}\n')
    # define topics
    topics = [f'Topic_{i}' for i in range(0, sing_topic_m.shape[0])]
    document_topic_df = pd.DataFrame(data=document_topic_m, index=raw_data.ticker, columns=topics)
    word_topic_df = pd.DataFrame(data=word_topic_m, index=tfidf_sparse_df.columns, columns=topics)
    sing_topic_df = pd.DataFrame(data=sing_topic_m, index=topics)
    
    return document_topic_df, word_topic_df, sing_topic_df


def plot_topic_top10_terms(data_df: pd.DataFrame):
    """
    Plot top 10 terms (ngram) within the topic for 
    a sample of topics.

    Parameters:
    -----------
    data_df : pd.DataFrame
        Word-topic DF computed from the LSA algoithm.
    """
    print('-'*50)
    print('Plot top terms (ngrams) within each topic\n')
    fig1 = plt.figure(figsize=(20, 20))
    fig1.subplots_adjust(hspace=.5, wspace=.5)
    plt.clf()
    for i in range(0, 6):
        fig1.add_subplot(5, 2, i+1)
        temp = data_df.iloc[:, i*15]
        temp = temp.sort_values(ascending=False)
        plt.title(f'Top 10 terms (ngrams) in Topic {i*15}', weight='bold', fontsize=14)
        sns.barplot(x= temp.iloc[:10].values, y=temp.iloc[:10].index)
        i += 1
    fig1.tight_layout(pad=2.0)
    plt.show()
    
def combine_and_normalize_data(ratios_pp_df: pd.DataFrame, document_topic_df: pd.DataFrame, normalise_function=std_normalise_data):
    """
    Combines the ratios_pp_df and document_topic_df dataframes, and applies standard scaling and normalization.
    """
    final_features_df = pd.concat([ratios_pp_df, document_topic_df], axis=1, join='inner')

    # Apply standard scaling and normalization
    final_features_df = normalise_function(final_features_df)
    
    fig, ax = plt.subplots(figsize=(8,8))
    plt.title('Feature space correlation map\n', fontsize=16)
    sns.heatmap(final_features_df.corr(), ax=ax)
    plt.show()

    return final_features_df
    
def plot_TSNE(data: pd.DataFrame, labels):
    """
    Plot the results of the t-SNE algorithm to visualise the 
    cluster formations from high dimensional data on a 2-D plot 

    Parameters:
    -----------
    data : pd.DataFrame
        Dataframe containing processed data on which the clustering model was fit.

    labels : nd array
        Array of computed cluster labels.

    Returns: 
    --------
    clustered_series_all : pd.Series
        Pandas series with tickers and cluster labels.
    """
    # all stock with its cluster label (including -1)
    clustered_series_all = pd.Series(index=data.index, data=labels)
    # use TSNE algorithm to plot multidimension into 2D
    tsne_data = TSNE(n_components=2, perplexity=80, random_state=1337).fit_transform(data)
    tsne_df = pd.DataFrame(data=tsne_data, index=data.index, columns=['tsne_1', 'tsne_2'])
    tsne_df['label'] = labels
    # clustered
    tsne_df = tsne_df[tsne_df['label'] != -1]
    sns.lmplot(data=tsne_df, x='tsne_1', y='tsne_2', hue='label', height=6, aspect=2, fit_reg=False, legend=True, palette=sns.color_palette(cc.glasbey, len(tsne_df.label.value_counts())), 
               scatter_kws={'s':150, 'alpha': 0.5})
    # unclustered in the background
    plt.scatter(tsne_data[(clustered_series_all==-1).values, 0], tsne_data[(clustered_series_all==-1).values, 1], s=50, alpha=0.05)
    plt.title('t-SNE plot for all stocks with cluster labels\n', weight='bold').set_fontsize('14')
    plt.xlabel('t-SNE comp 1', weight='bold', fontsize=12)
    plt.ylabel('t-SNE comp 2', weight='bold', fontsize=12)
    plt.tight_layout(pad=2.0)
    plt.show()
    return clustered_series_all


# show number of stocks in each cluster 
def plot_cluster_counts(labels_df: pd.DataFrame):
    """
    Plot cluster counts histogram bar chart 

    Parameters:
    -----------
    labels_df : pd.DataFrame (Series)
        Pandas Series containing tickers and cluster labels as output by plot_TSNE.
    """
    plt.figure(figsize=(12,8))
    labels_df[labels_df!=-1].value_counts().sort_index().plot(kind='barh')
    plt.title('Cluster stock counts\n', weight='bold', fontsize=12)
    plt.xlabel('Stocks in Cluster', weight='bold', fontsize=10)
    plt.ylabel('Cluster Number', weight='bold', fontsize=10)
    plt.tight_layout(pad=2.0)
    plt.show()

# plot price movements for cluster stocks
def plot_cluster_members(labels_df: pd.DataFrame):
    """
    Plot the cluster members' (stocks) log prices
    to observe if there's any similarity in patterns

    Parameters:
    -----------
    labels_df : pd.DataFrame (Series)
        Pandas Series containing tickers and cluster labels as output by plot_TSNE.
    """
    # get the number of stocks in each cluster 
    conn = sqlite3.connect('./data/pairs_trading.db')
    counts = labels_df[labels_df!=-1].value_counts()
    # let's visualize some clusters
    cluster_vis_list = list(counts[counts>1].sort_values().index)
    # this code needs to be replaced with code to fetch from db
    hist_price_df = pd.read_sql_query("SELECT * FROM stocks_hist_price", conn, parse_dates=['date'])
    #pd.read_csv('./data/stocks_hist_price.csv', date_parser=['date'])
    hist_price_df.date = pd.to_datetime(hist_price_df.date, format='ISO8601')
    sp500_stocks_profile_df = pd.read_sql_query("SELECT * FROM stocks_profile", conn)
    conn.close()
    temp_df = hist_price_df.pivot_table(values='Adj Close', index='date', columns='ticker')
    # plot a handful of the smallest clusters
    plt.figure(figsize=(24, 48))
    plt.subplots_adjust(hspace=.25, wspace=.25)
    plt.clf()
    i=1
    for clust in cluster_vis_list[0:min(len(cluster_vis_list), 4)]:
        tickers = list(labels_df[labels_df==clust].index)
        means = np.log(temp_df.loc[:dt.datetime(2019, 12, 31), tickers].mean())
        data = np.log(temp_df.loc[:dt.datetime(2019, 12, 31), tickers]).sub(means)
        plt.subplot(4, 1, i) 
        plt.plot(temp_df.loc[:dt.datetime(2019, 12, 31), tickers].index, data)
        plt.title(f'Time series of log returns for stocks in Cluster {clust}\n', weight='bold', fontsize=14)
        plt.xlabel('Date', weight='bold', fontsize=12)
        plt.ylabel('Log returns', weight='bold', fontsize=12)
        tkr_names = [f'{tkr} - {sp500_stocks_profile_df[sp500_stocks_profile_df.ticker == tkr].company_name.values[0]}' for tkr in tickers]
        plt.legend(tkr_names)
        i+=1
    plt.tight_layout(pad=2.0)
    plt.show()
    
    
def find_optimal_k(data = pd.DataFrame(), k_range: range=range(2, 50), init: str='k-means++'):
    """
    Plot the elbow curve & compute silhouette scores to find optimal 'k' for k-means
    for a range of k values.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataframe containing processed data on which the k-means model is to be fit.

    k_range : range, default value is range(2, 50)
        Range of 'k' values within which the model fit needs to be evaluated.

    init : str, default value is 'k-means++'
        {'k-means++', 'random'}, Method for initialisation.

    Returns: 
    --------
    knee_elbow.knee : int
        Elbow curve knee point.

    knee_ss.knee : int
        Silhouette score knee point.
    """
    WCSS = []
    SS = []
    for k in k_range:
        km = KMeans(n_clusters = k, init=init, max_iter=20000, n_init=100, random_state=1500)
        km.fit(data)
        WCSS += [km.inertia_]
        SS += [silhouette_score(data, labels=km.labels_, random_state=1500)]
    # plot the charts
    fig1 = plt.figure(figsize=(18, 8))
    fig1.subplots_adjust(hspace=.5, wspace=.25)
    plt.clf()
    # Elbow curve for WCSS scores
    plt.subplot(1, 2, 1)
    plt.plot(k_range, WCSS, color='violet', marker='*')
    plt.xlabel('k', weight='bold', fontsize=10)
    plt.ylabel('Within Cluster Sum of Squares', weight='bold', fontsize=10)
    plt.title(f'Elbow Curve: Plot of k vs WCSS (init={init})\n', weight='bold', fontsize=12)
    knee_elbow = KneeLocator(k_range, WCSS, S=10, online=True,  curve='convex', direction='decreasing', interp_method='interp1d')
    if knee_elbow.knee!=None and knee_elbow.knee in range(len(WCSS)):
        plt.axvline(x=knee_elbow.knee, ymax=0.95, ymin=0.05, color='brown', ls='--')
        print(f'Knee point, WCSS : {knee_elbow.knee, WCSS[knee_elbow.knee]}')
        plt.figtext(0.25, 0.25, f'optimal k = {knee_elbow.knee}', bbox=dict(facecolor='red', alpha=0.3), fontsize=12)

    # Silhouette scores plot
    plt.subplot(1, 2, 2)
    plt.plot(k_range, SS, color='darkgreen', marker='o')
    plt.xlabel('k', weight='bold', fontsize=10)
    plt.ylabel('Silhouette score', weight='bold', fontsize=10)
    plt.title(f'Silhouette score analysis for optimal k (init={init})\n', weight='bold', fontsize=12)
    knee_ss = KneeLocator(k_range, SS, S=10, online=True, curve='concave', direction='increasing', interp_method='interp1d')
    if knee_ss.knee!=None and knee_ss.knee in range(len(SS)):
        print(f'Knee point, SS : {knee_ss.knee, SS[knee_ss.knee]}')
        plt.figtext(0.75, 0.25, f'optimal k = {knee_ss.knee}', bbox=dict(facecolor='red', alpha=0.3), fontsize=12)
    fig1.tight_layout(pad=2.0)
    plt.show()
    return knee_elbow.knee, knee_ss.knee

def km_final_fit(data_df: pd.DataFrame, opt_k: int):
    """
    Fit k-means model on the processed data and 
    plot all charts for the model cluster predictions 

    Parameters:
    -----------
    data : pd.DataFrame
        Dataframe containing processed data on which the k-means model is to be fit.

    opt_k : int
        Optimal 'k' value computed by find_optimal_k().

    Returns: 
    --------
    km_final_clusters_df : pd.DataFrame
        DF with cluster labels.
    """
    # Fit the K-means model on the data with the optimal 'k'
    k_final = opt_k # we get this value from either Elbow test or Silhouette score analysis
    km_final = KMeans(n_clusters = k_final, init='k-means++', max_iter=20000, n_init=100, random_state=1500)
    km_final = km_final.fit(data_df)
    km_final_clusters_df = pd.DataFrame(index=data_df.index, columns=['km_cluster'])
    km_final_clusters_df['km_cluster'] = km_final.labels_

    # TSNE
    print('\n------------------------------------\n')
    print('TSNE plot for the model')
    labels_df = plot_TSNE(data_df, km_final.labels_)
    # plot cluster count (bar chart)
    print('\n------------------------------------\n')
    print('Cluster counts bar chart')
    plot_cluster_counts(labels_df)
    # plot cluster members
    print('\n------------------------------------\n')
    print('Cluster member price movements: sample plots of 4 smallest clusters')
    plot_cluster_members(labels_df)
    return km_final_clusters_df

def find_and_fit_optimal_kmeans(final_features_df: pd.DataFrame, k_range=range(2, 25), init='k-means++'):
    print('-' * 50)
    print('Find the optimal k value for fitting K-means model\n')
    opt_k_elbow, opt_k_SS = find_optimal_k(final_features_df, k_range=k_range, init=init)
    print(f'Optimal k based on Elbow curve: {opt_k_elbow}\n')
    print(f'Optimal k based on Silhouette score analysis: {opt_k_SS}\n')

    km_cluster_df = None
    if opt_k_SS is not None or opt_k_elbow is not None:
        k_final = 0
        if opt_k_elbow is None:
            k_final = opt_k_SS
        elif opt_k_SS is None:
            k_final = opt_k_elbow
        else:
            k_final = min(opt_k_elbow, opt_k_SS)
        # fit K-means using the derived optimal k value
        km_cluster_df = km_final_fit(final_features_df, opt_k=k_final)
    else:
        raise Exception("No optimal value for k found!")

    return km_cluster_df

def optics_fit(data_df: pd.DataFrame, max_eps: float=np.inf, min_samples: int=5):
    """
    Fit OPTICS clustering algorithm over the data;
    Plot all charts for the model predictions  

    Parameters:
    -----------
    data_df : pd.DataFrame
        Dataframe containing processed data on which the density based OPTICS model is to be fit.

    max_eps : int
        The maximum distance between two samples for one to be considered as in the neighborhood of the other. 
        Default value of np.inf will identify clusters across all scales; reducing max_eps will result in shorter run times.

    min_samples : int, default is 5
        The number of samples in a neighborhood for a point to be considered as a core point. 
        Also, up and down steep regions can't have more than min_samples consecutive non-steep points. 
        Expressed as an absolute number or a fraction of the number of samples (rounded to be at least 2).

    Returns: 
    --------
    db_final_clusters_df : pd.DataFrame
        DF with computed cluster labels.
    """
    print('Running density based clustering - OPTICS model')
    print('--'*50)
    print(f'max_eps: {max_eps}\nmin_samples: {min_samples}')
    db = OPTICS(max_eps=max_eps, min_samples=min_samples, xi=0.04).fit(data_df)
    labels = db.labels_
    print(f'model fit params: {db.get_params()}\n')
    # Compute Silouette score
    sil_score = silhouette_score(data_df, labels=labels, random_state=1500)
    print(f'\nThe Silhouette Score for our OPTICS model fit on preprocessed & scaled data : {sil_score}')
    # Compute Calinski Harabasz score
    cal_hb_score = calinski_harabasz_score(data_df, labels=labels)
    print(f'\nThe Calinski Harabasz Score for our OPTICS model fit on preprocessed & scaled data : {cal_hb_score}')
    
    # TSNE
    print('--'*100)
    print('TSNE plot for the model')
    labels_df = plot_TSNE(data_df, labels)
    # plot cluster count (bar chart)
    print('--'*100)
    print('Cluster counts bar chart')
    plot_cluster_counts(labels_df)
    # plot cluster members
    print('--'*100)
    print('Cluster member price movements: sample plots of 4 least dense clusters')
    plot_cluster_members(labels_df)

    db_final_clusters_df = pd.DataFrame(index=data_df.index, columns=['db_cluster'])
    db_final_clusters_df['db_cluster'] = labels
    return db_final_clusters_df

def agg_hc_fit(data_df: pd.DataFrame, n_clusters: int=10, linkage: str='average'):
    """
    Fit Agglomerative Clustering algorithm over the data;
    Plot all charts for the model predictions 

    Parameters:
    -----------
    data_df : pd.DataFrame
        Dataframe containing processed data on which Hierarchical Agglomerative Clustering model is to be fit.

    n_clusters : int, default is 10
        The number of clusters to find.

    linkage : str, default is 'average'
        {'ward', 'complete', 'average', 'single'}, 
        Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation. 
        The algorithm will merge the pairs of cluster that minimize this criterion.

            1. 'ward' minimizes the variance of the clusters being merged.

            2. 'average' uses the average of the distances of each observation of the two sets.

            3. 'complete' or 'maximum' linkage uses the maximum distances between all observations of the two sets.

            4. 'single' uses the minimum of the distances between all observations of the two sets.

    Returns: 
    --------
    hc_final_clusters_df : pd.DataFrame
        DF with computed cluster labels.
    """
    print('Running hierarchical clustering (bottom-up approach) - Agglomerative model')
    print('--'*50)
    print(f'n_clusters: {n_clusters}')
    hc = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage=linkage).fit(data_df)
    labels = hc.labels_
    print(f'model fit params: {hc.get_params()}\n')
    # Compute Silouette score
    sil_score = silhouette_score(data_df, labels=labels, random_state=1500)
    print(f'\nThe Silhouette Score for our Agglomerative model fit on preprocessed & scaled data : {sil_score}')
    # Compute Calinski Harabasz score
    cal_hb_score = calinski_harabasz_score(data_df, labels=labels)
    print(f'\nThe Calinski Harabasz Score for our Agglomerative model fit on preprocessed & scaled data : {cal_hb_score}')
    
    # TSNE
    print('--'*100)
    print('TSNE plot for the model')
    labels_df = plot_TSNE(data_df, labels)
    # plot cluster count (bar chart)
    print('--'*100)
    print('Cluster counts bar chart')
    plot_cluster_counts(labels_df)
    # plot cluster members
    print('--'*100)
    print('Cluster member price movements: sample plots of 4 least dense clusters')
    plot_cluster_members(labels_df)

    hc_final_clusters_df = pd.DataFrame(index=data_df.index, columns=['hc_cluster'])
    hc_final_clusters_df['hc_cluster'] = labels
    return hc_final_clusters_df

def compare_clustering_results(km_cluster_df: pd.DataFrame, db_cluster_df: pd.DataFrame, hc_cluster_df: pd.DataFrame, sp500_stocks_profile_df: pd.DataFrame):
    all_clustering_df = pd.concat([km_cluster_df, db_cluster_df, hc_cluster_df], axis=1)
    all_clustering_df['sector'] = [sp500_stocks_profile_df[sp500_stocks_profile_df['ticker'] == tkr]['sector'].values[0] for tkr in all_clustering_df.index]
    all_clustering_df = all_clustering_df[['sector', 'km_cluster', 'db_cluster', 'hc_cluster']]
    
    
    print("Head of combined clustering dataframe:")
    print(all_clustering_df.head(15))
    print("\nSector-wise breakdown for k-means clusters:")
    print(pd.crosstab(all_clustering_df.sector, all_clustering_df.km_cluster).style.highlight_max(color='orange', axis=0))
    
    print("\nSector-wise breakdown for OPTICS clusters:")
    print(pd.crosstab(all_clustering_df.sector, all_clustering_df.db_cluster).style.highlight_max(color='green', axis=0))
    
    print("\nSector-wise breakdown for Agglo clusters:")
    print(pd.crosstab(all_clustering_df.sector, all_clustering_df.hc_cluster).style.highlight_max(color='indigo', axis=0))

def get_final_clabel_profile_df(selected_label_df: pd.DataFrame, target_ratios: list):
    """
    Returns a final cluster profile dataframe containing the chosen cluster labels, 
    mean ratios data, sector and company info. 
    Mainly intended for EDA and drawing insights.

    Parameters:
    -----------
    selected_label_df : pd.DataFrame
        Cluster label DF for a selected clustering method.

    target_ratios: list
        Selecting the relevant ratios.

    Returns: 
    --------
    final_cluster_df : pd.DataFrame
        DF with computed cluster labels and stock profile details appended to it.
    """  
    conn = sqlite3.connect('./data/pairs_trading.db')
    sp500_stocks_profile_df = pd.read_sql_query("SELECT * FROM stocks_profile", conn)
    stock_ratios_df = pd.read_sql_query("SELECT * FROM stocks_hist_ratios", conn)

    conn.close()
    final_cluster_df = pd.DataFrame(index=selected_label_df.index, columns=['company_name', 'sector', 'cluster'])
    final_cluster_df['company_name'] = [sp500_stocks_profile_df[sp500_stocks_profile_df.ticker == tkr]['company_name'].values[0] for tkr in final_cluster_df.index]
    final_cluster_df['sector'] = [sp500_stocks_profile_df[sp500_stocks_profile_df.ticker == tkr]['sector'].values[0] for tkr in final_cluster_df.index]
    final_cluster_df['cluster'] = [selected_label_df.loc[tkr][0] for tkr in final_cluster_df.index]
    final_cluster_df['bus_desc'] = [sp500_stocks_profile_df[sp500_stocks_profile_df.ticker == tkr]['business_desc'].values[0] for tkr in final_cluster_df.index]
    # fetch the mean target ratio measures for all stocks
    temp_rat = stock_ratios_df[['ticker', 'Fiscal Date Ending'] + target_ratios].iloc[:, 2:].apply(pd.to_numeric, errors='coerce', downcast='float')
    temp_rat = temp_rat.fillna(0)
    temp_rat = pd.concat([stock_ratios_df[['ticker', 'Fiscal Date Ending']], temp_rat], axis=1)
    temp_rat = temp_rat[temp_rat['Fiscal Date Ending'] < 2020]
    temp_rat = temp_rat.reset_index(drop=True)
    ratios_10y_mean_df = temp_rat.drop(columns=['Fiscal Date Ending']).groupby(by=['ticker']).mean()
    
    final_cluster_df = pd.concat([final_cluster_df, ratios_10y_mean_df], axis=1, join='inner')
    
    print("\nSector-wise distribution among the OPTICS clusters:")
    print(pd.crosstab(final_cluster_df.sector, final_cluster_df.cluster).style.highlight_max(color='darkgreen', axis=0))
    return final_cluster_df

def boxplot_cluster_fin_ratios(data: pd.DataFrame, target_ratios: list):
    """
    Plot the boxplots of all target final ratios for each cluster.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DF with computed cluster labels and stock profile details appended to it. 
        This is output by get_final_clabel_profile_df()

    target_ratios : list
        List of target ratios.
    """
    plt.figure(figsize=(18, 24))
    plt.subplots_adjust(hspace=.5, wspace=.25)
    plt.clf()
    counter = 0
    for row in range(5):
        for col in range(2):
            if counter <len(data.cluster.value_counts()):
                plt.subplot(5, 2, counter+1)
                sns.boxplot(y=data[target_ratios[counter]], x=data.cluster)
            plt.title(f'Box plot of {target_ratios[counter]}\n', weight='bold', fontsize=14)
            plt.xlabel('Clusters', weight='bold', fontsize=12)
            plt.ylabel(f'{target_ratios[counter]}', weight='bold', fontsize=12)
            counter += 1
    plt.tight_layout(pad=2.0)
    plt.show()

def plot_cluster_wordclouds(data_df: pd.DataFrame):
    """
    Plot wordclouds of business descriptions of companies in each cluster

    Parameters:
    -----------
    data_df : pd.DataFrame
        DF with computed cluster labels and stock profile details appended to it. 
        This is output by get_final_clabel_profile_df()
    """
    # Plot wordclouds within each cluster
    print('-'*50)
    print('Plot wordclouds of business descriptions of companies in each cluster\n')
    
    fig1 = plt.figure(figsize=(18, 20))
    fig1.subplots_adjust(hspace=.25, wspace=.05)
    plt.axis('off')
    plt.clf()
    cluster_labels = data_df.cluster.value_counts().index
    cluster_labels = [label for label in cluster_labels if label != -1]
    for i in range(len(cluster_labels)):
        fig1.add_subplot(5, 3, i+1)
        wc = WordCloud(background_color='black', width=700, height=400, max_words=400, min_font_size=16, max_font_size=50, random_state=1500)
        full_text = '; '.join(data_df[data_df['cluster'] == cluster_labels[i]]['bus_desc'])
        clean_text = text_cleaning(full_text)
        wc = wc.generate(str(clean_text))
        plt.axis('off')
        plt.imshow(wc, cmap=cm.Dark2_r)
        plt.title(f'Top words in cluster {cluster_labels[i]}', weight='bold', fontsize=16)
        i += 1
    plt.tight_layout(pad=2.0)
    plt.show()
    
    
def filter_stock_profiles(sp500_comp_profile_df: pd.DataFrame):
    # Filter out only the necessary information
    filtered_stock_profiles = sp500_comp_profile_df[['ticker', 'company_name', 'sector', 'sub_industry', 'business_desc']]
    
    # Check for any NaN values
    print(f'Total count of Nan values in stock profile data: {filtered_stock_profiles.isna().sum().sum()}')
    print(filtered_stock_profiles.isna().sum())

    # Drop tickers with missing values in business_desc column only
    filtered_stock_profiles.dropna(axis=0, subset=['business_desc'], inplace=True)
    print(f'Total stocks in consideration after dropping the ones with missing business descriptions: {filtered_stock_profiles.shape[0]}')

    return filtered_stock_profiles

    
def preprocess_text(text, flg_stemm=False, flg_lemm=True):
    """
    Clean & preprocess input string.
    Note:   You should not set both flg_stemm & flg_lemm to be True. 
            Only one of them can be True at a time.
    """
    # clean (convert to lowercase and remove punctuations and special characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    # Tokenize (convert from string to list)
    tokens = text.split()

    # remove Stopwords
    stop_words.extend(['founded', 'firm', 'company', 'llc', 'inc', 'incorporated', 
                       'multinational', 'corporation', 'commonly', 'headquartered']) # adding more words to the default list
    if stop_words is not None:
        tokens = [word for word in tokens if word not in stop_words]
                
    # Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        tokens = [ps.stem(word) for word in tokens]
                
    # Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        tokens = [lem.lemmatize(word) for word in tokens]
            
    # back to string from list
    text = ' '.join(tokens)
    return text


def apply_preprocessing(stock_profiles: pd.DataFrame):
    stock_profiles['bd_clean'] = stock_profiles['business_desc'].apply(lambda x: preprocess_text(x))
    return stock_profiles


def count_unique_words(stock_profiles: pd.DataFrame):
    count = len(set([ele for arr in list(stock_profiles["bd_clean"].apply(lambda x: str(x).split(" "))) for ele in arr]))
    print(f'Total count of unique words (unigrams) found in our corpus of business descriptions across all stocks: {count}')
    return count


def tfidf_transform(stock_profiles: pd.DataFrame, min_df=1, max_df=0.1, ngram_range=(2, 4)):
    tfidf = feature_extraction.text.TfidfVectorizer(stop_words='english', max_features=10000, min_df=min_df, max_df=max_df,
                                                    strip_accents='unicode', ngram_range=ngram_range)
    tfidf_sparse = tfidf.fit_transform(stock_profiles['bd_clean'])
    return tfidf_sparse


def visualize_sparse_matrix(tfidf_sparse):
    sns.heatmap(tfidf_sparse.todense()[:, np.random.randint(0, tfidf_sparse.shape[1], 10)] == 0, vmin=0, vmax=1, cbar=False)\
        .set_title('Sparse Matrix Sample')




