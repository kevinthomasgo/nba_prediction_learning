# Standard libs
import math
import logging
import re
import json
import itertools
from io import StringIO
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns
import progressbar
from importlib import reload  

# Science libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

# ML specific
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Extras
import string
import requests
import datetime
import progressbar
import time
import lxml

# Viz
TM_pal_categorical_3 = ['#ef4631', '#10b9ce', '#ff9138']

def quick_checker(x):
    """
    Generates some summary stats for a column.
    
    """
    percent_na_x = round(sum(x.isna())/len(x)*100,0)
    min_x = x.min()
    max_x = x.max()
    val_counts_x = x.value_counts().iloc[0:5].to_dict()
    return(
        {
            'percent_na': percent_na_x,
            'min_x': min_x,
            'max_x': max_x,
            'val_counts_x': val_counts_x
        }
    )

def quick_checker_all(df_raw):
    """
    Reuns quick_checker() on all columns of the given df.
    
    """
    results = df_raw.apply(lambda x: quick_checker(x), axis=0)
    results_df = pd.DataFrame([i for i in results.values])
    results_df.insert(0, 'column', results.index)
    return(results_df)


def time_splitter(df_features_transformed, cutoff_date_train, cutoff_date_valid, cutoff_date_test):
    """
    Split data into train, validation, and test sets.
    
    Inputs:
    df_features_transformed: (pandas df) Dataframe that includes (transformed) features.
    features: (list) List of columns to be considered as features.
    index_list: (list) List of columns to be considered as indices.
    cutoff_date_train: (string) First date of train set.
    cutoff_date_valid: (string) Last date of test set, first date of validation set.
    cutoff_date_test: (string) Last date of train set, first date of test set.
    
    
    """
    
    train = df_features_transformed[(df_features_transformed['year_month_day']>=cutoff_date_train) & (df_features_transformed['year_month_day']<cutoff_date_valid)]
    valid = df_features_transformed[(df_features_transformed['year_month_day'] >= cutoff_date_valid) & (df_features_transformed['year_month_day']<cutoff_date_test)]
    test = df_features_transformed[df_features_transformed['year_month_day']>=cutoff_date_test]
    
    return(train, valid, test)



def feature_transformer(df_features, features_list, index_list, scaler=None):
    """
    Transforms features (e.g., scaling by mean-centering and standardization).
    
    Inputs:
    df_features: (pandas df) Dataframe that includes columns to be used as features.
    features_list: (list) List of columns to be considered as features.
    index_list: (list) List of columns to be considered as index. Useful for filtering later.
    scaler: (scaler) Fitted sklearn scaler for scaling features. If none is provided, a new scaler is fit. 
    """
    
    # Set index for later
    df_features = df_features.set_index(index_list)

    # If no scaler is provided, fit a new one
    if scaler is None:
#         print("No scaler provided. Fitting scaler...")
        scaler = StandardScaler()
        scaler.fit(df_features[features_list])
        transformed_features = pd.DataFrame(scaler.transform(df_features[features_list]))
        transformed_features.columns = features_list
        transformed_features.index = df_features.index
        df_features[features_list] = transformed_features
    # If scaler is provided, use it
    else:
#         print("Scaler provided. Using scaler.")
        transformed_features = pd.DataFrame(scaler.transform(df_features[features_list]))
        transformed_features.columns = features_list
        transformed_features.index = df_features.index
        df_features[features_list] = transformed_features
        
    return(df_features, scaler)



def X_y_splitter(df, features, y_var):
    """
    Splits df into X and y.
    
    Inputs:
    df (pandas df):
    features (list): List of columns in df to be considered features
    y_var (string): Variable name of dependent variable
    
    """
    
    X = df[features]
    y = df.reset_index()[y_var]
    y.index = X.reset_index()['index']
    
    return(X, y)


def top_features(df, features, top_n, y_var):
    """
    Returns top_n features based on F-values from ANOVA.
    
    """
    # Get X and y from df
    X , y = X_y_splitter(df, features, y_var)

    f_values = SelectKBest(f_classif, k=top_n).fit(X, y).scores_
    f_values = pd.DataFrame(
        {
            'feature': X.columns,
            'f_values': f_values
        }
    )
    f_values.sort_values('f_values', ascending=False, inplace=True)
    f_values.head()
    top_n_features = f_values['feature'][0:top_n].tolist()
    return(top_n_features)



def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the decision threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]



def threshold_finder(y_pred_proba, expected_event_rate):
    """
    Finds "best" decision threshold. Best decision threshold is defined as the threshold that
    minimizes difference between expected and predicted proportion of positive events (1's, wins, etc).
    
    Inputs:
    y_pred_proba (Series of floats): probability scores
    expected_event_rate (float): Expected event rate of the positive event. Ranges between [0, 1]
    
    """
    
    # Find best threshold
    thresholds = {}
    for i in np.arange(0.10,0.91,0.01):
        y_pred_i = adjusted_classes(y_pred_proba, i)
        prop_i = sum(y_pred_i)/len(y_pred_i)
        thresholds[i] = prop_i

    # Get diff in proportion vs expected for each threshold
    thresholds = pd.DataFrame(thresholds, index=[0]).T
    thresholds.reset_index(inplace=True)
    thresholds.columns = ['threshold', 'proportion_wins']
    thresholds['expected_prop'] = expected_event_rate
    thresholds['diff'] = thresholds['proportion_wins'] - thresholds['expected_prop']

    # Get best threshold (min distance from expected)
    best_threshold = thresholds.iloc[abs(thresholds['diff']).idxmin()]['threshold']
    
    return(best_threshold, thresholds)


def scorer(X, y, estimator, expected_event_rate):
    """
    Returns prediction scores and tags across default and "best" decision thresholds. 
    
    Inputs:
    X: (pandas df) Dataframe of independent variables for prediction.
    y: (array) Array of correct observations.
    estimator: (estimator type) Fitted estimator. Must have .predict() and .predict_proba() methods!
    expected_event_rate (float): Expected event rate of the positive event. Ranges between [0, 1]
    
    """
    
    # Get scores and "default" predictions
    y_pred_proba = [i[1] for i in estimator.predict_proba(X)]
    y_pred = estimator.predict(X)
    
    # Find best threshold
    best_threshold, thresholds = threshold_finder(y_pred_proba, expected_event_rate)
    
    # Compile predictions
    predictions_df = pd.DataFrame(
        {
            'pred_proba': y_pred_proba,
            'pred_default_threshold': y_pred,
            'pred_best_threshold': adjusted_classes(y_pred_proba, best_threshold)
        }
    )
    predictions_df.index = y.index
    predictions_df['actual'] = y
    predictions_df['actual'].astype('int')
    
    return(predictions_df, thresholds)



def safe_divide(x,y):
    """
    Returns "nan" if dividing by zero.
    """
    try:
        return(x/y)
    except ZeroDivisionError:
        return np.nan
    
    

def plot_probability(predictions_df_test, size_multiplier=10):
    """
    Plots probability of each probability score bin actually happening.
    
    """
    # Make bins for prediction assessment
    predictions_df_test['bin'] = pd.cut(predictions_df_test['pred_proba'],np.arange(0,1.1,0.05), labels=False)
    predictions_df_test['interval'] = predictions_df_test['bin']/20
    
    # Get combos
    intervals = list(np.sort(predictions_df_test['interval'].unique()))
    seasons = list(np.sort(predictions_df_test['season'].unique()))
    combos_df = pd.DataFrame({'season':seasons, 'key':1}).merge(pd.DataFrame({'interval':intervals, 'key':1}), on='key').drop(columns='key')

    # Make 
    bar = progressbar.ProgressBar(maxval=len(combos_df))
    bar.start()

    for_plotting = pd.DataFrame()
    for i in range(len(combos_df)): 
        season_i = combos_df.loc[i,'season']
        bin_i = predictions_df_test[(predictions_df_test['interval']==combos_df.loc[i,'interval']) & (predictions_df_test['season']==season_i)]
        actual_prob = safe_divide(len(bin_i[bin_i['home_pts']-bin_i['away_pts']>0]),len(bin_i))
        count_i = len(bin_i)*size_multiplier # Scale n_games to increase size of points later
        for_plotting = for_plotting.append(pd.DataFrame({
            'season':season_i,
            'predicted_prob':combos_df.loc[i,'interval'],
            'actual_prob':actual_prob,
            'n_games': count_i
        },index=[i]))
        bar.update(i)
    for_plotting = for_plotting.dropna()
    
    # Plot
    fig,ax = plt.subplots()
    fig.set_dpi(120)
    fig.set_size_inches(10,7.5)
    for i in seasons:
        for_plotting_i = for_plotting[for_plotting['season']==i]
        plt.scatter(x='actual_prob', y='predicted_prob', data=for_plotting_i, alpha=0.5, color=TM_pal_categorical_3[0], s='n_games')
    x = np.linspace(0,1)
    plt.plot(x,x,linestyle='dashed',color='grey')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.ylabel('Predicted probability of event')
    plt.xlabel('Actual probability of event')
    
    return(for_plotting)



def plot_precision_recall(y_true, y_pred_proba):
    """
    Plots precision and recall across different threshold values.
    
    """
    
    # Make df
    for_plotting = {}
    for i in np.arange(0.10, 0.91, 0.01):
        y_pred = adjusted_classes(y_pred_proba, i)
        for_plotting[i] = {
            'precision': classification_report(y_true, y_pred, output_dict=True)['1.0']['precision'],
            'recall': classification_report(y_true, y_pred, output_dict=True)['1.0']['recall']
        }
    for_plotting = pd.DataFrame(for_plotting).T.reset_index()
    for_plotting.rename(columns={'index':'decision_threshold'}, inplace=True)
    for_plotting.head()

    # Plot
    fig,ax = plt.subplots()
    fig.set_dpi(120)
    fig.set_size_inches(10,7.5)
    plt.plot(for_plotting['decision_threshold'], for_plotting['precision'], alpha=1, color=TM_pal_categorical_3[0])
    plt.plot(for_plotting['decision_threshold'], for_plotting['recall'], alpha=1, color=TM_pal_categorical_3[1])
    plt.ylabel('Value')
    plt.xlabel('Decision Threshold')
    plt.legend(['Precision', 'Recall'])
    
    return(for_plotting)


def plot_roc_curve(y_true, y_pred_proba):
    # Compute micro-average ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

    # Plot
    fig,ax = plt.subplots()
    fig.set_dpi(120)
    fig.set_size_inches(10,7.5)
    plt.plot(fpr, tpr, alpha=1, color=TM_pal_categorical_3[0])
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')