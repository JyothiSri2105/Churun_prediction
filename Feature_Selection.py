import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
import sys
import seaborn as sns

from scipy.stats import yeojohnson
from seaborn import boxplot
from scipy import stats
from scipy.stats import boxcox


#modularization
import logging
from logging_code import setup_logging
logger = setup_logging('Feature_Selection')

import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr

def filter_method(X_train_nums_cols,X_test_nums_cols,y_train,y_test):
    try:
        logger.info(f'before applying Feature Selection Train numerical columns and shapes  \n : {X_train_nums_cols.columns} : {X_train_nums_cols.shape}')
        logger.info( f'before applying Feature Selection  Test numerical columns and shapes  \n : {X_test_nums_cols.columns} : {X_test_nums_cols.shape}')
        #applying consant techinque threshold is 0
        logger.info('==============constant techinque t=0=================')

        var = VarianceThreshold(threshold=0)
        var.fit(X_train_nums_cols)
        logger.info(f'checking the good columns in train {sum(var.get_support())}:Good columns {X_train_nums_cols.columns[var.get_support()]}')
        logger.info(f'checking the bad columns in train {sum(~var.get_support())}:bad columns {X_train_nums_cols.columns[~var.get_support()]}')
        logger.info(f'checking the good columns in test {sum(var.get_support())}:Good columns {X_test_nums_cols.columns[var.get_support()]}')
        logger.info(f'checking the bad columns in test {sum(~var.get_support())}:bad columns {X_test_nums_cols.columns[~var.get_support()]}')

        # applying quesi consant techinque threshold is 0.01
        logger.info('==============quesi constant techinque=================')

        var = VarianceThreshold(threshold=0.01)
        var.fit(X_train_nums_cols)
        logger.info(
            f'checking the good columns in train {sum(var.get_support())}:Good columns {X_train_nums_cols.columns[var.get_support()]}')
        logger.info(
            f'checking the bad columns in train {sum(~var.get_support())}:bad columns {X_train_nums_cols.columns[~var.get_support()]}')
        logger.info(
            f'checking the good columns in test {sum(var.get_support())}:Good columns {X_test_nums_cols.columns[var.get_support()]}')
        logger.info(
            f'checking the bad columns in test {sum(~var.get_support())}:bad columns {X_test_nums_cols.columns[~var.get_support()]}')

        # Applying the Hypothesis Testing Technique
        # Applying the pearson Technique
        logger.info(f'the X_train values {X_train_nums_cols.shape}')
        logger.info(f'the y_train values {y_train.shape}')

        c = []
        for i in X_train_nums_cols.columns:
            result = pearsonr(X_train_nums_cols[i], y_train)
            c.append(result)

        t = np.array(c)

        p_values = pd.Series(t[:, 1], index=X_train_nums_cols.columns)
        logger.info(f'pvalues:{p_values}')
        p = 0 #1
        f = []

        for i in p_values:
            if i < 0.05:
                f.append(X_train_nums_cols.columns[p])
            p = p + 1

        logger.info(f'Checking the Good Columns: {f}')

        return X_train_nums_cols,X_test_nums_cols
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')
