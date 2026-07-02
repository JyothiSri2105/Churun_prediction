from logging import info

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import os
import sys

# importings
from sklearn.model_selection import train_test_split

# modularization
import logging
from logging_code import setup_logging

logger = setup_logging('Outliers')

from scipy.stats.mstats import winsorize


def outlier(X_train_num_cols, X_test_num_cols):
    try:

        logger.info(f'===================  Outliers  =============================')
        logger.info(
            f'Before apply Train numerical columns and shapes outlier \n : {X_train_num_cols.columns} : {X_train_num_cols.shape}')
        logger.info(
            f'Before apply Test numerical columns and shapes oulier \n : {X_test_num_cols.columns} : {X_test_num_cols.shape}')

        cols = ['tenure_yeo', 'MonthlyCharges_box','TotalChargeshmv_yeo']

        for col in cols:
            # Applying the Trimming technique
            if col == 'tenure_yeo':
                iqr = X_train_num_cols[col].quantile(0.75) - X_train_num_cols[col].quantile(0.25)
                lower_limit = X_train_num_cols[col].quantile(0.25) - (1.5 * iqr)
                upper_limit = X_train_num_cols[col].quantile(0.75) + (1.5 * iqr)
                X_train_num_cols[col + '_tri'] = np.where(X_train_num_cols[col] < lower_limit, lower_limit,
                                                          np.where(X_train_num_cols[col] > upper_limit, upper_limit,
                                                                   X_train_num_cols[col]))
                X_test_num_cols[col + '_tri'] = np.where(X_test_num_cols[col] < lower_limit, lower_limit,
                                                         np.where(X_test_num_cols[col] > upper_limit, upper_limit,
                                                                  X_test_num_cols[col]))

            else:
                # Applying the Capping mean and Standard
                lower = X_train_num_cols[col].mean() - 3 * X_train_num_cols[col].std()
                upper = X_train_num_cols[col].mean() + 3 * X_train_num_cols[col].std()

                X_train_num_cols[col + '_CapMS'] = np.where(X_train_num_cols[col] < lower, lower,
                                                            np.where(X_train_num_cols[col] > upper, upper,
                                                                     X_train_num_cols[col]))
                X_test_num_cols[col + '_CapMS'] = np.where(X_test_num_cols[col] < lower, lower,
                                                           np.where(X_test_num_cols[col] > upper, upper,
                                                                    X_test_num_cols[col]))

        original_col = ['tenure_yeo', 'MonthlyCharges_box','TotalChargeshmv_yeo']
        replaced_col = ['tenure_yeo_tri', 'MonthlyCharges_box_CapMS', 'TotalChargeshmv_yeo_CapMS']

        '''
        plt.figure(figsize=(6, 4))
        for i,j in zip(original_col, replaced_col):
            sns.kdeplot(X_train_num_cols[i], label=i, color='blue')
            sns.kdeplot(X_train_num_cols[j], label=j, color='red')
            plt.title(f'Distribution: {original_col} vs {replaced_col}')
            plt.legend()
            plt.show()
        '''


        X_train_num_cols = X_train_num_cols.drop(['tenure_yeo'], axis=1)
        X_test_num_cols = X_test_num_cols.drop(['tenure_yeo'], axis=1)

        X_train_num_cols = X_train_num_cols.drop(['MonthlyCharges_box'], axis=1)
        X_test_num_cols = X_test_num_cols.drop(['MonthlyCharges_box'], axis=1)

        X_train_num_cols = X_train_num_cols.drop(['TotalChargeshmv_yeo'], axis=1)
        X_test_num_cols = X_test_num_cols.drop(['TotalChargeshmv_yeo'], axis=1)

        logger.info(
            f'After apply Train numerical columns and shapes outlier \n : {X_train_num_cols.columns} : {X_train_num_cols.shape}')
        logger.info(
            f'After apply Test numerical columns and shapes outlier \n : {X_test_num_cols.columns} : {X_test_num_cols.shape}')

        return X_train_num_cols, X_test_num_cols


    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')