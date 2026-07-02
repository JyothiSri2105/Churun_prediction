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
logger = setup_logging('Variable_Transformation')

import warnings
warnings.filterwarnings('ignore')

def Var_Transforamtion(X_train_nums_cols,X_test_nums_cols):
    try:
        logger.info(f'===================  Variable Transformation  =============================')
        logger.info(f'Before apply Train numerical columns and shapes variable Transformation \n : {X_train_nums_cols.columns} : {X_train_nums_cols.shape}')
        logger.info(f'Before apply Test numerical columns and shapes variable Transformation \n : {X_test_nums_cols.columns} : {X_test_nums_cols.shape}')

        # tenure
        # applying the yeojohnson technique
        X_train_nums_cols['tenure_yeo'], lambda_yeo = yeojohnson(X_train_nums_cols['tenure']) # in this tech lambda valu is main
        X_test_nums_cols['tenure_yeo'] = yeojohnson(X_test_nums_cols['tenure'], lmbda=lambda_yeo)
        logger.info(f' lambda value for tenure column is {lambda_yeo}')

        '''
        plt.figure(figsize=(8, 3))
        plt.subplot(1, 4, 1)
        plt.title('Normal Distribution')
        X_train_nums_cols['tenure'].plot(kind='kde', color='r')
        X_train_nums_cols['tenure_yeo'].plot(kind='kde', color='black')
        plt.subplot(1, 4, 2)
        sns.boxplot(x=X_train_nums_cols['tenure'])
        sns.boxplot(x=X_train_nums_cols['tenure_yeo'])
        plt.subplot(1, 4, 3)
        plt.title("Probplot - Original")
        stats.probplot(X_train_nums_cols['tenure'], dist="norm", plot=plt)
        plt.subplot(1, 4, 4)
        plt.title("Probplot - Changed")
        stats.probplot(X_train_nums_cols['tenure_yeo'], dist="norm", plot=plt)
        plt.legend()
        plt.show()
        '''

        X_train_nums_cols = X_train_nums_cols.drop(['tenure'], axis=1)
        X_test_nums_cols = X_test_nums_cols.drop(['tenure'], axis=1)

        # MonthlyCharges
        # applying the boxcox technique
        X_train_nums_cols['MonthlyCharges_box'], lambda_box1 = boxcox(X_train_nums_cols['MonthlyCharges'] + 1)
        X_test_nums_cols['MonthlyCharges_box'] = boxcox(X_test_nums_cols['MonthlyCharges'] + 1, lmbda=lambda_box1)

        logger.info(f' lambda value for MonthlyCharges column is {lambda_box1}')

        '''
        plt.figure(figsize=(8, 3))
        plt.subplot(1, 4, 1)
        plt.title('Normal Distribution')
        X_train_nums_cols['MonthlyCharges'].plot(kind='kde', color='r')
        X_train_nums_cols['MonthlyCharges_box'].plot(kind='kde', color='black')
        plt.subplot(1, 4, 2)
        sns.boxplot(x=X_train_nums_cols['MonthlyCharges'])
        sns.boxplot(x=X_train_nums_cols['MonthlyCharges_box'])
        plt.subplot(1, 4, 3)
        plt.title("Probplot - Original")
        stats.probplot(X_train_nums_cols['MonthlyCharges'], dist="norm", plot=plt)
        plt.subplot(1, 4, 4)
        plt.title("Probplot - Changed")
        stats.probplot(X_train_nums_cols['MonthlyCharges_box'], dist="norm", plot=plt)
        plt.legend()
        plt.show()
        '''

        X_train_nums_cols = X_train_nums_cols.drop(['MonthlyCharges'], axis=1)
        X_test_nums_cols = X_test_nums_cols.drop(['MonthlyCharges'], axis=1)

        # TotalCharges_rep
        # applying the yeojohnson technique
        X_train_nums_cols['TotalChargeshmv_yeo'], lambda_yeo1 = yeojohnson(X_train_nums_cols['TotalChargeshmv'])
        X_test_nums_cols['TotalChargeshmv_yeo'] = yeojohnson(X_test_nums_cols['TotalChargeshmv'], lmbda=lambda_yeo1)

        logger.info(f' lambda value for TotalChargeshmv column is {lambda_yeo1}')

        '''
        plt.figure(figsize=(8, 3))
        plt.subplot(1, 4, 1)
        plt.title('Normal Distribution')
        X_train_nums_cols['TotalChargeshmv'].plot(kind='kde', color='r')
        X_train_nums_cols['TotalChargeshmv_yeo'].plot(kind='kde', color='black')
        plt.subplot(1, 4, 2)
        sns.boxplot(x=X_train_nums_cols['TotalChargeshmv'])
        sns.boxplot(x=X_train_nums_cols['TotalChargeshmv_yeo'])
        plt.subplot(1, 4, 3)
        plt.title("Probplot - Original")
        stats.probplot(X_train_nums_cols['TotalChargeshmv'], dist="norm", plot=plt)
        plt.subplot(1, 4, 4)
        plt.title("Probplot - Changed")
        stats.probplot(X_train_nums_cols['TotalChargeshmv_yeo'], dist="norm", plot=plt)
        plt.legend()
        plt.show()
        '''


        X_train_nums_cols = X_train_nums_cols.drop(['TotalChargeshmv'], axis=1)
        X_test_nums_cols = X_test_nums_cols.drop(['TotalChargeshmv'], axis=1)

        logger.info(f'After apply Train numerical columns and shapes variable Transformation \n : {X_train_nums_cols.columns} : {X_train_nums_cols.shape}')
        logger.info(f'After apply Test numerical columns and shapes variable Transformation \n : {X_test_nums_cols.columns} : {X_test_nums_cols.shape}')

        return X_train_nums_cols, X_test_nums_cols

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')
