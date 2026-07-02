import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os
import sys

import logging
from logging_code import setup_logging
logger = setup_logging('Handling_missing_values')

def handling_missing_values(X_train,X_test):
    try:
        logger.info(f'Before replacing Train null values : {X_train.isnull().sum()}')
        logger.info(f'Before replacing Test null values : {X_test.isnull().sum()}')
        logger.info(f'Before replacing Train column  {X_train.columns}')
        logger.info(f'Before replacing Train column shape {X_train.shape}')
        logger.info(f'Before replacing Test column shape {X_test.shape}')


        for i in X_train.columns:
            if X_train[i].isnull().sum()>0:
                X_train[i+'hmv']=X_train[i].copy()
                X_test[i + 'hmv'] = X_test[i].copy()

                X_train[i + 'hmv'] = X_train[i + 'hmv'].fillna(X_train[i + 'hmv'].mode()[0])
                X_test[i + 'hmv'] =  X_test[i + 'hmv'].fillna( X_test[i + 'hmv'].mode()[0])

                X_train = X_train.drop([i],axis=1)
                X_test = X_test.drop([i],axis=1)

        logger.info(f'After replacing Train null values : {X_train.isnull().sum()}')
        logger.info(f'After replacing Test null values : {X_test.isnull().sum()}')
        logger.info(f'After replacing Train column  {X_train.columns}')
        logger.info(f'After replacing Train column shape {X_train.shape}')
        logger.info(f'After replacing Test column shape {X_test.shape}')
        return X_train,X_test

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')

