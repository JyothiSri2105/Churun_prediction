import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
import sys
import seaborn as sns



#modularization
import logging
from logging_code import setup_logging
logger = setup_logging('Categorical_to_Numerical')

import warnings
warnings.filterwarnings('ignore')


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

def cat_to_num(X_train_cat_cols, X_test_cat_cols):
    try:
        logger.info(f'Before doing Train Categorical to numerical : {X_train_cat_cols.columns} \n : and its shape {X_train_cat_cols.shape}')
        logger.info(f'Before doing Test Categorical to numerical : {X_test_cat_cols.columns} \n : and its shape {X_test_cat_cols.shape}')



        # Applying the nominal encoder(OneHotEncoder)
        # Columns are ('gender', 'Partner', 'Dependents')

        logger.info(f'============= Applying the nominal encoder(OneHotEncoder) ========================= ')
        logger.info(f' Columns are gender Partner Dependents')

        one_hot = OneHotEncoder(drop='first')
        one_hot.fit(X_train_cat_cols[['gender','Partner','Dependents']]) # one hot coding -> same priority
        val_train = one_hot.transform(X_train_cat_cols[['gender','Partner','Dependents']]).toarray() #fit -> analyse & learn from the data
        val_test = one_hot.transform(X_test_cat_cols[['gender','Partner','Dependents']]).toarray()  # transform -> apply the rules -> to modify
                                                                                                #  why we use array -> tostore the mutliple itmes of the same data type under a single variable name
        t1 = pd.DataFrame(val_train)# it will convert into dataframes or excel
        t2 = pd.DataFrame(val_test)

        t1.columns = one_hot.get_feature_names_out()#is a method that returns the names of the output features (columns) created after a transformation.
        t2.columns = one_hot.get_feature_names_out() # it output will be in ex:Gender_male,Gender_female

        X_train_cat_cols.reset_index(drop=True, inplace=True) # it will get index to 0,1,2,3....
        X_test_cat_cols.reset_index(drop=True, inplace=True) # it is old data

        t1.reset_index(drop=True, inplace=True) #0,1,2,3....
        t2.reset_index(drop=True, inplace=True) # it is data after transform

        X_train_cat_cols = pd.concat([X_train_cat_cols,t1],axis=1)# it will combine old data and new data
        X_test_cat_cols = pd.concat([X_test_cat_cols,t2],axis=1)

        X_train_cat_cols = X_train_cat_cols.drop(['gender','Partner','Dependents'],axis=1)# it remove the old columns on the which we perform tranform function is applied
        X_test_cat_cols = X_test_cat_cols.drop(['gender','Partner','Dependents'],axis=1)


        logger.info(f'Before doing Train Categorical to numerical : {X_train_cat_cols.columns} \n : and its shape {X_train_cat_cols.shape}')
        logger.info(f'Before doing Test Categorical to numerical : {X_test_cat_cols.columns} \n : and its shape {X_test_cat_cols.shape}')


        # applying the OrdinalEncoder
        logger.info(f'====================  OrdinalEncoder  =======================')

        ordinal_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Sim']

        od = OrdinalEncoder()
        od.fit(X_train_cat_cols[['PhoneService','MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup','DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies','Contract', 'PaperlessBilling', 'PaymentMethod', 'Sim']])

        results_train = od.transform(X_train_cat_cols[['PhoneService','MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup','DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies','Contract', 'PaperlessBilling', 'PaymentMethod', 'Sim']])
        results_test = od.transform(X_test_cat_cols[['PhoneService','MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup','DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies','Contract', 'PaperlessBilling', 'PaymentMethod', 'Sim']])

        p1 = pd.DataFrame(results_train, columns=[c+"_ordinal" for c in ordinal_cols])
        p2 = pd.DataFrame(results_test, columns=[c+"_ordinal" for c in ordinal_cols])
        ''' 
            columns = []
                  for i in ordinal_cols:
                      a = results_train[c+'oridnal']
                      c.append(a)
        '''
        #p1.columns = od.get_feature_names_out()+"_ordinal"
        #p2.columns = od.get_feature_names_out()+"_ordinal"

        p1.reset_index(drop=True, inplace=True) #reset the index of updated data set
        p2.reset_index(drop=True, inplace=True)

        X_train_cat_cols = pd.concat([X_train_cat_cols, p1], axis=1)
        X_test_cat_cols = pd.concat([X_test_cat_cols, p2], axis=1)

        X_train_cat_cols = X_train_cat_cols.drop(['PhoneService','MultipleLines', 'InternetService',
                                                  'OnlineSecurity', 'OnlineBackup','DeviceProtection',
                                                  'TechSupport', 'StreamingTV', 'StreamingMovies','Contract',
                                                  'PaperlessBilling', 'PaymentMethod', 'Sim'], axis=1)
        X_test_cat_cols = X_test_cat_cols.drop(['PhoneService','MultipleLines', 'InternetService',
                                                'OnlineSecurity', 'OnlineBackup','DeviceProtection',
                                                'TechSupport', 'StreamingTV', 'StreamingMovies','Contract',
                                                'PaperlessBilling', 'PaymentMethod', 'Sim'], axis=1)
        logger.info(f"After Odinal X_train_cat Column : {X_train_cat_cols.shape} : \n : {X_train_cat_cols.columns}")
        logger.info(f"After Odinal X_test_cat Column : {X_test_cat_cols.shape} : \n : {X_test_cat_cols.columns}")

        return X_train_cat_cols, X_test_cat_cols

    except Exception as e:
      er_type, er_msg, er_line = sys.exc_info()
      logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')