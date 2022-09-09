import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# train test split from sklearn
from sklearn.model_selection import train_test_split
# imputer from sklearn
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import os



def impute_age (train, validate, test):
    '''imputes the mean age of train to all data sets'''
    imputer = SimpleImputer(strategy = "mean", missing_values = np.nan)
    imputer = imputer.fit(train[["age"]])
    train[["age"]] = imputer.transform(train[["age"]])
    validate[["age"]] = imputer.transform(validate[["age"]])
    test[["age"]] = imputer.transform(test[["age"]])
    return train, validate,test

def wrangle(df):
    '''cleans up the data frame '''
    df.drop_duplicates(inplace = True)
    columns_to_drop = ['person_age']
    df = df.drop(columns = columns_to_drop)
    df = df.dropna()
    # dummy_df = pd.get_dummies(df[['sex', 'embark_town']], dummy_na =False, drop_first = [True,True])
    # df = pd.concat([df, dummy_df], axis = 1)
    return df


    

def split_credit_defult_df(df_no_age):
    '''
    This function performs split on the data, stratify target variable.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df_no_age, test_size=.2, 
                                        random_state=123, 
                                        stratify=df_no_age.cb_person_default_on_file)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.cb_person_default_on_file)
    return train, validate, test



    #################### CONFUSION MATRIX CHART #########

def show_scores(TN, FP, FN, TP):
    
    ALL = TP + TN + FP + FN
    
    accuracy = (TP + TN)/ALL # How often did the model get it right?
    precision = TP/(TP+FP) # What is the quality of a positive prediction made by the model?
    recall = TP/(TP+FN) # How many of the true positives were found?   
    
    true_positive_rate = TP/(TP+FN) # Same as recall, actually
    true_negative_rate = TN/(TN+FP) # How many of the true negatives were found?
    false_positive_rate = FP/(FP+TN) # How often did we miss the negative and accidentally call it positive?
    false_negative_rate = FN/(FN+TP) # How often did we miss the positive and accidentally call it negative?
    
    f1_score = 2*(precision*recall)/(precision+recall) # Harmonic mean, good for imbalanced data sets
    support_pos = TP + FN # Number of actual positives in the sample
    support_neg = FP + TN # Number of actual negatives in the sample
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"True Positive Rate: {true_positive_rate}")
    print(f"True Negative Rate: {true_negative_rate}")
    print(f"False Positive Rate: {false_positive_rate}")
    print(f"False Negative Rate: {false_negative_rate}")
    print(f"F1 Score: {f1_score}")
    print(f"Support (0): {support_pos}")
    print(f"Support (1): {support_neg}")





