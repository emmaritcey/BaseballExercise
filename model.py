import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin

from xgboost import XGBClassifier


def main():
    #get working directory path
    wd = os.getcwd()
    #load in training data and labels as dataframe
    trainDF = pd.read_csv(wd + '/training.csv')
    
    #remove observations with velocity less than 80
    trainDF = trainDF.drop(trainDF[trainDF.Velo < 80].index).reset_index()
    train_data = trainDF[['Velo','SpinRate','HorzBreak','InducedVertBreak']]#.to_numpy()
    train_labels = trainDF[['InPlay']]

    #create training and validation sets 
    #validation set is for hyperparameter tuning
    x_train, x_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.30, random_state=5)

    #replace missing values with median and standardize features
    #all features are numeric
    numeric_features = ['Velo', 'SpinRate', 'HorzBreak', 'InducedVertBreak']
    
    #replace missing values with median and standardize features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    #preprocessor contains numeric transformer
    preprocessor = ColumnTransformer(
        transformers=[ ('numeric', numeric_transformer, numeric_features)]
    ) 

    #compile preprocessing pipeline
    pp_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Transform the data
    x_train_clean = pp_pipeline.fit_transform(x_train)
    x_val_clean = pp_pipeline.transform(x_val)

    #build model
    xgb_clf = XGBClassifier(booster='gbtree', 
                            n_estimators=100,
                            eta=0.05, 
                            scale_pos_weight=2.7,
                            subsample=0.5
                           # early_stopping_rounds=5
                            )

    #train xgboost model on train data, 
    #evaluate on validation data (used for hyperparameter tuning)
    xgb_clf.fit(x_train_clean, y_train, 
            eval_set=[(x_train_clean, y_train),(x_val_clean, y_val)], 
            verbose=0
            )

    #apply model to test data
    x_test = pd.read_csv(wd + '/deploy.csv')
    x_test = x_test.drop(x_test[x_test.Velo < 80].index).reset_index()
    x_test_clean = pp_pipeline.transform(x_test) 
    y_test = xgb_clf.predict(x_test_clean)
    
    #save predictions to csv
    y_test = pd.DataFrame(y_test)
    y_test.to_csv("predictions.csv", index=False)
    #print(y_test)
    
    sorted_idx = xgb_clf.feature_importances_.argsort()
    plt.barh(train_data.columns[sorted_idx], xgb_clf.feature_importances_[sorted_idx])
    plt.xlabel("Xgboost Feature Importance")
    plt.show
    
main()