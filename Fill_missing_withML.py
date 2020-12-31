"""
You may have seen many ways and implementations to fill missing values up of your dataset. But most of them may give a high bias. 
Here is another efficient way to predict missing values using Gradient Boosting Model where you can predict missing values with 
values which may have some missing values too...

Created on Sun Dec 20 21:28:04 2020
@author: Rakib Mahmud
"""
#Import All the packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
#%matplotlib qt

#Import Dataset
df = pd.read_csv('AllData.csv') #The dataset used here is confidential till now
nrows, nnodes = df.shape

##Convert catagorical string data to numeric codes--------------------------------------
#Select all categorical data resemble string/object datatype
obj_df = df.select_dtypes(include=['object']).copy()
obj_df = obj_df.iloc[:,1:]
col_nam = obj_df.columns
df = df.drop(col_nam, axis = 1) #delete those columns from dataset temporarily
#Convert all datatype to categorical datatype to help further conversions
obj_df[col_nam] = obj_df[col_nam].astype('category')
obj_df.dtypes
#Now, convert all categorial string values to numeric value
for col in col_nam:
    obj_df[col] = obj_df[col].cat.codes
#Convert back null values to null
for ind in obj_df.index:
    for col in col_nam:
        if obj_df[col][ind] == -1:
           obj_df[col][ind] = np.nan 
#Now, concatenate them back
dataset = pd.concat([df, obj_df], axis=1)
dataset = dataset.drop(['Name'], axis = 1)
#Find correlation matrix
correlation = dataset.corr(min_periods=1) 

##--------------------------------------------------------------------------------------



##Predict age to fill up the missing ages-----------------------------------------------
#Select highly correlated columns with Age
age_df = dataset[['Weight','Height','Evidence of having Severe Acute respiratory Distress Syndrome'
                 ,'Occupation','[Cardiac problem]Chronic diseases','[Respiratory problem]Chronic diseases',
                 'Concurrent risk factors','Age']]
#Spliting test-train data
test_df = age_df[age_df["Age"].isnull()]
age_df_temp = age_df.dropna(subset = ["Age"])

y_train = age_df_temp["Age"]
X_train = age_df_temp.drop("Age", axis=1)
X_test = test_df.drop("Age", axis=1)
#train model to fit dataset and predict missing values from column "Age"
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
#Change column names as it don't support [] or < or , in column's name
X_train.columns = ['Weight','Height','Evidence of having Severe Acute respiratory Distress Syndrome'
                 ,'Occupation','Cardiac problem','Respiratory problem','Concurrent risk factors']
X_test.columns = ['Weight','Height','Evidence of having Severe Acute respiratory Distress Syndrome'
                 ,'Occupation','Cardiac problem','Respiratory problem','Concurrent risk factors']

xg_reg.fit(X_train,y_train)

y_pred = np.floor(xg_reg.predict(X_test))

#replace the missing values with predicted values
it = 0
ind = 0
for ind in dataset.index:
    if np.isnan(dataset['Age'][ind]):
        dataset['Age'][ind] = y_pred[it]
        it = it+1
##--------------------------------------------------------------------------------------
