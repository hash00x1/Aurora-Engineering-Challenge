###AURORA GASSING FEE PRICE PREDICTION MODEL

import glob
import json
import pandas as pd
import numpy as np
from import_files_2 import readFiles_2
from import_files_2 import flatten_json
from import_files_2 import readFiles_threshold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#set filepath of .json- DB
path = "/home/ubuntu/Visual_Studio/Aurora_Egineering_Challenge/tx_data_20220223/*" #tested on AWS with Ubuntu instance || replace with path to .json files

#Initialize df
new_df = readFiles_2(path, 100)
print(new_df)
print(new_df.keys())
print(new_df['result.block.height'])

#remove all rows from new_df with more than 5% NaN Values // residue of NaN results from flatten_json- function in readFiles_2
percentage = 5
min_count =  int(((100-percentage)/100)*new_df.shape[0] + 1)
new_df = new_df.dropna(axis=1, thresh=min_count)
print(new_df)

#Sort df by sequence, using feature 'Block Height'
new_df = new_df.sort_values('result.block.height')

###Set 1: Initialize Regression Models
#create list of keys from df
key_list = [i for i in new_df.keys()]
print(key_list)

#remove target from key_list
key_list.remove('result.receipts_outcome.l1.outcome.gas_burnt')

##initialize feature, target - dfs (X, y)
X = new_df[key_list]
y = new_df[['result.receipts_outcome.l1.outcome.gas_burnt']]

#remove non-numeric values from X
for index in X:
    if X[index].dtype == object:
        X[index] = X[index].str.replace(r'\D', '') #remove non-numeric values
        X[index] = X[index].str.lstrip('0')
        X[index] = X[index].str.replace(r'^\s*$', '0') #replace white-space with 0
        

#ensure all numeric fields are computable, by setting type to float       
X = X.astype(float)
y = y.astype(float)

#Clean df
X = X.loc[:, (X != 0).any(axis=0)] #drop all columns with only zeroes (0)
X.replace([np.inf, -np.inf], np.nan, inplace=True) #remove infinite values
for index in X:
    X[index] = X[index].fillna(X[index].mean()) #replace NaN with median values

#filter X by features
#remove features with low variance
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0)  # 0 is default
selector.fit_transform(X)
selector.get_support(indices=True)

# Use indices to get the corresponding column names of selected features
num_cols = list(X.columns[selector.get_support(indices=True)])
print(num_cols)

# Subset `X_num` to retain only selected features
X = X[num_cols]
print(X)

#drop highly correlated features from X
X_cor = X.corr()
upper_tri = X_cor.where(np.triu(np.ones(X_cor.shape),k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
X = X.drop(X[to_drop], axis=1)

##create correlation df(feature, target) for advanced feature selection
X_y = X.copy()
X_y['gas_burnt'] = y


cor_matrix = X_y.corr()
cor_target = abs(cor_matrix["gas_burnt"])
##Plot correlation matrix. Uncomment if you would like to use it.
#plt.figure(figsize=(12,10))
#sns.heatmap(cor_matrix, annot=True, cmap='RdBu_r') #plt.cm.Reds
#plt.show()


#Filter df to select highly correlated features only
relevant_features = cor_target[cor_target>0.5]
print(relevant_features)
relevant_features = relevant_features.drop('gas_burnt')
X = X[relevant_features.keys()]


#last check on finalized X dataframe. Ready for regression.
print(X)


#MODEL I: MULTILINEAR REGRESSION MODEL
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


##Gas Price formula in Aurora Documentation states that total gas price T is bound to (base cost * gas price) at point t and (base cost * gas price) at t+1. 
#Hence, gas price T is bound to relative change in T = x * delta(t, t+n). Where x is an unknown but constant variable and n are time-bound micro-changes [...] 
#[...] in the machine-state between two blocks t and t+1 (as an example for state changes, s. e.g. receipts_outcome.metadata.gas_profile).
#>> the following steps therefore filter "receipts_outcome" to represent learning-relevant features of delta(t, t+n) only. URL: https://docs.near.org/concepts/basics/transactions/gas#:~:text=NEAR%20is%20configured%20with%20base%20costs.%20An%20example%3A

#create list of keys, by which to filter X
filter_list = X.loc[:, X.columns.str.startswith("result.receipts_outcome.")].keys()

#choose first n entries in list | goal is to use the lowest number of features while maintaining highest accuracy
n = 2
if n == 0:
    filter_list = filter_list[0]
    X = X[['result.transaction_outcome.outcome.gas_burnt', filter_list]]
else:
    filter_list = filter_list[0:n]
    filter_list = [i for i in filter_list]
    filter_list.append('result.transaction_outcome.outcome.gas_burnt')
    X = X[filter_list]
X_copy = X
#Initialize the multilinear regression model
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size= 0.8, test_size = 0.2)

#Scale the training / testing data
#from sklearn.preprocessing import RobustScaler
#sc = RobustScaler() 
#x_train = sc.fit_transform(x_train)
#x_test = sc.transform(x_test)
#y_train = sc.fit_transform(y_train)
#y_test = sc.transform(y_test)

#Fit values and start the model computation
regr = LinearRegression()
regr.fit(x_train, y_train)
print(regr.score(x_train, y_train))
print(regr.score(x_test, y_test))

#plot x_test/y-predict
y_predict = regr.predict(x_test)

plt.scatter(y_test, y_predict, alpha=0.4) #original and predicted values should be aligned
plt.xlabel("transaction_features")
plt.ylabel("predicted_gasrate")
plt.show()

#Evaluate model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print(regr.coef_)
print(regr.intercept_)
r2_score = r2_score(y_test, y_predict)
mean_absolute_error = mean_absolute_error(y_test, y_predict)
mean_squared_error = np.sqrt(mean_squared_error(y_test, y_predict))
print("Multilinear Regression Testing: R2- Score = {r2_score} || Mean Absolute Error = {mean_absolute_error} || Mean Squared Error = {mean_squared_error}".format(r2_score=r2_score, mean_absolute_error=mean_absolute_error, mean_squared_error=mean_squared_error))

##write second file import function to pick all values from df with outlier value
##run these values through predict, if y >= treshold == TRUE for >90% of cases, challenge criteria should be fulfiled.
threshold = 1.5E14
test_df = readFiles_threshold(path, threshold, X.keys())
test_df = test_df.astype(float)
y_predict = regr.predict(test_df)
print(regr.score(test_df, y_predict))
counter = 0
for i in y_predict:
    if i >= threshold: #or 1.5E15
        print('Threshold reached')
        counter += 1
print(len(test_df))
print("Total Thresholds Found:" + str(counter))
accuracy = counter / len(test_df)
print("Model Accuracy 'Multilinear Regression' is {accuracy}".format(accuracy=accuracy))
#test value t:: t = pd.concat([test_df, new[0:50]]) >> y_pred = regr.predict(t)
t = test_df
t = pd.concat([t, X[0:50]]) 
y_pred = regr.predict(t)
counter = 0 #reset counter
for i in y_pred:
    if i >= threshold:
        print('Threshold reached')
        counter += 1
    else:
        print("Processing...")
print(len(test_df))
print("Total Thresholds Found via Multilinear Regression for test var t:" + str(counter))

###RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
#from numpy import float32
X = X_copy
print(X)
print(X.info())
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=0)
#scale the training / testing data
from sklearn.preprocessing import RobustScaler
#sc = RobustScaler() 
#x_train = sc.fit_transform(x_train)
#x_test = sc.transform(x_test)
#y_train = sc.fit_transform(y_train)
#y_test = sc.transform(y_test)
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
#r2_score = r2_score(y_test, y_pred)
#mean_absolute_error = mean_absolute_error(y_test, y_pred)
#mean_squared_error = np.sqrt(mean_squared_error(y_test, y_pred))
#print("Random Forest Testing: R2- Score = {r2_score} || Mean Absolute Error = {mean_absolute_error} || Mean Squared Error = {mean_squared_error}".format(r2_score=r2_score, mean_absolute_error=mean_absolute_error, mean_squared_error=mean_squared_error))


##RF TEST
#scaled_test_df = sc.fit_transform(test_df)
y_pred = regressor.predict(test_df.to_numpy())
print(y_pred)
#print(regr.score(test_df, y_pred))
counter = 0
for i in y_pred:
    if i >= threshold:
        print('Threshold reached')
        counter += 1
    else:
        print('Processing...')
print(len(test_df))
print("Total Thresholds Found via RandomForest:" + str(counter))
accuracy = counter / len(test_df)
print("Model Accuracy 'Random Forest Regression' is {accuracy}".format(accuracy=accuracy))
#test value t:: t = pd.concat([test_df, new[0:50]]) >> y_pred = regr.predict(t)
t = test_df
t = pd.concat([t, X[0:50]]) 
y_pred = regressor.predict(t)
counter = 0 #reset counter
for i in y_pred:
    if i >= threshold:
        print('Threshold reached')
        counter += 1
    else:
        print("Processing...")
print(len(test_df))
print("Total Thresholds Found via RandomForest for test var t:" + str(counter))
