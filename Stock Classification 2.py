import catboost

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from skopt import BayesSearchCV
from xgboost import XGBClassifier
from sklearn import preprocessing
from pycaret.classification import *
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_curve
from catboost import CatBoostClassifier

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier

from skopt.space import Real, Categorical, Integer
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier


from sklearn.metrics import classification_report, confusion_matrix 

df14 = pd.read_csv("E:\\MMA\\Courses\\MMA 823 - Analytics for Financial Management\\Final Project\\2014_Financial_Data.csv")
df15 = pd.read_csv("E:\\MMA\\Courses\\MMA 823 - Analytics for Financial Management\\Final Project\\2015_Financial_Data.csv")
df16 = pd.read_csv("E:\\MMA\\Courses\\MMA 823 - Analytics for Financial Management\\Final Project\\2016_Financial_Data.csv")
df17 = pd.read_csv("E:\\MMA\\Courses\\MMA 823 - Analytics for Financial Management\\Final Project\\2017_Financial_Data.csv")
df18 = pd.read_csv("E:\\MMA\\Courses\\MMA 823 - Analytics for Financial Management\\Final Project\\2018_Financial_Data.csv")



# Set max row display
pd.set_option('display.max_row', 1000)
# Set max column width to 50
pd.set_option('display.max_columns', 1000)
pd.options.display.width = 1000

### Drop the Stock Column
df14 = df14.drop(df14.columns[0], axis = 1)
df15 = df15.drop(df15.columns[0], axis = 1)
df16 = df16.drop(df16.columns[0], axis = 1)
df17 = df17.drop(df17.columns[0], axis = 1)
df18 = df18.drop(df18.columns[0], axis = 1)

### Convert Sector to Numeric
df14 = pd.get_dummies(df14,columns=['Sector'],dtype= 'int64')
df15 = pd.get_dummies(df15,columns=['Sector'],dtype= 'int64')
df16 = pd.get_dummies(df16,columns=['Sector'],dtype= 'int64')
df17 = pd.get_dummies(df17,columns=['Sector'],dtype= 'int64')
df18 = pd.get_dummies(df18,columns=['Sector'],dtype= 'int64')

### Change Data Type
df14['Class'] = df14['Class'].astype(object)
df15['Class'] = df15['Class'].astype(object)
df16['Class'] = df16['Class'].astype(object)
df17['Class'] = df17['Class'].astype(object)
df18['Class'] = df18['Class'].astype(object)

### Rename
df14 = df14.rename(columns = {"2015 PRICE VAR [%]": "PRICE VAR [%]"})
df15 = df15.rename(columns = {"2016 PRICE VAR [%]": "PRICE VAR [%]"})
df16 = df16.rename(columns = {"2017 PRICE VAR [%]": "PRICE VAR [%]"})
df17 = df17.rename(columns = {"2018 PRICE VAR [%]": "PRICE VAR [%]"})
df18 = df18.rename(columns = {"2019 PRICE VAR [%]": "PRICE VAR [%]"})

### Impute Data
imputer = KNNImputer(n_neighbors=20, weights='distance', metric='nan_euclidean', copy=True)
df14_clean = imputer.fit_transform(df14)
df14_clean = pd.DataFrame(df14_clean)
df14_clean.columns = list(df14)

df15_clean = imputer.fit_transform(df15)
df15_clean = pd.DataFrame(df15_clean)
df15_clean.columns = list(df15)

df16_clean = imputer.fit_transform(df16)
df16_clean = pd.DataFrame(df16_clean)
df16_clean.columns = list(df16)

df17_clean = imputer.fit_transform(df17)
df17_clean = pd.DataFrame(df17_clean)
df17_clean.columns = list(df17)

df18_clean = imputer.fit_transform(df18)
df18_clean = pd.DataFrame(df18_clean)
df18_clean.columns = list(df18)

### Add S&P Annual Average Closing Price as a column, and let 2014 be 1
### 2014: 1931.38, 2015: 2061.07, 2016: 2094.65, 2017: 2449.08, 2018: 2746.21
###### Add Year into data
df14_clean['S&P Price'] = 1931.38/1931.38
df15_clean['S&P Price'] = 2061.07/1931.38
df16_clean['S&P Price'] = 2094.65/1931.38
df17_clean['S&P Price'] = 2449.08/1931.38
df18_clean['S&P Price'] = 2746.21/1931.38

##### Check Missing Value Again
df14_clean.isnull().sum()
df15_clean.isnull().sum()
df16_clean.isnull().sum()
df17_clean.isnull().sum()
df18_clean.isnull().sum()

###### Concatenate
train = pd.concat([df14_clean, df15_clean, df16_clean])
test = pd.concat([df17_clean, df18_clean])
df_all = pd.concat([df14_clean, df15_clean, df16_clean, df17_clean, df18_clean])

### Feature to Remove
drop_features = ['inventoryTurnover', 'eBITperRevenue', 'returnOnCapitalEmployed', 'debtEquityRatio', 'debtRatio',
                'payablesTurnover', 'eBTperEBIT', 'cashFlowCoverageRatios', 'nIperEBT', 'priceBookValueRatio',
                'priceToBookRatio', 'PTB ratio', 'Days Payables Outstanding', 'priceToOperatingCashFlowsRatio', 
                'priceCashFlowRatio', 'interestCoverage', 'priceSalesRatio', 'priceToSalesRatio', 'payoutRatio',
                'operatingCashFlowPerShare', 'freeCashFlowPerShare', 'priceToFreeCashFlowsRatio', 'priceEarningsRatio',
                'cashPerShare', 'returnOnEquity', 'Net Income Com', 'eBITperRevenue', 'ebitperRevenue', 'netProfitMargin',
                'priceSalesRatio', 'currentRatio', 'PRICE VAR [%]']

train = train.drop(columns = drop_features)
test = test.drop(columns = drop_features)
df_all = df_all.drop(columns = drop_features)

x_train = train.drop(columns = 'Class')
y_train = pd.DataFrame(train['Class'])
x_test = test.drop(columns = 'Class')
y_test = pd.DataFrame(test['Class'])

### Initial Classification
### Change Data Type
df_all['Class'] = df_all['Class'].astype(object)

class_all = setup(data = df_all, target = 'Class', session_id=123)
compare_models(sort='AUC')
class_all ### CatBoost, Light Gradient Boosting, Extra Trees, Gradient Boosting, XGBoosting, Ada Boost, RF, LDA


### Def function to plot ROC curve
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
### XGBoost
y_train['Class'] = y_train['Class'].astype(int)
y_test['Class'] = y_test['Class'].astype(int)

xgb = XGBClassifier()
xgb.fit(x_train, y_train)

y_pred = xgb.predict(x_test)

y_probs = xgb.predict_proba(x_test)
y_probs = y_probs[:, 1]

print(classification_report(y_test, y_pred)) # Accuracy 0.54
print(roc_auc_score(y_test, y_probs)) # AUC 0.624

fpr, tpr, thresholds = roc_curve(y_test, y_probs) 
plot_roc_curve(fpr, tpr)

### XGBoost Tuning
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 2.5, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
xgb_gs = GridSearchCV(estimator = xgb, param_grid= params, scoring = 'accuracy', n_jobs = 4, verbose = 3)
xgb_gs.fit(x_train, y_train.values.ravel())

y_pred = xgb_gs.predict(x_test)

y_probs = xgb_gs.predict_proba(x_test)
y_probs = y_probs[:, 1]

print(classification_report(y_test, y_pred)) # Accuracy 0.58
print(roc_auc_score(y_test, y_probs)) #0.662

fpr, tpr, thresholds = roc_curve(y_test, y_probs) 
plot_roc_curve(fpr, tpr)

