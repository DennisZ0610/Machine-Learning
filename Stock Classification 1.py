import catboost

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn import preprocessing
from pycaret.classification import *
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_curve
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
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

###
df14.isnull().sum().sort_values(ascending = False)
df15.isnull().sum().sort_values(ascending = False)
df16.isnull().sum().sort_values(ascending = False)
df17.isnull().sum().sort_values(ascending = False)
df18.isnull().sum().sort_values(ascending = False)

### Correlation
corr = df14.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

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

###### Add Year into data
df14_clean['Year'] = 2014
df15_clean['Year'] = 2015
df16_clean['Year'] = 2016
df17_clean['Year'] = 2017
df18_clean['Year'] = 2018

##### Check Missing Value Again
df14_clean.isnull().sum()
df15_clean.isnull().sum()
df16_clean.isnull().sum()
df17_clean.isnull().sum()
df18_clean.isnull().sum()

###### Concatenate
df_all = pd.concat([df14_clean, df15_clean, df16_clean, df17_clean, df18_clean])
df_all.shape
df_all.isnull().sum()
list(df_all)

###### Check Correlation
df_all['Class'] = df_all['Class'].astype(int)
df_all['Debt to Equity'] = df_all['Debt to Equity'].astype(float)
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(df_all, 200))

### Duplicate Features
# inventoryTurnover -  Inventory Turnover, ebitperRevenue - eBITperRevenue, returnOnCapitalEmployed - ROIC,
# debtEquityRatio - Debt to Equity, debtRatio - Debt to Assets, payablesTurnover - Payables Turnover, 
# ebtperEBIT - eBTperEBIT, cashFlowToDebtRatio - cashFlowCoverageRatios, niperEBT - nIperEBT, 
# priceBookValueRatio - priceToBookRatio - PB ratio - PTB ratio, daysOfPayablesOutstanding -  Days Payables Outstanding,
# priceToOperatingCashFlowsRatio - POCF ratio - priceCashFlowRatio, interestCoverage - Interest Coverage, 
# priceSalesRatio - priceToSalesRatio - Price to Sales Ratio,
# payoutRatio - Payout Ratio, operatingCashFlowPerShare - Operating Cash Flow per Share, 
# freeCashFlowPerShare - Free Cash Flow per Share, priceToFreeCashFlowsRatio - PFCF ratio, priceEarningsRatio - PE ratio,
# cashPerShare - Cash per Share, returnOnEquity - ROE,
# Net Income - Net Income Com, EBIT Margin - eBITperRevenue - ebitperRevenue, Net Profit Margin - netProfitMargin,
# priceSalesRatio - Price to Sales Ratio, currentRatio - Current ratio

### Feature to Remove
dup_features = ['inventoryTurnover', 'eBITperRevenue', 'returnOnCapitalEmployed', 'debtEquityRatio', 'debtRatio',
                'payablesTurnover', 'eBTperEBIT', 'cashFlowCoverageRatios', 'nIperEBT', 'priceBookValueRatio',
                'priceToBookRatio', 'PTB ratio', 'Days Payables Outstanding', 'priceToOperatingCashFlowsRatio', 
                'priceCashFlowRatio', 'interestCoverage', 'priceSalesRatio', 'priceToSalesRatio', 'payoutRatio',
                'operatingCashFlowPerShare', 'freeCashFlowPerShare', 'priceToFreeCashFlowsRatio', 'priceEarningsRatio',
                'cashPerShare', 'returnOnEquity', 'Net Income Com', 'eBITperRevenue', 'ebitperRevenue', 'netProfitMargin',
                'priceSalesRatio', 'currentRatio']
df_all = df_all.drop(columns = ['PRICE VAR [%]'])

df_all = df_all.drop(columns = dup_features)
df_all.shape
print(get_top_abs_correlations(df_all, 200))


### EDA
df14.shape
df14.columns
df14.dtypes
df14.head()

corr = df_all.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
### Initial Classification
### Change Data Type Again
df14_clean['Class'] = df14_clean['Class'].astype(object)
df15_clean['Class'] = df15_clean['Class'].astype(object)
df16_clean['Class'] = df16_clean['Class'].astype(object)
df17_clean['Class'] = df17_clean['Class'].astype(object)
df18_clean['Class'] = df18_clean['Class'].astype(object)

class_all = setup(data = df_all, target = 'Class', session_id=123)
compare_models(sort='AUC')
class_all ### Extra Trees, Catboost, Light Gradient Boosting, XGBoost, Ada Boost, LDA, RF
### Use 2014 to predict 2015, whether should purchase that stock

# Check Balancing
y_train14['Class'].value_counts() #relatively balanced
y_test15['Class'].value_counts() # relatively balanced 

### Def function to plot ROC curve
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

### CatBoost
y_train14 = y_train14.astype(int)
y_test15 = y_test15.astype(int)
cb1415 = CatBoostClassifier()
cb1415.fit(x_train14, y_train14)

y_probs15 = cb1415.predict_proba(x_test15)
y_probs15 = y_probs15[:, 1]
y_pred15 = cb.predict(x_test15)

print(classification_report(y_test15, y_pred15))
print(roc_auc_score(y_test15, y_probs15)) ### 0.999

fpr, tpr, thresholds = roc_curve(y_test15, y_probs15)
plot_roc_curve(fpr, tpr)

### Feature Importance for all the year 
fi_all = setup(data = df_all, target = 'Class', session_id=123)
rf_fi = create_model('rf')
tune_rf_fi = tune_model(rf_fi)

x = fi_all[2]
importances = tune_rf_fi.feature_importances_
std = np.std([tree.feature_importances_ for tree in tune_rf_fi.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")

print(x.shape[1])

for f in range(x.shape[1]):
    print("%d. feature %d %s (%f)" % (f + 1,indices[f], x.keys()[f], importances[indices[f]]))
plt.figure(figsize=(20,12))
plt.title("Feature importances")
plt.bar(range(x.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(x.shape[1]), indices)
plt.xlim([-1, x.shape[1]])
plt.show()





###### Classification based on 11 Sectors 
# Financial Services, Healthcare, Technology, Industrials, Consumer Cyclical
# Basic Materials, Real Estate, Energy, Consumer Defensive, Utilities, Communication Services
df14_2 = pd.read_csv("E:\\MMA\\Courses\\MMA 823 - Analytics for Financial Management\\Final Project\\2014_Financial_Data.csv")
df15_2 = pd.read_csv("E:\\MMA\\Courses\\MMA 823 - Analytics for Financial Management\\Final Project\\2015_Financial_Data.csv")
df16_2 = pd.read_csv("E:\\MMA\\Courses\\MMA 823 - Analytics for Financial Management\\Final Project\\2016_Financial_Data.csv")
df17_2 = pd.read_csv("E:\\MMA\\Courses\\MMA 823 - Analytics for Financial Management\\Final Project\\2017_Financial_Data.csv")
df18_2 = pd.read_csv("E:\\MMA\\Courses\\MMA 823 - Analytics for Financial Management\\Final Project\\2018_Financial_Data.csv")


### Drop the Stock Column
df14_2 = df14_2.drop(df14_2.columns[0], axis = 1)
df15_2 = df15_2.drop(df15_2.columns[0], axis = 1)
df16_2 = df16_2.drop(df16_2.columns[0], axis = 1)
df17_2 = df17_2.drop(df17_2.columns[0], axis = 1)
df18_2 = df18_2.drop(df18_2.columns[0], axis = 1)

### Rename
df14_2 = df14_2.rename(columns = {"2015 PRICE VAR [%]": "PRICE VAR [%]"})
df15_2 = df15_2.rename(columns = {"2016 PRICE VAR [%]": "PRICE VAR [%]"})
df16_2 = df16_2.rename(columns = {"2017 PRICE VAR [%]": "PRICE VAR [%]"})
df17_2 = df17_2.rename(columns = {"2018 PRICE VAR [%]": "PRICE VAR [%]"})
df18_2 = df18_2.rename(columns = {"2019 PRICE VAR [%]": "PRICE VAR [%]"})

###### Add Year into data
df14_2['Year'] = 2014
df15_2['Year'] = 2015
df16_2['Year'] = 2016
df17_2['Year'] = 2017
df18_2['Year'] = 2018

###### Concatenate
df_all2 = pd.concat([df14_2, df15_2, df16_2, df17_2, df18_2])
df_all2.isnull().sum().sort_values(ascending = False)
# remove cashConversionCycle, operatingCycle
df_all2 = df_all2.drop(columns = ['PRICE VAR [%]', 'cashConversionCycle', 'operatingCycle'])
df_all2 = df_all2.drop(columns = dup_features)



## Split based on Sector
df_finance = df_all2[df_all2['Sector'] == 'Financial Services']
df_health = df_all2[df_all2['Sector'] == 'Healthcare']
df_tech = df_all2[df_all2['Sector'] == 'Technology']
df_industrials = df_all2[df_all2['Sector'] == 'Industrials']
df_cylical = df_all2[df_all2['Sector'] == 'Consumer Cyclical']
df_basic = df_all2[df_all2['Sector'] == 'Basic Materials']
df_real = df_all2[df_all2['Sector'] == 'Real Estate']
df_energy = df_all2[df_all2['Sector'] == 'Energy']
df_defensive = df_all2[df_all2['Sector'] == 'Consumer Defensive']
df_utilities = df_all2[df_all2['Sector'] == 'Utilities']
df_communication = df_all2[df_all2['Sector'] == 'Communication Services']

df_finance = df_finance.drop(columns = ['Sector'])
df_health = df_health.drop(columns = ['Sector'])
df_tech = df_tech.drop(columns = ['Sector'])
df_industrials = df_industrials.drop(columns = ['Sector'])
df_cylical = df_cylical.drop(columns = ['Sector'])
df_basic = df_basic.drop(columns = ['Sector'])
df_real = df_real.drop(columns = ['Sector'])
df_energy = df_energy.drop(columns = ['Sector'])
df_defensive = df_defensive.drop(columns = ['Sector'])
df_utilities = df_utilities.drop(columns = ['Sector'])
df_communication = df_communication.drop(columns = ['Sector'])

### Impute
imputer = KNNImputer(n_neighbors=20, weights='distance', metric='nan_euclidean', copy=True)

df_finance2 = imputer.fit_transform(df_finance)
df_finance2 = pd.DataFrame(df_finance2)
df_finance2.columns = list(df_finance)

df_health2 = imputer.fit_transform(df_health)
df_health2= pd.DataFrame(df_health2)
df_health2.columns = list(df_health)

df_tech2 = imputer.fit_transform(df_tech)
df_tech2 = pd.DataFrame(df_tech2)
df_tech2.columns = list(df_tech)

df_industrials2 = imputer.fit_transform(df_industrials)
df_industrials2 = pd.DataFrame(df_industrials2)
df_industrials2.columns = list(df_industrials)

df_cylical2 = imputer.fit_transform(df_cylical)
df_cylical2 = pd.DataFrame(df_cylical2)
df_cylical2.columns = list(df_cylical)

df_basic2 = imputer.fit_transform(df_basic)
df_basic2 = pd.DataFrame(df_basic2)
df_basic2.columns = list(df_basic)

df_real2 = imputer.fit_transform(df_real)
df_real2 = pd.DataFrame(df_real2)
df_real2.columns = list(df_real)

df_energy2 = imputer.fit_transform(df_energy)
df_energy2 = pd.DataFrame(df_energy2)
df_energy2.columns = list(df_energy)

df_defensive2 = imputer.fit_transform(df_defensive)
df_defensive2 = pd.DataFrame(df_defensive2)
df_defensive2.columns = list(df_defensive)

df_utilities2 = imputer.fit_transform(df_utilities)
df_utilities2 = pd.DataFrame(df_utilities2)
df_utilities2.columns = list(df_utilities)

df_communication2 = imputer.fit_transform(df_communication)
df_communication2 = pd.DataFrame(df_communication2)
df_communication2.columns = list(df_communication)

### Classification
# Change Type
df_finance2['Class'] = df_finance2['Class'].astype(object)
df_health2['Class'] = df_health2['Class'].astype(object)
df_tech2['Class'] = df_tech2['Class'].astype(object)
df_industrials2['Class'] = df_industrials2['Class'].astype(object)
df_cylical2['Class'] = df_cylical2['Class'].astype(object)
df_basic2['Class'] = df_basic2['Class'].astype(object)
df_cylical2['Class'] = df_cylical2['Class'].astype(object)
df_real2['Class'] = df_real2['Class'].astype(object)
df_energy2['Class'] = df_energy2['Class'].astype(object)
df_defensive2['Class'] = df_defensive2['Class'].astype(object)
df_utilities2['Class'] = df_utilities2['Class'].astype(object)
df_communication2['Class'] = df_communication2['Class'].astype(object)



# Financial Services
class_finance = setup(data = df_finance2, target = 'Class', session_id=123)
compare_models(sort='AUC')
class_finance

# Healthcare
class_health = setup(data = df_health2, target = 'Class', session_id=123)
compare_models(sort='AUC')
class_health

# Healthcare
class_tech = setup(data = df_tech2, target = 'Class', session_id=123)
compare_models(sort='AUC')
class_tech

# industrials
class_indus = setup(data = df_industrials2, target = 'Class', session_id=123)
compare_models(sort='AUC')
class_indus

# cylical
class_cylical = setup(data = df_cylical2, target = 'Class', session_id=123)
compare_models(sort='AUC')
class_cylical

# basic materials
class_basic = setup(data = df_basic2, target = 'Class', session_id=123)
compare_models(sort='AUC')
class_basic

# Real Estate df_real2
class_real = setup(data = df_real2, target = 'Class', session_id=123)
compare_models(sort='AUC')
class_real

# Energy
class_energy = setup(data = df_energy2, target = 'Class', session_id=123)
compare_models(sort='AUC')
class_energy

# Comsumer Defensive
class_defensive = setup(data = df_defensive2, target = 'Class', session_id=123)
compare_models(sort='AUC')
class_defensive

# utilities
class_utilities = setup(data = df_utilities2, target = 'Class', session_id=123)
compare_models(sort='AUC')
class_utilities

# communication
class_communication = setup(data = df_communication2, target = 'Class', session_id=123)
compare_models(sort='AUC')
class_communication
