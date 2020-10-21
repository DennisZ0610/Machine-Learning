pip install pycaret
pip install scikit-plot
pip install scikit-optimize


import skopt
import sklearn
import catboost
import numpy as np
import pandas as pd
import seaborn as sns
import scikitplot as skplt
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn import metrics
from skopt import BayesSearchCV
from xgboost import XGBClassifier
from sklearn.utils import resample
from pycaret.classification import *
from sklearn.metrics import roc_curve
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV 
from sklearn.preprocessing import StandardScaler
from skopt.space import Real, Categorical, Integer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



# Set max row display
pd.set_option('display.max_row', 1000)
# Set max column width to 50
pd.set_option('display.max_columns', 1000)
pd.options.display.width = 1000

df = pd.read_excel("E:\MMA\Courses\MMA 823 - Analytics for Financial Management\Assignment 2\Bankruptcy_data_Final.xlsx")

df.head()
df.shape
df.describe()
df.info
df.dtypes
### check missing value
df.isnull().sum()
### Replace infinite value with NAN
df = df.replace([np.inf, -np.inf], np.nan)

df2 = df.dropna()
df2.isnull().sum() # Now it's complete, no missing value
df2.shape

###### check for correlation
corr = df2.corr()
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

### Initial Classification via PyCaret
class1 = setup(data = df2, target = 'BK', session_id=123)
compare_models(sort = 'AUC')
class1

### EDA
df2.columns.values
df2['Data Year - Fiscal'].value_counts()
df2['Data Year - Fiscal'].nunique()

plt.hist(df2['Data Year - Fiscal'], bins = 'auto')
plt.hist(df2['Tobin\'s Q'], bins = 'auto')
plt.hist(df2['EPS'], bins = 'auto')
plt.hist(df2['Liquidity'], bins = 'auto')
plt.hist(df2['Profitability'], bins = 'auto')
plt.hist(df2['Productivity'], bins = 'auto')
plt.hist(df2['Leverage Ratio'], bins = 'auto')
plt.hist(df2['Asset Turnover'], bins = 'auto')
plt.hist(df2['Operational Margin'], bins = 'auto')
plt.hist(df2['Return on Equity'], bins = 'auto')
plt.hist(df2['Market Book Ratio'], bins = 'auto')
plt.hist(df2['Assets Growth'], bins = 'auto')
plt.hist(df2['Sales Growth'], bins = 'auto')
plt.hist(df2['Employee Growth'], bins = 'auto')
plt.hist(df2['BK'], bins = 'auto')




### Split into Train and Test based on Year
train = df2[df2['Data Year - Fiscal'] <= 2010]
test = df2[df2['Data Year - Fiscal'] >= 2011]

### Check data balancing
train['BK'].value_counts()
### It's very imbalanced

### Upsampling
df_bk0 = train[train['BK'] == 0]
df_bk1 = train[train['BK'] == 1]

df_bk1_upsampled = resample(df_bk1, replace = True, n_samples = 54186, random_state = 123)

# Combine the majority set with the upsampled set
train = pd.concat([df_bk0, df_bk1_upsampled])
train['BK'].value_counts() # Now it's balanced

train.describe
test.describe

x_train = train.drop(['BK'], axis = 1)
y_train = train['BK']
x_test = test.drop(['BK'], axis = 1)
y_test = test['BK']

y_train.value_counts()
y_test.value_counts()

### Def function to plot ROC curve
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

### Random Forest
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Predict Probabilities and values
y_probs = rf.predict_proba(x_test)
y_probs = y_probs[:, 1]
y_pred = rf.predict(x_test)

print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_probs)) ### 0.790

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plot_roc_curve(fpr, tpr)


### Random Forest Tuning - Random Search
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
#random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
# Train the model
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                               n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(x_train, y_train)
# Predict
y_probs_rs = rf_random.predict_proba(x_test)
y_probs_rs = y_probs_rs[:, 1]
y_pred_rs = rf_random.predict(x_test)

print(classification_report(y_test, y_pred_rs))
print(roc_auc_score(y_test, y_probs_rs)) # 0.869

fpr, tpr, thresholds = roc_curve(y_test, y_probs_rs)
plot_roc_curve(fpr, tpr)

rf_random.best_estimator_
rf_random.best_params_
rf_random.best_score_


rf_tuned = RandomForestClassifier(n_estimators = 400, min_samples_split = 2, min_samples_leaf = 1, 
                            max_features = 'sqrt', max_depth = None, bootstrap = False, random_state= 123)
rf_tuned.fit(x_train, y_train)

y_probs_tuned = rf_tuned.predict_proba(x_test)
y_probs_tuned = y_probs_tuned[:, 1]
y_pred_tuned = rf_tuned.predict(x_test)

print(classification_report(y_test, y_pred_tuned))
print(roc_auc_score(y_test, y_probs_tuned)) # 0.864

fpr, tpr, thresholds = roc_curve(y_test, y_probs_tuned)
plot_roc_curve(fpr, tpr)

###### CatBoost without tuning
cb = CatBoostClassifier()0.
cb.fit(x_train, y_train)

y_probs = cb.predict_proba(x_test)
y_probs = y_probs[:, 1]
y_pred = cb.predict(x_test)

print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_probs)) ### 0.910

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plot_roc_curve(fpr, tpr)

###### CatBoost with Tuning
cb_param_grid = {'iterations': Integer(10, 1000),
                 'depth': Integer(1, 8),
                 'learning_rate': Real(0.01, 1.0, 'log-uniform'),
                 'random_strength': Real(1e-9, 10, 'log-uniform'),
                 'bagging_temperature': Real(0.0, 1.0),
                 'border_count': Integer(1, 255),
                 'l2_leaf_reg': Integer(2, 30),
                 'scale_pos_weight':Real(0.01, 1.0, 'uniform')}
cb_bs = BayesSearchCV(cb, cb_param_grid, scoring = 'roc_auc', n_iter = 100, n_jobs = 1,
                      return_train_score = False, refit = True, optimizer_kwargs = {'base_estimator': 'GP'}, 
                      random_state = 123)

cb_bs.fit(x_train, y_train)

y_probs = cb_bs.predict_proba(x_test)
y_probs = y_probs[:, 1]
y_pred = cb_bs.predict(x_test)

print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_probs)) ### 0.903

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plot_roc_curve(fpr, tpr)

# Find the best parameters
cb_bs.best_params_
# Use the parameters to re-run the model
cb_tuned = CatBoostClassifier(iterations = 1000, depth = 8,
                 learning_rate = 0.11574, random_strength = 1e-9,
                 bagging_temperature = 1.0,
                 border_count = 178,
                 l2_leaf_reg = 2,
                 scale_pos_weight = 1.0)
cb_tuned.fit(x_train, y_train)

y_probs = cb_tuned.predict_proba(x_test)
y_probs = y_probs[:, 1]
y_pred = cb_tuned.predict(x_test)

print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_probs)) ### 0.903

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plot_roc_curve(fpr, tpr)

###### XGBoost Without Tuning
xgb = XGBClassifier()
xgb.fit(x_train, y_train)

y_pred = xgb.predict(x_test)

y_probs = xgb.predict_proba(x_test)
y_probs = y_probs[:, 1]

print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_probs))

fpr, tpr, thresholds = roc_curve(y_test, y_probs) #0.887
plot_roc_curve(fpr, tpr)

### XGBoost with Tuning
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
xgb_gs = GridSearchCV(estimator = xgb, param_grid= params, scoring = 'roc_auc', n_jobs = 4, verbose = 3)
xgb_gs.fit(x_train, y_train)

y_pred = xgb_gs.predict(x_test)

y_probs = xgb_gs.predict_proba(x_test)
y_probs = y_probs[:, 1]

print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_probs)) #0.927

fpr, tpr, thresholds = roc_curve(y_test, y_probs) 
plot_roc_curve(fpr, tpr)

# Best Parameters
xgb_gs.best_params_
xgb_tuned = XGBClassifier(colsample_bytree = 0.6, gamma = 2, max_depth = 5, min_child_weight = 1, subsample = 0.8)
xgb_tuned.fit(x_train, y_train)

y_pred = xgb_tuned.predict(x_test)

y_probs = xgb_tuned.predict_proba(x_test)
y_probs = y_probs[:, 1]

print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_probs)) #0.927

fpr, tpr, thresholds = roc_curve(y_test, y_probs) 
plot_roc_curve(fpr, tpr)

###### Gradient Boosting without Tuning
gb = GradientBoostingClassifier()
gb.fit(x_train, y_train)

y_pred = gb.predict(x_test)
y_probs = gb.predict_proba(x_test)
y_probs = y_probs[:, 1]

print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_probs)) #0.945

fpr, tpr, thresholds = roc_curve(y_test, y_probs) 
plot_roc_curve(fpr, tpr)

### Gradient Boosting with Tuning
params = {'learning_rate':[0.1,0.05,0.01,0.005,0.001], 
          'n_estimators':[100, 500, 1000, 1500, 2000],
          'max_depth':[2,3,4,5,6,7],
          'max_features':[2,3,4,5,6,7],
          'min_samples_split':[2,4,6,8,10,20,30,40,50],
          'min_samples_leaf':[1,3,5,7,9],
          'subsample':[0.6,0.7,0.8,0.9,1]}
gb_gs = GridSearchCV(gb, param_grid=params, scoring = 'roc_auc')
gb_gs.fit(x_train, y_train)

###### LDA without tuning
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
y_pred = lda.predict(x_test)

y_probs = lda.predict_proba(x_test)
y_probs = y_probs[:, 1]

print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_probs))

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plot_roc_curve(fpr, tpr)

### LDA with Tuning
lda_param_grid = {"solver": ['svd', 'lsqr', 'eigen'],
              'tol': (0.0001, 0.0002, 0.0003),
              'n_components': (1, 2, 3, 4, 5),
              'store_covariance': [True, False]}
lda_gs = GridSearchCV(lda, param_grid = lda_param_grid, scoring = "accuracy", n_jobs = 4, verbose = 1)
lda_gs.fit(x_train, y_train)
lda_best = lda_gs.best_estimator_
lda_best
lda_gs.best_score_

y_pred = lda_gs.predict(x_test)

y_probs = lda_gs.predict_proba(x_test)
y_probs = y_probs[:, 1]

print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_probs))

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plot_roc_curve(fpr, tpr)

###### QDA without Tuning
qda = QuadraticDiscriminantAnalysis()
qda.fit(x_train, y_train)
y_pred = qda.predict(x_test)

y_probs = qda.predict_proba(x_test)
y_probs = y_probs[:, 1]

print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_probs))

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plot_roc_curve(fpr, tpr)

### QDA with Tuning
qda_param_grid = {'reg_param': [0.00001, 0.0001, 0.001,0.01, 0.1], 
    'store_covariance': (True, False),
    'tol': [0.0001, 0.001,0.01, 0.1]
    }
qda_gs = GridSearchCV(
    qda,
    param_grid = qda_param_grid,
    scoring = 'accuracy',
    n_jobs = -1,
    cv = 5
)
qda_gs.fit(x_train, y_train)
y_pred = qda_gs.predict(x_test)

y_probs = qda_gs.predict_proba(x_test)
y_probs = y_probs[:, 1]

print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_probs))

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plot_roc_curve(fpr, tpr)

###### Naive Bayes without Tuning
nb = GaussianNB()
nb.fit(x_train, y_train)

y_pred = nb.predict(x_test)

y_probs = nb.predict_proba(x_test)
y_probs = y_probs[:, 1]

print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_probs)) ### 0.576
print(accuracy_score(y_test, y_pred))

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plot_roc_curve(fpr, tpr)

###### Naive Bayes Tuning
nb_params = {'var_smoothing': np.logspace(0,-9, num=100)}
nb_gs = GridSearchCV(estimator=nb, 
                     param_grid=nb_params, 
                     verbose=1, 
                     scoring='roc_auc')

nb_gs.fit(x_train, y_train)

y_pred = nb_gs.predict(x_test)

y_probs = nb_gs.predict_proba(x_test)
y_probs = y_probs[:, 1]

nb_gs.best_estimator_
nb_gs.best_score_
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_probs)) ### 0.772
print(accuracy_score(y_test, y_pred))

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plot_roc_curve(fpr, tpr)

###### SVM
svm = SVC()
svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)

print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_pred)) 

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plot_roc_curve(fpr, tpr)

### SVM with Tuning
tuned_parameters = {
 'C': (np.arange(5,10,1)) , 'kernel': ['linear'],
 'C': (np.arange(5,10,1)) , 'gamma': [0.01,0.1,1,2,10], 'kernel': ['rbf'],
 'degree': [2,3,4] ,'gamma': [0.01,0.1,1,2,10], 'C':(np.arange(5,10,1)) , 'kernel':['poly']
                   }
svm_gs = GridSearchCV(svm, tuned_parameters,cv=10,scoring='roc_auc',n_jobs=-1, verbose = True)
svm_gs.fit(x_train,y_train)

y_pred = svm_gs.predict(x_test)

print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_pred)) 

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plot_roc_curve(fpr, tpr)

###### Logistic Regression without Tuning ######
lr = LogisticRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

y_probs = lr.predict_proba(x_test)
y_probs = y_probs[:, 1]

print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_probs)) ### 0.777

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plot_roc_curve(fpr, tpr)

### Logistic Regression with Tuning
lr_param_grid = {"penalty": ['l1', 'l2'],
                 'tol': [0.0001, 0.0002, 0.0003],
                 'max_iter': [100,200,300],
                 "C": [0.01, 0.1, 1, 10, 100],
                 "solver": ['liblinear'],
                 "verbose":[1],
                 "max_iter": [100, 500, 1000, 2000]}
lr_gs = GridSearchCV(lr, param_grid = lr_param_grid, scoring = "accuracy", n_jobs = 4, verbose = 1)
lr_gs.fit(x_train, y_train)

y_pred = lr.predict(x_test)

y_probs = lr.predict_proba(x_test)
y_probs = y_probs[:, 1]

print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_probs))

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plot_roc_curve(fpr, tpr)