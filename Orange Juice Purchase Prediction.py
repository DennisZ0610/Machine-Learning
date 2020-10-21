# [Dennis, Zhao]
# [20190903]
# [MMA 2021W]
# [Section 1]
# [MMA 869]
# [Aug 16, 2020]


# Answer to Question [7], Part [2]
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import randint
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV


pd.options.display.width = 100
pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 100)
# TODO: import other packages as necessary


# Read in data from Uncle Steve's GitHub repository
url = 'https://raw.githubusercontent.com/stepthom/sandbox/master/data/OJ.csv'
df = pd.read_csv(url, error_bad_lines=False)

# TODO: insert code here to perform the given task. Don't forget to document your code!
df.head()
df.shape # It has 1070 instances and 19 features
df.describe()
df.dtypes
# check missing value
df.isnull().sum() # there is no missing value
# change all the feature names to lower case
df = df.rename(columns = str.lower) 
# Basic EDA
df['purchase'].value_counts() # CH 653 vs. MM 417, it's relatively balanced
# Set up x and y
x = df.drop(columns = {'purchase', 'id'})
x.head()
y = pd.DataFrame(df['purchase'])
y.head()
# store7 is categorical variable in x, convert to dummy
x = pd.get_dummies(x, columns = ['store7'])
# Split df into train and test 0.8-0.2, random_state is used to reproduce the process.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)

###### 3 models to be used are: Logistic Regression, Decision Trees and Random Forest
### Logistic Regression
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr.predict(x_test)

# predict probabilities
lr_probs = lr.predict_proba(x_test)
# keep positive probabilities 
lr_probs = lr_probs[:, 1]
# calculate scores
lr_auc = roc_auc_score(y_test, lr_probs)
print('Logistic AUC: ' + str(lr_auc)) # print AUC 0.887
print(lr.fit(x_train, y_train)) # Print Hyperparameters, no tuning

# Hyperparameter Tuning - Grid Search
lr = LogisticRegression()
penalty = ['l1', 'l2'] # different penalty values
C = np.logspace(-4, 4, 20) # different C values
hyperparameters = dict(C = C, penalty = penalty)
gs = GridSearchCV(lr, hyperparameters, cv = 5, verbose = 0)
best_model = gs.fit(x, y)
print(best_model.best_estimator_.get_params()['penalty']) #l2
print(best_model.best_estimator_.get_params()['C']) # 4.28

lr_gs = LogisticRegression(C = 4.28, penalty = 'l2')

# predict probabilities
lr_gs_probs = gs.predict_proba(x_test)
# keep positive probabilities 
lr_gs_probs = lr_gs_probs[:, 1]
# calculate scores
lr_gs_auc = roc_auc_score(y_test, lr_gs_probs)
print('Logistic AUC: ' + str(lr_gs_auc)) # print AUC 0.891, improved
print(lr_gs.fit(x_train, y_train)) # Print Hyperparameters

# Hyperparameter Tuning - Random Search
lr = LogisticRegression()
C = np.linspace(1, 200)
penalty = ['l1', 'l2']
hyperparamemters = dict(C = C, penalty = penalty)
rs = RandomizedSearchCV(lr, hyperparamemters, n_iter = 100, random_state= 42)
rs.fit(x, y)
print(rs.best_params_) # l2, C = 171.57

lr_rs = LogisticRegression(C = 171.57, penalty = 'l2')

# predict probabilities
lr_rs_probs = rs.predict_proba(x_test)
# keep positive probabilities 
lr_rs_probs = lr_rs_probs[:, 1]
# calculate scores
lr_rs_auc = roc_auc_score(y_test, lr_rs_probs)
print('Logistic AUC: ' + str(lr_rs_auc)) # print AUC 0.893, improved
print(lr_rs.fit(x_train, y_train)) # Print Hyperparameters

### To make model better, scale the data
# Hyperparameter Tuning - Grid Search with Scale
scaler = StandardScaler().fit(x)
x_scaled = pd.DataFrame(scaler.transform(x))
x_scaled.head()
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=66)

lr = LogisticRegression()
penalty = ['l1', 'l2'] # different penalty values
C = np.logspace(-4, 4, 20) # different C values
hyperparameters = dict(C = C, penalty = penalty)
gs = GridSearchCV(lr, hyperparameters, cv = 5, verbose = 0)
best_model = gs.fit(x_scaled, y)
print(best_model.best_estimator_.get_params()['penalty']) #l2
print(best_model.best_estimator_.get_params()['C']) # 11.29

lr_gs = LogisticRegression(C = 11.29, penalty = 'l2')

# predict probabilities
lr_gs_probs = gs.predict_proba(x_test)
# keep positive probabilities 
lr_gs_probs = lr_gs_probs[:, 1]
# calculate scores
lr_gs_auc = roc_auc_score(y_test, lr_gs_probs)
print('Logistic AUC: ' + str(lr_gs_auc)) # print AUC 0.895, improved further
print(lr_gs.fit(x_train, y_train)) # Print Hyperparameters

# Hyperparameter Tuning - Random Search with Scale
scaler = StandardScaler().fit(x)
x_scaled = pd.DataFrame(scaler.transform(x))
x_scaled.head()
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=66)

lr = LogisticRegression()

C = np.linspace(1, 200)
penalty = ['l1', 'l2']
hyperparamemters = dict(C = C, penalty = penalty)
rs = RandomizedSearchCV(lr, hyperparamemters, n_iter = 100, random_state= 42)
rs.fit(x_scaled, y)
print(rs.best_params_) # l2, C = 13.18

lr_rs = LogisticRegression(C = 13.18, penalty = 'l2')

# predict probabilities
lr_rs_probs = rs.predict_proba(x_test)
# keep positive probabilities 
lr_rs_probs = lr_rs_probs[:, 1]
# calculate scores
lr_rs_auc = roc_auc_score(y_test, lr_rs_probs)
print('Logistic AUC: ' + str(lr_rs_auc)) # print AUC 0.896, improved
print(lr_rs.fit(x_train, y_train)) # Print Hyperparameters

###### Decision Trees
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

# predict probabilities
dt_probs = dt.predict_proba(x_test)
# keep positive probabilities 
dt_probs = dt_probs[:, 1]
# calculate scores
dt_auc = roc_auc_score(y_test, dt_probs)
print('Decision Tree AUC: ' + str(dt_auc)) # print AUC 0.721, original, no tuning at all
print(dt.fit(x_train, y_train)) # Print default Hyperparameter

### Tune the model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
param_dict = {"max_depth": [2, None],
              "max_features": randint(1, 10),
              "min_samples_leaf": randint(1, 10),
              "criterion": ["gini", "entropy"]}
dt = DecisionTreeClassifier(random_state= 42)
# Use RandomizedSearchCV
dt_cv = RandomizedSearchCV(dt, param_dict, cv=5)
dt_cv.fit(x, y)
# Print the tuned parameters and AUC
y_pred = dt_cv.predict(x_test)
# predict probabilities
dt_probs = dt_cv.predict_proba(x_test)
# keep positive probabilities 
dt_probs = dt_probs[:, 1]
# calculate scores
dt_auc = roc_auc_score(y_test, dt_probs)
print(dt_cv.best_params_)
print('Decision Tree AUC: ' + str(dt_auc)) # print AUC, it's much better, AUC score is different everytime
print(dt_cv.fit(x_train, y_train)) # Print Hyperparameter

# to make model better, scale the data
scaler = StandardScaler().fit(x)
x_scaled = pd.DataFrame(scaler.transform(x))
x_scaled.head()
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=66)

param_dict = {"max_depth": [2, None],
              "max_features": randint(1, 10),
              "min_samples_leaf": randint(1, 10),
              "criterion": ["gini", "entropy"]}
dt = DecisionTreeClassifier()
# Use RandomizedSearchCV
dt_cv_scaled = RandomizedSearchCV(dt, param_dict, cv=5)
dt_cv_scaled.fit(x_scaled, y)
# Print the tuned parameters and AUC
y_pred = dt_cv_scaled.predict(x_test)
# predict probabilities
dt_probs = dt_cv_scaled.predict_proba(x_test)
# keep positive probabilities 
dt_probs = dt_probs[:, 1]
# calculate scores
dt_auc = roc_auc_score(y_test, dt_probs)
print(dt_cv_scaled.best_params_)
print('Decision Tree AUC: ' + str(dt_auc)) # print AUC

###### Random Forest
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)
rf = RandomForestClassifier()
rf.fit(x_train, y_train) # default hyperparameters
y_pred = rf.predict(x_test)

rf_probs = rf.predict_proba(x_test)
# keep positive probabilities 
rf_probs = rf_probs[:, 1]
# calculate scores
rf_auc = roc_auc_score(y_test, rf_probs)
print('Random Forest AUC: ' + str(rf_auc)) # print AUC 0.862, original, no tuning at all
print(rf.fit(x_train, y_train)) # Print default Hyperparameter

### Tune the model: GridSearchCV
param_grid = [{'n_estimators': [5, 25], 'max_features': [5, 20], 
 'max_depth': [10, 50, None], 'bootstrap': [True, False], 'criterion': ['gini', 'entropy']}]

rf_gs = GridSearchCV(rf, param_grid, cv = 10, scoring = 'roc_auc')
rf_gs.fit(x_train, y_train)

rf_probs = rf_gs.predict_proba(x_test)
# keep positive probabilities 
rf_probs = rf_probs[:, 1]
# calculate scores
rf_gs_auc = roc_auc_score(y_test, rf_probs)
print('Random Forest AUC: ' + str(rf_gs_auc)) # print AUC 0.885, improved
print(rf_gs.best_estimator_) # print the best hyperparameters

### Tune the model: Randomized Search
n_estimators = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
max_features = ['auto', 'sqrt']
max_depth = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
min_samples_split = [5, 10, 15, 20, 25]
criterion = ['gini', 'entropy']

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'criterion': criterion}
rf_rs = RandomizedSearchCV(rf, param_distributions = random_grid, n_iter = 10, cv = 10, verbose = 2,
                           random_state = 66, n_jobs = -1, scoring = 'roc_auc')
rf_rs.fit(x_train, y_train)

rf_probs = rf_rs.predict_proba(x_test)
# keep positive probabilities 
rf_probs = rf_probs[:, 1]
# calculate scores
rf_rs_auc = roc_auc_score(y_test, rf_probs)
print('Random Forest AUC: ' + str(rf_rs_auc)) # print AUC 0.880, improved more
print(rf_rs.best_estimator_) # print the best hyperparameters

### To make model better: scale the data
scaler = StandardScaler().fit(x)
x_scaled = pd.DataFrame(scaler.transform(x))
x_scaled.head()
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=66)

rf = RandomForestClassifier()
rf.fit(x_train, y_train) # default hyperparameters
y_pred = rf.predict(x_test)

rf_probs = rf.predict_proba(x_test)
# keep positive probabilities 
rf_probs = rf_probs[:, 1]
# calculate scores
rf_auc = roc_auc_score(y_test, rf_probs)
print('Random Forest AUC: ' + str(rf_auc)) # print AUC 0.864, improved a little bit
print(rf.fit(x_train, y_train)) # Print default Hyperparameter

### GridSearchCV with scale
scaler = StandardScaler().fit(x)
x_scaled = pd.DataFrame(scaler.transform(x))
x_scaled.head()
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=66)

param_grid = [{'n_estimators': [5, 25], 'max_features': [5, 20], 
 'max_depth': [10, 50, None], 'bootstrap': [True, False], 'criterion': ['gini', 'entropy']}]
rf = RandomForestClassifier()
rf_gs = GridSearchCV(rf, param_grid, cv = 10, scoring = 'roc_auc')
rf_gs.fit(x_train, y_train)

rf_probs = rf_gs.predict_proba(x_test)
# keep positive probabilities 
rf_probs = rf_probs[:, 1]
# calculate scores
rf_gs_auc = roc_auc_score(y_test, rf_probs)
print('Random Forest AUC: ' + str(rf_gs_auc)) # print AUC 0.875, improved more
print(rf_gs.best_estimator_) # print the best hyperparameters

### Randomized Search with scale
scaler = StandardScaler().fit(x)
x_scaled = pd.DataFrame(scaler.transform(x))
x_scaled.head()
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=66)

n_estimators = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
max_features = ['auto', 'sqrt']
max_depth = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
min_samples_split = [5, 10, 15, 20, 25]
criterion = ['gini', 'entropy']

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'criterion': criterion}
rf = RandomForestClassifier()
rf_rs = RandomizedSearchCV(rf, param_distributions = random_grid, n_iter = 10, cv = 10, verbose = 2,
                           random_state = 66, n_jobs = -1, scoring = 'roc_auc')
rf_rs.fit(x_train, y_train)

rf_probs = rf_rs.predict_proba(x_test)
# keep positive probabilities 
rf_probs = rf_probs[:, 1]
# calculate scores
rf_rs_auc = roc_auc_score(y_test, rf_probs)
print('Random Forest AUC: ' + str(rf_rs_auc)) # print AUC 0.890, improved more
print(rf_rs.best_estimator_) # print the best hyperparameters
