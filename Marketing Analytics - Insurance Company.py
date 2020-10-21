
import time
import sklearn
import datetime
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib as mpl
import statsmodels.api as sm
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import scipy.cluster.hierarchy as shc

from math import sqrt
from scipy import stats
from pycaret.clustering import *
from kmodes.kmodes import KModes
from sklearn import preprocessing
from scipy.spatial import distance
from pycaret.classification import *
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


from IPython.core.interactiveshell import InteractiveShell

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from yellowbrick.cluster import SilhouetteVisualizer, InterclusterDistance, KElbowVisualizer



from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier, BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier



###### Import Data
df = pd.read_csv("E:\MMA\Courses\MMA 831 - Marketing Analytics\Final Project\Marketing-Customer-Value-Analysis.csv")
# drop customer, policy and effective to date
df = df.rename(columns = str.lower) # convert all column names to lowercase
df = df.drop(columns = {'customer', 'policy', 'effective to date'})

# Set max row display
pd.set_option('display.max_row', 1000)
# Set max column width to 50
pd.set_option('display.max_columns', 50)

pd.options.display.width = 100
### check missing value
df.isnull().sum() # it's clean

###### EDA
df.head()
df.info()
df.shape
df.dtypes
df.describe()

df.state.value_counts()
df.state.unique()
df.response.value_counts()
df.coverage.value_counts()
df.education.value_counts()
df.employmentstatus.value_counts()
df.gender.value_counts()
df['location code'].value_counts()
df['marital status'].value_counts()
df['policy type'].value_counts()
df['renew offer type'].value_counts()
df['sales channel'].value_counts()
df['vehicle class'].value_counts()
df['vehicle size'].value_counts()

df['customer lifetime value'].describe()
df['income'].describe()
df['monthly premium auto'].describe()
df['months since last claim'].describe()
df['months since policy inception'].describe()
df['number of open complaints'].describe()
df['number of policies'].describe()
df['total claim amount'].describe()

# change 'response' from Yes, No to 1, 0
df['response'] = df['response'].replace(['No'], 0)
df['response'] = df['response'].replace(['Yes'], 1)


###### check for correlation
corr = df.corr()
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

###### Visualization
sns.countplot(x = 'state', data = df)
sns.countplot(x = 'response', data = df)
sns.countplot(x = 'education', data = df)
sns.countplot(x = 'employmentstatus', data = df)
sns.countplot(x = 'gender', data = df)
sns.countplot(x = 'location code', data = df)
sns.countplot(x = 'marital status', data = df)
sns.countplot(x = 'policy type', data = df)
sns.countplot(x = 'renew offer type', data = df)
sns.countplot(x = 'sales channel', data = df)
sns.countplot(x = 'vehicle class', data = df)
sns.countplot(x = 'vehicle size', data = df)

plt.hist(df['customer lifetime value'], bins = 'auto')
plt.hist(df['income'], bins = 'auto')
plt.hist(df['monthly premium auto'], bins = 'auto')
plt.hist(df['months since last claim'], bins = 'auto')
plt.hist(df['months since policy inception'], bins = 'auto')
plt.hist(df['number of open complaints'], bins = 'auto')
plt.hist(df['number of policies'], bins = 'auto')
plt.hist(df['total claim amount'], bins = 'auto')

sns.countplot("response", hue = 'gender', data = df)
sns.countplot("response", hue = 'state', data = df)
sns.countplot("response", hue = 'coverage', data = df)
sns.countplot("response", hue = 'education', data = df)
sns.countplot("response", hue = 'employmentstatus', data = df)
sns.countplot("response", hue = 'location code', data = df)
sns.countplot("response", hue = 'marital status', data = df)
sns.countplot("response", hue = 'policy type', data = df)
sns.countplot("response", hue = 'renew offer type', data = df)
sns.countplot("response", hue = 'sales channel', data = df)
sns.countplot("response", hue = 'vehicle class', data = df)
sns.countplot("response", hue = 'vehicle size', data = df)

###### clustering (2 variables: 'customer lifetime value', 'total claim amount')
### Kmeans
### from the above plots, 'customer lifetime value' and 'total claim amount' are very similar 
df_cluster = df.loc[:,['customer lifetime value', 'total claim amount']]
df_cluster.head()
scaler = StandardScaler()
features = ['customer lifetime value','total claim amount']
df_cluster[features] = scaler.fit_transform(df_cluster[features])
df_cluster.describe().transpose()

# Elbow Method: determine the appropriate number of K
inertias = {}
silhouettes = {}
for k in range(2, 11):
    kmeans = KMeans(init='k-means++', n_init=10, n_clusters=k, max_iter=1000, random_state=42).fit(df_cluster)
    inertias[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    silhouettes[k] = silhouette_score(df_cluster, kmeans.labels_, metric='euclidean')
   
plt.figure();
plt.grid(True);
plt.plot(list(inertias.keys()), list(inertias.values()));
plt.title('K-Means, Elbow Method')
plt.xlabel("Number of clusters, K");
plt.ylabel("Inertia");

plt.figure();
plt.grid(True);
plt.plot(list(silhouettes.keys()), list(silhouettes.values()));
plt.title('K-Means, Elbow Method')
plt.xlabel("Number of clusters, K");
plt.ylabel("Silhouette");
# Choose k = 5 is good enough.

# plot the data
plt.style.use('default');
plt.figure(figsize=(16, 10));
plt.grid(True);
plt.scatter(df_cluster.iloc[:, 0], df_cluster.iloc[:, 1], c="black", s=2);
plt.title("IBM Watson Data", fontsize=16);
plt.xlabel('Customer Lifetime Value', fontsize=16);
plt.ylabel('Total Claim Amount', fontsize=16);
plt.xticks(fontsize=16);
plt.yticks(fontsize=16);

# Kmeans Plot, K = 5
k_means = KMeans(init='k-means++', n_clusters=5, n_init=10, random_state=42)
k_means.fit(df_cluster)
k_means.labels_
k_means.cluster_centers_

plt.figure(figsize=(16, 10));
plt.grid(True);

sc = plt.scatter(df_cluster.iloc[:, 0], df_cluster.iloc[:, 1], s=2, c=k_means.labels_);
#plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], marker='x', s=5, c="black")
plt.title("K-Means (K=5)", fontsize=20);
plt.xlabel('Customer Lifetime Value', fontsize=22);
plt.ylabel('Total Claim Amount', fontsize=22);
plt.xticks(fontsize=18);
plt.yticks(fontsize=18);

for label in k_means.labels_:
    plt.text(x=k_means.cluster_centers_[label, 0], y=k_means.cluster_centers_[label, 1], s=label, fontsize=32, 
             horizontalalignment='center', verticalalignment='center', color='black',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1', alpha=0.02));

### Internal Validation Metrics
k_means.inertia_
silhouette_score(df_cluster, k_means.labels_)
sample_silhouette_values = silhouette_samples(df_cluster, k_means.labels_)
sizes = 200*sample_silhouette_values

plt.figure(figsize=(16, 10));
plt.grid(True);

plt.scatter(df_cluster.iloc[:, 0], df_cluster.iloc[:, 1], s=sizes, c=k_means.labels_)
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], marker='x', s=500, c="black")

plt.title("K-Means (Dot Size = Silhouette Distance)", fontsize=20);
plt.xlabel('Customer Lifetime Value', fontsize=22);
plt.ylabel('Total Claim Amount', fontsize=22);
plt.xticks(fontsize=18);
plt.yticks(fontsize=18);

### DBSCAN
# Use Elbow Method to find the appropriate eps
silhouettes = {}
for eps in np.arange(0.1, 0.6, 0.1):
    db = DBSCAN(eps=eps, min_samples=3).fit(df_cluster)
    silhouettes[eps] = silhouette_score(df_cluster, db.labels_, metric='euclidean')
    

plt.figure();
plt.plot(list(silhouettes.keys()), list(silhouettes.values()));
plt.title('DBSCAN, Elbow Method')
plt.xlabel("Eps");
plt.ylabel("Silhouette");
plt.grid(True); # eps = 0.3 is good enough

# DBSCAN Plot
db = DBSCAN(eps=0.3, min_samples=5)
db.fit(df_cluster)
db.labels_
silhouette_score(df_cluster, db.labels_)

plt.figure(figsize=(16, 10));
plt.grid(True);
unique_labels = set(db.labels_)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))];

for k in unique_labels:
    if k == -1:        # Black used for noise.
        col = [0, 0, 0, 1]
    else:
        col = colors[k]

    xy = df_cluster[db.labels_ == k]
    plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=10);

plt.title('');
plt.title("DBSCAN (n_clusters = {:d}, black = outliers)".format(len(unique_labels)), fontsize=20);
plt.xlabel('Customer Lifetime Value', fontsize=22);
plt.ylabel('Total Claim Amount', fontsize=22);
plt.xticks(fontsize=18);
plt.yticks(fontsize=18);

###### Hierarchical Clustering (all variables)
df_cluster2 = df
lb = LabelEncoder() # convert all categorical variables to numeric 
df_cluster2['response'] = lb.fit_transform(df_cluster2['response'])
df_cluster2['renew offer type'] = lb.fit_transform(df_cluster2['renew offer type'])
df_cluster2['state'] = lb.fit_transform(df_cluster2['state'])
df_cluster2['coverage'] = lb.fit_transform(df_cluster2['coverage'])
df_cluster2['education'] = lb.fit_transform(df_cluster2['education'])
df_cluster2['employmentstatus'] = lb.fit_transform(df_cluster2['employmentstatus'])
df_cluster2['gender'] = lb.fit_transform(df_cluster2['gender'])
df_cluster2['location code'] = lb.fit_transform(df_cluster2['location code'])
df_cluster2['marital status'] = lb.fit_transform(df_cluster2['marital status'])
df_cluster2['policy type'] = lb.fit_transform(df_cluster2['policy type'])
df_cluster2['sales channel'] = lb.fit_transform(df_cluster2['sales channel'])
df_cluster2['vehicle class'] = lb.fit_transform(df_cluster2['vehicle class'])
df_cluster2['vehicle size'] = lb.fit_transform(df_cluster2['vehicle size'])
df_cluster2.head()

df_cluster_scaled = normalize(df_cluster2) # normalize the data
df_cluster_scaled = pd.DataFrame(df_cluster_scaled, columns=df_cluster2.columns)
df_cluster_scaled.head()

# Dendrograms
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(df_cluster_scaled, method='ward'))
# From the plot, we can set the threshold as 20
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(df_cluster_scaled, method='ward'))

plt.axhline(y=20, color='r', linestyle='--') # 2 clusters
plt.axhline(y=15, color='b', linestyle='--') # 3 clusters
plt.axhline(y=8, color='g', linestyle='--') # 4 clusters
plt.axhline(y=5.2, color='y', linestyle='--') # 5 clusters

# plot clusters
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward') # cluster is 2
cluster.fit_predict(df_cluster_scaled)
plt.figure(figsize=(10, 7))  
plt.scatter(df_cluster_scaled['customer lifetime value'], df_cluster_scaled['total claim amount'], c=cluster.labels_) 

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward') # cluster is 3
cluster.fit_predict(df_cluster_scaled)
plt.figure(figsize=(10, 7))  
plt.scatter(df_cluster_scaled['customer lifetime value'], df_cluster_scaled['total claim amount'], c=cluster.labels_) 

cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  # cluster is 4
cluster.fit_predict(df_cluster_scaled)
plt.figure(figsize=(10, 7))  
plt.scatter(df_cluster_scaled['customer lifetime value'], df_cluster_scaled['total claim amount'], c=cluster.labels_) 

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  # cluster is 4
cluster.fit_predict(df_cluster_scaled)
plt.figure(figsize=(10, 7))  
plt.scatter(df_cluster_scaled['customer lifetime value'], df_cluster_scaled['total claim amount'], c=cluster.labels_) 

### Clustering using PyCaret
dum2 = ['state', 'coverage', 'education', 'employmentstatus', 'gender', 'location code', 'marital status', 'policy type', 'renew offer type', 'sales channel', 'vehicle class', 'vehicle size']
dum_prefix2= ['state',  'coverage', 'education', 'employmentstatus', 'gender', 'location code', 'marital status', 'policy type', 'renew offer type', 'sales channel', 'vehicle class', 'vehicle size']

df_cluster3 = pd.get_dummies(df,columns=dum2,prefix=dum_prefix2,dtype= 'int64')
df_cluster3.head().transpose()
cluster3 = setup(df_cluster3, ignore_features = ['response'],remove_multicollinearity = True, multicollinearity_threshold = 0.9, session_id=123, log_experiment=True, log_plots = True, 
             experiment_name='df_exp',transformation=True )
cluster3
models()
kmeans = create_model('kmeans', num_clusters = 5)
hclust = create_model('hclust', num_clusters = 5)
kmodes = create_model('kmodes', num_clusters = 5)
kmeans_results = assign_model(kmeans)
kmeans_results.head()
kmeans_results.to_csv("E:\MMA\Courses\MMA 831 - Marketing Analytics\Final Project\Cluster Result.csv")
plot_model(kmeans, plot = 'tsne')
plot_model(kmeans, plot = 'elbow')
plot_model(kmeans, plot = 'silhouette')
plot_model(kmeans, plot = 'distance')
plot_model(kmeans, plot = 'distribution')

X = kmeans_results
X_df = pd.DataFrame(X, columns=X.columns)
X_df.head()
X_df['Cluster']
X_df.head()

cl_group = X_df.groupby(['Cluster']).agg('describe')
cl_group
cl_group.info()
cl_group['gender_M']
cl_group['income']

kmeans_results.drop(columns = ['response'], inplace=True) 
labels = kmeans_results['Cluster'].str.replace('Cluster ','')

X = kmeans_results.drop(columns ='Cluster')
pd.set_option("display.precision", 2)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

def stats_to_df(d):
    tmp_df = pd.DataFrame(columns=X.columns)
    
    tmp_df.loc[0] = d.minmax[0]
    tmp_df.loc[1] = d.mean
    tmp_df.loc[2] = d.minmax[1]
    tmp_df.loc[3] = d.variance
    tmp_df.loc[4] = d.skewness
    tmp_df.loc[5] = d.kurtosis
    tmp_df.index = ['Min', 'Mean', 'Max', 'Variance', 'Skewness', 'Kurtosis'] 
    return tmp_df.T

print('All Data:')
print('Number of Instances: {}'.format(X.shape[0]))
#d = stats.describe(X, axis=0)
d = stats.describe(X)
display(stats_to_df(d))

for i, label in enumerate(set(labels)):
    d = stats.describe(X[labels==label], axis=0)
    print('\nCluster {}:'.format(label))
    print('Number of Instances: {}'.format(d.nobs))
    display(stats_to_df(d))

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

K=5
means = np.zeros((K, X.shape[1]))

for i, label in enumerate(set(labels)):
    means[i,:] = X[labels==label].mean(axis=0)
    print('\nCluster {} (n={}):'.format(label, sum(labels==label)))
    print(means[i,:])
    
#means

for i, label in enumerate(set(labels)):
    X_tmp= X
    exemplar_idx = distance.cdist([means[i]], X).argmin()
   
    print('\nCluster {}:'.format(label))
    print("  Examplar ID: {}".format(exemplar_idx))
    print("  Label: {}".format(labels[exemplar_idx]))
    print("  Features:")
    display(df.iloc[[exemplar_idx]])


###### Classification
dum = ['state', 'coverage', 'education', 'employmentstatus', 'gender', 'location code', 'marital status', 'policy type', 'renew offer type', 'sales channel', 'vehicle class', 'vehicle size']
dum_prefix= ['state',  'coverage', 'education', 'employmentstatus', 'gender', 'location code', 'marital status', 'policy type', 'renew offer type', 'sales channel', 'vehicle class', 'vehicle size']

df_class = pd.get_dummies(df,columns=dum,prefix=dum_prefix,dtype= 'int64')
df_class.head().transpose()
class1 = setup(df_class, ignore_features = [],remove_multicollinearity = True, multicollinearity_threshold = 0.9, session_id=123, log_experiment=True, log_plots = True, 
             target='response', feature_selection = True, experiment_name='df_exp',remove_outliers=True,
                outliers_threshold=0.05,transformation=True )
class1
type(class1)
x = class1[2]
print(x.keys())

models() # create models
compare_models(blacklist=['catboost'])

# Decision Tree Classifier
dt = create_model('dt')
tune_dt = tune_model(dt)
tune_dt

# Random Forest Classifier
rf = create_model('rf')
tune_rf = tune_model(rf)
tune_rf

# Light Gradient Boosting Machine
lightgbm = create_model('lightgbm')
tune_lightgbm = tune_model(lightgbm)

# Extra Trees Classifier
et = create_model('et')
tune_et = tune_model(et)
tune_et

plot_model(et)
plot_model(tune_et)
plot_model(et, plot = 'boundary')
plot_model(tune_et, plot = 'boundary')
plot_model(tune_et, plot = 'class_report')

# Feature Importance
importances = tune_et.feature_importances_
std = np.std([tree.feature_importances_ for tree in tune_et.estimators_],
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

###### define function and combine all the ML techniques, split into train and test
def do_all_for_dataset(dataset_name, df, target_col, drop_cols=[]):

    # If target_col is an object, convert to numbers
    if df[target_col].dtype == 'object':
      df[target_col] =  df[target_col].astype('category').cat.codes

    # OHE all categorical columns
    cat_cols = list(df.select_dtypes(include=['object']).columns) 
    if target_col in cat_cols: cat_cols.remove(targe_col)
    if len(cat_cols) > 0:
      df = pd.concat([df,pd.get_dummies(df[cat_cols])],axis=1)

    # Split into X and y
    X = df.drop(drop_cols + cat_cols + [target_col], axis=1)
    y = df[target_col]

    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print('Y (train) counts:')
    print(y_train.value_counts())
    print('Y (test) counts:')
    print(y_test.value_counts())
    
    nb = GaussianNB()   
    lr = LogisticRegression(random_state=42, solver='lbfgs', max_iter=5000)
    dt = DecisionTreeClassifier(random_state=42)
    knn = KNeighborsClassifier(n_neighbors=7)

    rf = RandomForestClassifier(random_state=42, n_estimators=200)
    ada = AdaBoostClassifier(random_state=42, n_estimators=200)

    est_list = [('DT', dt), ('LR', lr), ('NB', nb), ('RF', rf), ('ADA', ada)]
       
    dict_classifiers = {
        "LR": lr, 
        "NB": nb,
        "DT": dt,
        "KNN": knn,
        "Voting": VotingClassifier(estimators = est_list, voting='soft'),
        "Bagging": BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, random_state=42),
        "RF": rf,
        "ExtraTrees": ExtraTreesClassifier(random_state=42, n_estimators=200),
        "Adaboost": ada,
        "GBC": GradientBoostingClassifier(random_state=42, n_estimators=200),
        "Stacking": StackingClassifier(estimators=est_list, final_estimator=LogisticRegression())
        }
    
    model_results = list()
    
    for model_name, model in dict_classifiers.items():
        y_pred = model.fit(X_train, y_train).predict(X_test)        
        accuracy       = accuracy_score(y_test, y_pred)
        f1             = f1_score(y_test, y_pred)
        recall         = recall_score(y_test, y_pred)
        precision      = precision_score(y_test, y_pred)
        roc_auc        = roc_auc_score(y_test, y_pred)
    
        df = pd.DataFrame({"Dataset"   : [dataset_name],
                           "Method"    : [model_name],
                           "Accuracy"  : [accuracy],
                           "Recall"    : [recall],
                           "Precision" : [precision],
                           "F1"        : [f1],
                           "AUC"       : [roc_auc],
                          })
        model_results.append(df)
   

    dataset_results = pd.concat([m for m in model_results], axis = 0).reset_index()

    dataset_results = dataset_results.drop(columns = "index",axis =1)
    dataset_results = dataset_results.sort_values(by=['F1'], ascending=False)
    dataset_results['Rank'] = range(1, len(dataset_results)+1)
    
    return dataset_results
results = list()

r = do_all_for_dataset( 'IBM Watson', df, target_col='response', drop_cols=[])
results.append(r)
r

###### Principal Component Analysis
features = ['customer lifetime value', 'income', 'monthly premium auto', 'months since last claim', 'months since policy inception', 'number of open complaints', 'number of policies', 'total claim amount']
x = df.loc[:, features].values
y = df.loc[:, ['response']].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components= 2)
principalcomponents = pca.fit_transform(x)
principalDF = pd.DataFrame(data = principalcomponents, columns = ['PC 1', 'PC 2'])
finaldf = pd.concat([principalDF, df[['response']]], axis = 1)

# Visualize 2D Projection
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC 1', fontsize = 15)
ax.set_ylabel('PC 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
responses = ['Yes', 'No']
colors = ['r', 'b']
for response, color in zip(responses, colors):
    indicesToKeep = finaldf['response'] == response
    ax.scatter(finaldf.loc[indicesToKeep, 'PC 1']
               , finaldf.loc[indicesToKeep, 'PC 2']
               , c = color
               , s = 50)
ax.legend(response)
ax.grid()

#
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

pca = PCA()
x_train = pca.fit_transform(x_train) ### only apply pca transform, not sc.fit, for PCA variance plot
x_test = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_
explained_cum_var = np.cumsum(explained_variance)
explained_variance

# variance plot
plt.figure(1,figsize=(6,4))
plt.clf()  
plt.plot(explained_cum_var, linewidth=2)  
plt.axis('tight')  
plt.grid() 
plt.xlabel('n_components') 
plt.ylabel('Cumulative_Variance_explained')  
plt.show()

classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)
#Performance Evaluation
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

###### Predicting Customer Lifetime Value using Regression
df2 = pd.get_dummies(df)
df2.head()
x = df2.drop(columns = 'customer lifetime value')
x.head()
y = df2['customer lifetime value']
y.head()

# OLS Regression
x_1 = sm.add_constant(x)
model = sm.OLS(y, x_1).fit()
model.pvalues
model.summary() # R^2 is 0.167

# use backward elimination
cols = list(x.columns)
pmax = 1
while (len(cols)>0):
    p= []
    x_1 = x[cols]
    x_1 = sm.add_constant(x_1)
    model = sm.OLS(y,x_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)

x2 = x.loc[:, selected_features_BE]
x_2 = sm.add_constant(x2)
model2 = sm.OLS(y, x_2).fit()
model2.pvalues
model2.summary() 
# customer lifetime value = 71.1489*(monthly premium auto) - 262.5786*(number of open complaints) + 
#                           58.4798*(number of policies) + 350.4839*(education_High School or Below) + 
#                           701.0618*(employmentstatus_Employed) - 381.2508*(marital status_Single) + 
#                           665.6740*(policy type_Special Auto) + 1024.9877*(renew offer type_Offer1) + 
#                           531.0210*(renew offer type_Offer3) - 719.5962*(vehicle class_Four-Door Car) - 
#                           616.6373*(vehicle class_Two-Door Car) + 824.5237
# R^2 is the same, 0.167, which indicates a possible overfitting, and also linear model is not good in this case.


