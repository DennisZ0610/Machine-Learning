# [Dennis, Zhao]
# [20190903]
# [MMA 2021W]
# [Section 1]
# [MMA 869]
# [Aug 16, 2020]


# Answer to Question [1], Part [1]


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score




pd.options.display.width = 100
pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 100)

# TODO: import other packages as necessary


# Read in data from Uncle Steve's GitHub repository
url = 'https://raw.githubusercontent.com/stepthom/sandbox/master/data/jewelry_customers.csv'
df = pd.read_csv(url, error_bad_lines=False)


# TODO: insert code here to perform the given task. Don't forget to document your code!

### EDA
df.head()
df.shape
df.describe()
df.dtypes

### Check Missing Values
df.isnull().sum()

### Use Elbow Method to Determine the K number
inertias = {}
silhouettes = {}
for k in range(2, 11):
    kmeans = KMeans(init='k-means++', n_init=10, n_clusters=k, max_iter=1000, random_state=42).fit(df)
    inertias[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    silhouettes[k] = silhouette_score(df, kmeans.labels_, metric='euclidean')
   
plt.figure();
plt.grid(True);
plt.plot(list(inertias.keys()), list(inertias.values()));
plt.title('K-Means, Elbow Method')
plt.xlabel("Number of clusters, K");
plt.ylabel("Inertia"); #Inertia Plot

plt.figure();
plt.grid(True);
plt.plot(list(silhouettes.keys()), list(silhouettes.values()));
plt.title('K-Means, Elbow Method')
plt.xlabel("Number of clusters, K");
plt.ylabel("Silhouette");  # Silhouette Plot

### From the plot we can use K = 3 as the optimal number of clusters.
### Run the Kmeans model
k_means = KMeans(init='k-means++', n_clusters=3, n_init=5, random_state=123)
k_means.fit(df)
k_means_pred = k_means.predict(df)
k_means_pred
df['Cluster'] = pd.Series(k_means_pred, index = df.index) # Add cluster label to the df
df.head()
### I tried to change the hyperparameters, n_clusters, n_init and random_state. It shows the n_init and random_state
### have very low impact on the clustering result, however n_clusters does have huge impact, it changes the cluster completely.

### Print the summary for each cluster
df.groupby(['Cluster']).describe()
print(df[df['Cluster'] == 0].describe())
print(df[df['Cluster'] == 1].describe())
print(df[df['Cluster'] == 2].describe())

### Measure Goodness of Fit
k_means.labels_
silhouette_score(df, k_means.labels_) # silhouette score, the higher, the better
davies_bouldin_score(df, k_means.labels_) # Davies Bouldin Score, the higher, the better
metrics.calinski_harabasz_score(df, k_means.labels_) # Calinski-Harabasz Index, the higher, the better


###### Improving the model by Standardization ######
scaler = StandardScaler()
df_scale = scaler.fit_transform(df)
df_scale
df2 = df.drop(columns = 'Cluster')
df2.head()

### Use Elbow Method to Determine the K number
inertias = {}
silhouettes = {}
for k in range(2, 11):
    kmeans = KMeans(init='k-means++', n_init=10, n_clusters=k, max_iter=1000, random_state=42).fit(df_scale)
    inertias[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    silhouettes[k] = silhouette_score(df_scale, kmeans.labels_, metric='euclidean')
   
plt.figure();
plt.grid(True);
plt.plot(list(inertias.keys()), list(inertias.values()));
plt.title('K-Means, Elbow Method')
plt.xlabel("Number of clusters, K");
plt.ylabel("Inertia"); #Inertia Plot

plt.figure();
plt.grid(True);
plt.plot(list(silhouettes.keys()), list(silhouettes.values()));
plt.title('K-Means, Elbow Method')
plt.xlabel("Number of clusters, K");
plt.ylabel("Silhouette");  # Silhouette Plot

### From the plot we can use K = 5 as the optimal number of clusters, which is different with the original results
### Run the Kmeans model
k_means = KMeans(init='k-means++', n_clusters=5, n_init=10, random_state=123)
k_means.fit(df_scale)
k_means_pred = k_means.predict(df_scale)
k_means_pred
df2['Cluster'] = pd.Series(k_means_pred, index = df.index) # Add cluster label to the df
df2.head()

### Print the summary for each cluster
df2.groupby(['Cluster']).describe()
print(df2[df2['Cluster'] == 0].describe())
print(df2[df2['Cluster'] == 1].describe())
print(df2[df2['Cluster'] == 2].describe())
print(df2[df2['Cluster'] == 3].describe())
print(df2[df2['Cluster'] == 4].describe())

# Plot Clusters
plt.figure(figsize=(10, 7))  
plt.scatter(df2['Age'], df2['Income'], c=cluster.labels_, cmap = 'rainbow') 

### Measure Goodness of Fit
k_means.labels_
silhouette_score(df2, k_means.labels_) # silhouette score, the higher, the better
davies_bouldin_score(df2, k_means.labels_) # Davies Bouldin Score, the higher, the better
metrics.calinski_harabasz_score(df2, k_means.labels_) # Calinski-Harabasz Index, the higher, the better

### Also, hierarchical clustering is used here, maybe will get a better result
# Generate Dendrogram
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(df_scale, method='ward'))
plt.axhline(y=10, color='r', linestyle='--')

# Generate clusters
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  
cluster.fit_predict(df_scale)

# Plot clusters with Age and Income
plt.figure(figsize=(10, 7))  
plt.scatter(df['Age'], df['Income'], c=cluster.labels_, cmap = 'rainbow') 

# Check measuring scores
silhouette_score(df, cluster.labels_) # silhouette score, the higher, the better
davies_bouldin_score(df, cluster.labels_) # Davies Bouldin Score, the higher, the better
metrics.calinski_harabasz_score(df, cluster.labels_) # Calinski-Harabasz Index, the higher, the better
# It's the same with Kmeans