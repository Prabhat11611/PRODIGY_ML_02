# Importing the required libraries
from sklearn.cluster import KMeans
import pandas as pd

# Loading the customer purchase history data
data = pd.read_csv('customer_purchase_history.csv')

# Preprocessing the data
X = data.iloc[:, 1:].values

# Creating the K-means clustering model
kmeans = KMeans(n_clusters=3, random_state=0)

# Fitting the model to the data
kmeans.fit(X)

# Predicting the clusters for the data points
clusters = kmeans.predict(X)

# Adding the cluster labels to the original data
data['Cluster'] = clusters

# Printing the resulting clusters
print(data)