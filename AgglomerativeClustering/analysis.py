
# Agglomerative Clustering Analysis
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

data = pd.read_csv('customer_segmentation.csv')
print(data.describe())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

for linkage_method in ['single','complete','average','ward']:
    model = AgglomerativeClustering(n_clusters=4, linkage=linkage_method)
    labels = model.fit_predict(X_scaled)
    print(linkage_method, silhouette_score(X_scaled, labels))

Z = linkage(X_scaled, method='ward')
dendrogram(Z)
plt.show()
