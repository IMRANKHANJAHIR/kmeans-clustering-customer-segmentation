import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


data = pd.read_csv(r"C:\30 day intesnhip online\Mall_Customers.csv")
print(data.head())

features = data.iloc[:, [3, 4]] 
pca = PCA(2)
pca_features = pca.fit_transform(features)

plt.figure(figsize=(8, 5))
plt.scatter(pca_features[:, 0], pca_features[:, 1], c='blue', s=50, alpha=0.5)
plt.title("PCA of Customer Data (Before Clustering)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()

inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(features)

plt.figure(figsize=(8, 5))
plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=labels, cmap='viridis', s=50)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.title('K-Means Clustering')
plt.show()

sil_score = silhouette_score(features, labels)
print(f"Silhouette Score: {sil_score:.2f}")
