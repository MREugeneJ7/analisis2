import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

le = LabelEncoder()

data = pd.read_csv("mallCustomers.csv")

le.fit(data["Genre"])
data["Genre"] = le.transform(data["Genre"])

data = data[["Annual Income", "Spending Score", "Age"]]
#Elbow Method
wcss = [] 
for i in range(1, 20): 
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(data) 
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 20), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.show()

km = KMeans(n_clusters=6, random_state=0, n_init="auto")
km.fit(data)


fig = plt.figure()
ax =fig.add_subplot(projection='3d')
ax.scatter(data["Annual Income"],data["Spending Score"], data["Age"], c=km.labels_.astype(float))
ax.scatter(
            km.cluster_centers_[:,0],
            km.cluster_centers_[:,1],
            km.cluster_centers_[:,2],
            color="red",
        )
plt.show()

wardClustering = AgglomerativeClustering(distance_threshold=None, linkage="ward", n_clusters=3, compute_distances=True)
model = wardClustering.fit(data)

plt.scatter(data["Annual Income"],data["Spending Score"], c=model.labels_.astype(float))
plt.show()

plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

singleClustering = AgglomerativeClustering(distance_threshold=6, linkage="single", n_clusters=None)
model2 = singleClustering.fit(data)

plt.scatter(data["Annual Income"],data["Spending Score"], c=model2.labels_.astype(float))
plt.show()


plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(model2, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()


centroidClustering = AgglomerativeClustering(distance_threshold=None, linkage="average", n_clusters=3, compute_distances=True)
model3 = centroidClustering.fit(data)

plt.scatter(data["Annual Income"],data["Spending Score"], c=model3.labels_.astype(float))
plt.show()


plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(model3, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()