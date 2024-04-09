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
km = KMeans(n_clusters=5, random_state=0, n_init="auto")
km.fit(data)

plt.scatter(data["Age"],data["Annual Income"],data["Spending Score"], c=km.labels_.astype(float))
plt.show()

wardClustering = AgglomerativeClustering(distance_threshold=0, linkage="ward", n_clusters=None)
model = wardClustering.fit(data)

plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

singleClustering = AgglomerativeClustering(distance_threshold=0, linkage="single", n_clusters=None)
model2 = singleClustering.fit(data)

plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(model2, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()


centroidClustering = AgglomerativeClustering(distance_threshold=0, linkage="average", n_clusters=None)
model3 = centroidClustering.fit(data)

plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(model3, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()