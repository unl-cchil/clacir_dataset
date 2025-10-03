import case_interfacing as case
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

dataset, annotation = case.load_csv_dataset(r"C:\GitHub\hai_stress\software\python_processing\datasets\case_raw")

# k_mean_data_x = []
# k_mean_data_y = []
#
# for a in annotation:
#     k_mean_data_x.extend(a[1])
#     k_mean_data_y.extend(a[2])
#     break
#
# print("Clustering data with k-means...")
# Kmean = KMeans(n_clusters=4)
# Kmean.fit(np.column_stack((k_mean_data_x, k_mean_data_y)))
# y_pred = Kmean.predict(np.column_stack((k_mean_data_x, k_mean_data_y)))
# plt.scatter(k_mean_data_x, k_mean_data_y, c=y_pred, s=50)
# centers = Kmean.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], marker='s', s=100)
#
# print("Getting Voronoi from centers...")
# vor = Voronoi(centers)
# fig = voronoi_plot_2d(vor, plt.gca())
#
# print("Calculating Silhouette score...")
# s_score = metrics.silhouette_score(np.column_stack((k_mean_data_x, k_mean_data_y)), Kmean.labels_, metric='euclidean')
#
# print(f"Silhouette Score: {s_score}")
# print(f"Centroids: {centers}")
# print(f"Vertices: {vor.vertices}")
# print(f"Regions: {vor.regions}")
# print(f"Ridge Vertices: {vor.ridge_vertices}")
# print(f"Ridge Points: {vor.ridge_points}")
#
# plt.xlim([0, 10])
# plt.ylim([0, 10])
# plt.show()
