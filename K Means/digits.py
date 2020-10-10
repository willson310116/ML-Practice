from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import metrics

#  Using K Means clustering to deal with a unsupervising task
#  Time-consuming compare to other ml algorithm

digits = load_digits()
data = scale(digits.data)
y = digits.target
# k = len(np.unique(y))
k = 10
samples, features = data.shape


# print(features)

def bench_k_means(estimator, name, data):
    estimator.fit(data)
    output = f"{name}\t{round(estimator.inertia_)}\t" \
             f"{round(metrics.homogeneity_score(y, estimator.labels_), 3)}\t" \
             f"{round(metrics.completeness_score(y, estimator.labels_), 3)}\t" \
             f"{round(metrics.v_measure_score(y, estimator.labels_), 3)}\t" \
             f"{round(metrics.adjusted_rand_score(y, estimator.labels_), 3)}\t" \
             f"{round(metrics.adjusted_mutual_info_score(y, estimator.labels_), 3)}\t" \
             f"{round(metrics.silhouette_score(data, estimator.labels_, metric='euclidean'), 3)}"
    print(output)


clf = KMeans(n_clusters=k, init="random", n_init=10)

bench_k_means(clf, "1", data)
