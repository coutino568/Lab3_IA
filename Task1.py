import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sb
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import sklearn.preprocessing
import scipy.cluster.hierarchy as sch
import skfuzzy as fuzz
import pylab
import sklearn.mixture as mixture
import pyclustertend
import random
import datetime
from mpl_toolkits.mplot3d import Axes3D

# Leemos la base de datos
dataset = pd.read_csv('minute_weather.csv')
a = (dataset.iloc[:, 1])
date = a.astype(str).str[:4]
# Agregamos una nueva columna para que pueda organizar los datos como fecha
dataset["Fecha"] = date
# Sabemos cuantas filas hay de cada categoria
print(dataset.groupby('Fecha').size())
# Primero debemos de saber si vale la pena el agrupamiento
random.seed(123)


X = np.array(
    dataset[['air_pressure', 'air_temp']])
Y = np.array(dataset[['Fecha']])


X_scale = sklearn.preprocessing.scale(X)

numeroClusters = range(1, 11)
wcss = []
for i in numeroClusters:
    kmeans = cluster.KMeans(n_clusters=i)
    kmeans.fit(X_scale)
    wcss.append(kmeans.inertia_)

plt.plot(numeroClusters, wcss)
plt.xlabel("Número de clusters")
plt.ylabel("Score")
plt.title("Gráfico de Codo")
plt.savefig('codo.png')
plt.show()

km = cluster.KMeans(n_clusters=4).fit(X)
print(km)
centroides = km.cluster_centers_
print(centroides)
date = km.predict(X)
plt.scatter(X[date == 0, 0], X[date == 0, 1],
            s=100, c='red', label="Cluster 1")
plt.scatter(X[date == 1, 0], X[date == 1, 1],
            s=100, c='blue', label="Cluster 2")
plt.scatter(X[date == 2, 0], X[date == 2, 1],
            s=100, c='green', label="Cluster 3")
plt.scatter(X[date == 3, 0], X[date == 3, 1],
            s=100, c='purple', label="Cluster 4")
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[
            :, 1], s=300, c="yellow", marker="*", label="Centroides")
plt.title("Grupos de clima")
plt.xlabel("air_pressure")
plt.ylabel("air_temp")
plt.legend()
plt.savefig("Cluster.png")
plt.show()
