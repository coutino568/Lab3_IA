from sklearn.cluster import KMeans

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from matplotlib.patches import Ellipse

from sklearn.mixture import GaussianMixture



# air_pressure	
# air_temp	
# avg_wind_direction	
# avg_wind_speed	
# max_wind_direction	
# max_wind_speed	
# min_wind_direction	
# min_wind_speed	
# rain_accumulation	
# rain_duration	
# relative_humidity



dataset = pd.read_csv('minute_weather.csv')
df2 = pd.DataFrame(dataset)


print(df2)

X = df2[['relative_humidity', 'air_temp']]
kmeans= KMeans(4, random_state=0)
print (X)
Y = df2[['avg_wind_speed']]
print(Y)





## metodo para determinar el numero de clusters
N = np.arange(1,10)
state= 0
gmm = [ GaussianMixture( n, covariance_type ="full", random_state=state).fit(X)  for n in N]
for model in gmm :
	print(model.bic(X))

plt.plot(N, [model.bic(X) for model in gmm] , "o-", label ="BIC")
plt.xlabel("# de clusters")
plt.ylabel("BIC score")
plt.title("Gráfico de Codo Gaussian Mixture models ")
plt.savefig('codo Gaussian mixture models.png')
plt.show()





## parte de EMM
# labels = kmeans.fit(X).predict(X)
# plt.scatter(X['relative_humidity'],Y['avg_wind_speed'] , c= labels, s=3)
# plt.show()


# gm = GaussianMixture( covariance_type = 'full',)


