import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

data = pd.read_csv('close_prices.csv')
X = data.iloc[:, 1:]
pca = PCA(n_components=10)
pca.fit_transform(X)
disp = 0
for i in pca.explained_variance_ratio_:
    disp += i
    if disp >= 0.9:
        print(i)
        break
print(pca.explained_variance_ratio_)
r = pca.transform(X)
idj_counted = []
for i in range(r.shape[0]):
    idj_counted.append(r[i][0])
print(idj_counted)
compons = pca.components_[0]
generator = enumerate(compons)
out = [i for i, x in generator if x == max(pca.components_[0])]
print(out)


data = pd.read_csv('djia_index.csv')
ind_dj = data.iloc[:, 1:]
ind_dj = np.array(ind_dj).reshape(374,1)
idj_counted = np.array(idj_counted).reshape(374,1)
print(np.corrcoef(idj_counted, ind_dj, rowvar=False))