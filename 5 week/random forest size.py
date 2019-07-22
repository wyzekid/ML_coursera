import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score


def myscorer(estimator, X, y):
    return r2_score(y, estimator.predict(X))

data = pd.read_csv('abalone.csv')
y = data['Rings']
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
X = data.iloc[:, 0:8]
cv = KFold(n_splits=5, shuffle=True, random_state=1)
res = {}
for i in range(1, 51):
    clf = RandomForestRegressor(random_state=1, n_estimators=i)
    clf.fit(X, y)
    #r2 = r2_score(y, clf.predict(X))
    #sc = make_scorer(myscorer(clf, X, y))
    res[i] = np.mean(cross_val_score(clf, X, y, cv=cv, scoring='r2'))
print(res)