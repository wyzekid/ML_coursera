import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

'''
#Задание 2_1
data = pd.read_csv('22.data', names=['Class', 'Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols',
                                           'Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity',
                                           'Hue', 'OD280','Proline'])
y = data['Class']
X = data.iloc[:, 1:]
X_scale = scale(X)
n_fold_gen = KFold(n_splits=5, shuffle=True, random_state=42)
res = {}
for k in range(1, 51):
    estimator = KNeighborsClassifier(n_neighbors=k)
    res[k] = np.mean(cross_val_score(estimator, X_scale, y, cv=n_fold_gen, scoring='accuracy'))
l = lambda x: x[1]
print(sorted(res.items(), key=l, reverse=True))'''

'''
#Задание 2_2
boston = load_boston()
X = boston.data
X_scale = scale(X)
y = boston.target
p = np.linspace(1, 10, num=200)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
res = {}
for i in p:
    estimator = KNeighborsRegressor(n_neighbors=5, weights='distance', p=i)
    res[i] = np.mean(cross_val_score(estimator, X_scale, y, cv=kf, scoring='neg_mean_squared_error'))
l = lambda x: x[1]
print(sorted(res.items(), key=l, reverse=True))
'''


data_train = pd.read_csv('perceptron-train.csv', names=['y_train', 'x1','x2'], header=None)
data_test = pd.read_csv('perceptron-test.csv', names=['y_test', 'x1', 'x2'], header=None)
y_train = data_train['y_train']
X_train = data_train.iloc[:, 1:]
y_test = data_test['y_test']
X_test = data_test.iloc[:, 1:]
clf = Perceptron(random_state=241)
scaler = StandardScaler()
clf.fit(X_train, y_train)
acc_before = accuracy_score(y_test, clf.predict(X_test))
print(acc_before)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf.fit(X_train_scaled, y_train)
acc_after = accuracy_score(y_test, clf.predict(X_test_scaled))
print(acc_after)
print(acc_after-acc_before)

