import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

#Обучающая выборка
data_train = pd.read_csv('salary-train.csv')
data_train_lower = []
for i in range(len(data_train['FullDescription'])):
    data_train_lower.append(data_train.iloc[i][0].lower())
data_train['FullDescription'] = np.array(data_train_lower)
#print(data_train)
data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
vect = TfidfVectorizer(min_df=5)
idf = vect.fit_transform(data_train['FullDescription'])
data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)
enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
print(type(idf), type(X_train_categ))
X_train = hstack([idf, X_train_categ])
print(X_train.shape[0], X_train.shape[1])
y_train = data_train['SalaryNormalized']
clf = Ridge(alpha=1, random_state=241)
clf.fit(X_train, y_train)

#Тестовая выборка
data_test = pd.read_csv('salary-test-mini1.csv')
data_test_lower = []
for i in range(len(data_test['FullDescription'])):
    data_test_lower.append(data_test.iloc[i][0].lower())
data_test['FullDescription'] = np.array(data_test_lower)
data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
idf_test = vect.transform(data_test['FullDescription'])
print(idf_test.shape[0], idf_test.shape[1])
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
print(X_test_categ.shape[0], X_test_categ.shape[1])
print(type(idf_test), type(X_test_categ))
X_test = hstack([idf_test, X_test_categ])
print(X_test.shape[0], X_test.shape[1])
print(clf.predict(X_test))


