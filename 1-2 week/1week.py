import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('2.csv', index_col='PassengerId').dropna(axis=0)
data_2 = pd.concat([data['Pclass'], data['Fare'], data['Age'], data['Sex']], axis=1)
print(data_2)
target = np.array(data['Survived']).reshape(183, 1)
data_2.Sex[data_2.Sex == "male"] = 1
data_2.Sex[data_2.Sex == "female"] = 0
X = data_2.as_matrix()
clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, target)
importances = clf.feature_importances_
print(importances)















'''print(data.head())
print('####################')
sexes = data['Sex'].value_counts()
print('Распределение полов ', sexes)
print('###################')
survived = data['Survived'].value_counts()
print('Пассажиров выжило  ', survived[1]/(survived[0]+survived[1]))
print('###################')
p_first_classes = data['Pclass'].value_counts()
print('Пассажиров первого класса ', p_first_classes[1]*100/891)
print('#####################')
ages = data['Age'].describe()
print(ages)
print('#####################')
print('Корелляция Пирсона', data['SibSp'].corr(data['Parch']))
print('#####################')
names = data['Name']
female_names = []
for i in names:
    if "Miss" in i:
        female_names.append(re.findall(r'\. (\w+)', i))
    if "Mrs" in i:
        female_names.append(re.findall(r'\((\w+)', i))
female_names_fin = ['c'] * len(female_names)
for i in range(len(female_names)):
    if len(female_names[i]) > 0:
        female_names_fin[i] = female_names[i][0]
c = Counter(female_names_fin)
print(c)'''
