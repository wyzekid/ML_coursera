import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


def sigmafun(y):
  return 1 / (1 + np.exp(-y))


data = pd.read_csv('gbm-data.csv')
y = data.iloc[:, 0]
X = data.iloc[:, 1:]
y = np.array(y)
X = np.array(X)
print(y.shape, X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                      test_size=0.8,
                                      random_state=241)
learning = [1, 0.5, 0.3, 0.2, 0.1]
test_loss = []
train_loss = []
result = {}
# for rate in learning:
#     clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=rate)
#     clf.fit(X_train, y_train)
#     for i, y_decision in enumerate(clf.staged_decision_function(X_train)):
#         y_pred = 1.0 / (1.0 + np.exp(-y_decision))
#         l = log_loss(y_train, y_pred)
#         train_loss.append(l)
#     for i, y_decision in enumerate(clf.staged_decision_function(X_test)):
#         y_pred = 1.0 / (1.0 + np.exp(-y_decision))
#         l = log_loss(y_test, y_pred)
#         test_loss.append(l)
#         result[i] = l
#     lamb = lambda x: x[1]
#     print('rate = ', rate, sorted(result.items(), key=lamb, reverse=False))
#     print(train_loss)
#     print(test_loss)
#     plt.figure()
#     plt.plot(train_loss, 'g', linewidth=math.sqrt(rate * 20))
#     plt.plot(test_loss, 'r', linewidth=math.sqrt(rate * 20))
#     plt.legend(['train', 'test'])
#     plt.draw()
#     train_loss.clear()
#     test_loss.clear()
# plt.show()
clf1 = RandomForestClassifier(n_estimators=37, random_state=241)
clf1.fit(X_train, y_train)
desic = clf1.predict_proba(X_test)
res = log_loss(y_test, desic)
print(res)

