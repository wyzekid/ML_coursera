import copy

import math
import numpy as np
import pandas as pd


def sum1(y,x1,x2,w,l):
    sum_result = 0
    for i in range(l):
        sum_result+=y[i]*x1[i]*(1-1/(1+math.exp(-y[i]*(w[0]*x1[i]+w[1]*x2[i]))))
    return sum_result


def sum2(y,x1,x2,w,l):
    sum_result = 0
    for i in range(l):
        sum_result+=y[i]*x2[i]*(1-1/(1+math.exp(-y[i]*(w[0]*x1[i]+w[1]*x2[i]))))
    return sum_result

def sigmoid(w,a,b):
    return 1/(1+math.exp(-w[0,0]*a-w[0,1]*b))

data = pd.read_csv('data-logistic.csv', names=['y', 'x1', 'x2'], header=None)
y = data['y']
X = data.iloc[:, 1:]
x1 = data['x1']
x2 = data['x2']
C = 10
k = 0.1
w = [0, 0]
eps = 1e-5
count = 0
error = 1
result = 0
length = len(y)
i = 0
coef1 = k/length
coef2 = k*C
while error > eps and count <= 10000:
    tmp = copy.copy(w)
    w[0] = w[0] + coef1*sum1(y, x1, x2, w, length)-coef2*w[0]
    w[1] = w[1] + coef1*sum2(y, x1, x2, w, length)-coef2*w[1]
    error = math.sqrt((w[0]-tmp[0])**2 + (w[1]-tmp[1])**2)
    print(count)
    print(error)
    count += 1
w = np.array(w).reshape(1, 2)
X = np.array(X).reshape(2,length)
x1 = np.array(x1).reshape(length,1)
x2 = np.array(x2).reshape(length,1)
answer = np.ones((length, 1))
for j in range(length):
    answer[j] = 1/(1+math.exp(-w[0,0]*x1[j]-w[0,1]*x2[j]))
    print(y[j], '\t', answer[j])

# with open("3-4.txt", "w") as file:
#     print(roc_auc_score(y, answer), end=' ', file=file)



