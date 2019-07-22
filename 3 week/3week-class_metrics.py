import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,precision_recall_curve

data = pd.read_csv('classification.csv', names=['true', 'pred'], header = 0)
true_column = data['true']
pred_column = data['pred']
length = len(true_column)
TP = FP = FN = TN = 0
for i in range(length):
    if true_column[i] == 1 and pred_column[i]==1:
        TP+=1
    if true_column[i]==0 and pred_column[i]==1:
        FP+=1
    if true_column[i] == 1 and pred_column[i]==0:
        FN+=1
    if true_column[i] == 0 and pred_column[i]==0:
        TN+=1
print(TP, FP, FN, TN)
with open("3-5.txt", "w") as file:
     print(TP, FP, FN, TN, file=file)
accur = accuracy_score(true_column, pred_column)
prec = precision_score(true_column, pred_column)
recall = recall_score(true_column, pred_column)
F_metr = f1_score(true_column, pred_column)
with open("3-6.txt", "w") as file:
    print(accur, prec, recall, F_metr, file=file)
data2 = pd.read_csv('scores.csv')
y = data2['true']
length = len(y)
for i in range(1,5):
    answer = roc_auc_score(y, data2.iloc[:,i])
    print(i, answer)
# with open("3-7.txt", "w") as file:
#     print('score_logreg', file=file)
print('***********')
maxim = -1
for i in range(1,5):
    answer = precision_recall_curve(y,data2.iloc[:,i])
    for j in range(len(answer[0])):
        if answer[1][j] >= 0.7:
            if answer[0][j]>maxim:
                maxim = answer[0][j]
                print('i=',i)
    print('####################')
print(maxim)
with open("3-8.txt", "w") as file:
    print(maxim, file=file)