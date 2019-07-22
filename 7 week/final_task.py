import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

data_train = pd.read_csv('features.csv', index_col='match_id')
y_train = data_train['radiant_win'] # целевая переменная
X_train = data_train.drop(['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire',
                           'barracks_status_radiant', 'barracks_status_dire'], 1) # удаляем целевые столбцы из признаков в обучающей выборке
X_test = pd.read_csv('features_test.csv', index_col='match_id')
X_test = X_test.fillna(0) # заполняем пустые значения нулями
print(X_train.count()) # количество непустых значений в столбцах признаков
X_train = X_train.fillna(0) #заполняем пустые значения нулями
cv = KFold(n_splits=5, shuffle=True) #инициализируем кросс-валидатор

# Изменяем количество деревьев в классификаторе и получаем оценки классификации (метрика качества - 'roc_auc')
fig = plt.figure()
plt.title('Градиентный бустинг') # plot - для построения графиков зависимостей характеристик работы алгоритма от количества деревьев
plt.xlabel('Количество деревьев (n_estimators)')
plt.ylabel('Характеристика работы')
plt.grid(True)
mass_val = []
mass_seconds = []
mass_est = []
####..................................Градиентный бустинг.......................................###
'''for estimators in range(10, 41, 10):
    mass_est.append(estimators)
    clf = GradientBoostingClassifier(n_estimators=estimators, verbose=True, random_state=241) #инициализируем классификатор
    clf.fit(X_train, y_train) #обучаем классификатор
    start_time = datetime.datetime.now() #замеряем время кросс-валидации
    scores = cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=cv) # сохраняем качество кросс-валидации в массив
    gradient_cv_time = datetime.datetime.now() - start_time
    mass_seconds.append(float(gradient_cv_time.seconds))
    val = round(scores.mean()*100, 2) # среднее качество кросс-валидации в процентах
    mass_val.append(val)
    print('n_estimators=', estimators, ' Среднее качество работы: ', val, '%')
    print('Время кросс-валидации составляет: ', gradient_cv_time)
    print('====================================================')
plt.plot(mass_est, mass_val)
plt.plot(mass_est, mass_seconds)
plt.show()
param_grid = {'n_estimators': [90, 100], 'max_depth': range(3, 6), 'max_features': ["log2"]} #создаем "сетку параметров", чтобы весь диапазон параметров подставить в классификатор
clf_grid = GridSearchCV(GradientBoostingClassifier(random_state=241), param_grid, cv=cv, n_jobs=1, verbose=1, scoring='roc_auc') #подставляем диапазон параметров в классификатор
clf_grid.fit(X_train, y_train) # обучаем классификатор на обучающей выборке с подставлением соответствующих параметров
print("best_params")
print(clf_grid.best_params_)# выводим параметры лучшего результата классификации
print("best_score")
print(clf_grid.best_score_)# выводим наилучший результат классификации
clf = GradientBoostingClassifier(**clf_grid.best_params_) #создаем классификатор с лучшими параметрами
start_time = datetime.datetime.now()
clf.fit(X_train, y_train) #обучаем классификатор с лучшими параметрами
print('Время обучения классификатора с наилучшими параметрами: ', datetime.datetime.now() - start_time)
scores = cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=cv) # сохраняем качество кросс-валидации в массив
val = round(scores.mean()*100, 2) # среднее качество кросс-валидации в процентах
print('Качество при лучших параметрах ', val, '%')

####Определяем наиболее важные признаки в модели, сортируем и выводим их на экран######
featureImportances = pd.DataFrame(data=clf.feature_importances_)
featureImportances.sort_values([0], ascending=False, inplace=True)
listCol = data_train.columns.values.tolist()
print('***Наиболее значимые признаки***')
count = 1
for i in featureImportances.index:
    if featureImportances.loc[i][0] < 0.01:
        break
    print("%s: %s=%s" %(count, listCol[i], round(featureImportances.loc[i][0]*100, 2)))
    count+=1

####Получаем и выводим предсказанное целевое значение для тестовых данных
pred = clf.predict_proba(X_test)[:, 1]
# with open("7-1.txt", "w") as file:
#     for i in pred:
#         print(i, file=file)'''

###Ответы на вопросы по градиентному бустингу:
# 1.	Пропуски среди значений имеют следующие признаки:
# first_blood_time
# first_blood_team
# first_blood_player1
# first_blood_player2
# radiant_bottle_time
# radiant_courier_time
# radiant_flying_courier_time
# radiant_first_ward_time
# dire_bottle_time
# dire_courier_time
# dire_flying_courier_time
# dire_first_ward_time
# Пропуски в признаках, связанных с «первой кровью» (first_blood_time, first_blood_player1), образуются в случае, если событие «первая кровь» не произошло
# в первые 5 минут матча. Пропуски в признаках, связанных с приобретением предметов (radiant_courier_time, radiant_bottle_time), образуются в случае,
# если эти предметы не были приобретены в первые 5 минут матча.
# 2.    Т.к. в результате необходимо предсказать победителя матча, то целевой переменной является radiant_win.
# 3.    Время кросс-валидации для 30 деревьев составляет 1 минуту 17 секунд. При этом среднее качество работы алгоритма составляет 68,96%.
# 4.	На графике представлены зависимости точности работы алгоритма (синий график, %) и времени кросс-валидации (оранжевый график, с) от числа деревьев в классификаторе.
# Из графика видно, что время работы кросс-валидации с увеличением числа деревьев растет гораздо быстрее, чем качество работы.
# Следовательно, можно сделать вывод, что увеличение числа деревьев хоть и приведет к увеличению качества классификации, но, ввиду вычислительных затрат,
# увеличение числа деревьев нецелесообразно. Поэтому нет смысла использовать больше 30 деревьев в градиентном бустинге.
# Максимальное качество работы алгоритма на рассмотренном диапазоне параметров было достигнуто при n_estimators = 100, max_depth = 5, и составляет 70,97%.
# Процесс обучения классификатора можно ускорить, выявив ключевые признаки, влияющие на результат работы алгоритма классификации, и использовать только первые
# n самых информативных признаков. За счет сокращения количества признаков сократится количество вычислений, что приведет к ускорению работы алгоритма кросс-валидации.
# Наиболее значимые признаки данной модели представлены ниже и отсортированы по убыванию значимости:
# d1_gold=4.25
# d2_gold=4.21
# d4_gold=4.09
# r1_gold=4.07
# r3_gold=3.84
# r4_gold=3.75
# r2_gold=3.74
# d5_gold=3.68
# r5_gold=3.35
# d3_gold=3.14
# r1_lh=2.23
# d3_lh=2.2
# d4_lh=2.16
# r5_lh=2.07
# Кроме сокращения числа признаков для ускорения работы алгоритма можно прибегнуть к уменьшению глубины решающих деревьев.
#
#
#

####..............................Логистическая регрессия.................................................

print('*********************************************')
print('............Логистическая регрессия................')
param_grid = {'C': np.logspace(-6, 2, 15)}#параметры сетки тестирования алгоритма - логарифмическая

# функция для кросс-валидации и предсказания результата
def getCrossValidationResult(text, X_train, y_train, X_test, param_grid, writeToFile):
    print(text)
    clf_grid = GridSearchCV(LogisticRegression(random_state=241, n_jobs=-1), param_grid, cv=cv,
                            n_jobs=1, verbose=1, scoring ='roc_auc') # создаем сетку с параметрами
    clf_grid.fit(X_train, y_train) #обучаем классификатор
    print(u"best_params")
    print(clf_grid.best_params_) #лучший параметр
    print(u"best_score")
    print(clf_grid.best_score_) #лучший результат

    lr = LogisticRegression(n_jobs=-1, random_state=241, **clf_grid.best_params_)# создаем логистрическую регрессию с лучшими параметрами
    lr.fit(X_train, y_train) #Обучаем
    scores = cross_val_score(lr, X_train, y_train, scoring='roc_auc', cv=cv)# массив оценок качества алгоритма
    val = round(scores.mean()*100, 2) # среднее значение оценки
    print("Оценка качества GridSearchCV ", val, '%')

    y_pred = pd.DataFrame(data=lr.predict_proba(X_test)) # предсказываем вероятность победы Radiant
    if writeToFile:
        with open("best_prog.csv", "w") as file:
            for j in y_pred[1].get_values():
                print(j, sep=',', file=file)
    y_pred.sort_values([0], inplace=True) #сортируем
    print(u'max вероятность победы =', y_pred.iloc[0, 1], '; min вероятность победы =', y_pred.iloc[y_pred.shape[0]-1, 1])#1 - класс означает, что Radiant победил


getCrossValidationResult('Кросс-валидация для сырой обучающей выборки', X_train, y_train, X_test, param_grid, False)
print('================================')
# удаляем категориальные признаки
X_train_whithout_categ = X_train.drop(['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                          'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'], 1)
X_test_without_categ = X_test.drop(['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                          'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'], 1)
getCrossValidationResult('Кросс-валидация для выборки без категориальных признаков и без масштабирования ', X_train_whithout_categ, y_train, X_test_without_categ, param_grid, False)
print('================================')
print('Масштабирование признаков')
# масштабируем числовые признаки
X_train_whithout_categ_norm = pd.DataFrame(data=StandardScaler().fit_transform(X_train_whithout_categ))# обучающая выборка с
X_test_without_categ_norm = pd.DataFrame(data=StandardScaler().fit_transform(X_test_without_categ))
getCrossValidationResult('Кросс-валидация для выборки без категориальных признаков и с масштабированием ',
                       X_train_whithout_categ_norm, y_train, X_test_without_categ_norm, param_grid, False)
print('================================')
print('Подсчет количества уникальных героев')
data_heroes = pd.concat([X_train['r1_hero'], X_train['r2_hero'], X_train['r3_hero'], X_train['r4_hero'], X_train['r5_hero'],
                         X_train['d1_hero'], X_train['d2_hero'], X_train['d3_hero'], X_train['d4_hero'], X_train['d5_hero']], axis=1)
iid = pd.Series(data_heroes[['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                          'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']].values.flatten()).drop_duplicates()
N = iid.shape[0]
iid = pd.DataFrame(data=list(range(N)), index=iid.tolist())
iid.sort_index(inplace=True)
print('Количество различных идентификаторов героев в данной выборке: ', N)
print('===============================')
print('Dummy-кодирование')
hero_D = np.unique(X_train['d1_hero'])
N = max(hero_D)
X_pick = np.zeros((X_train.shape[0], N))
for i, match_id in enumerate(X_train.index):
    for p in range(5):
        X_pick[i, X_train.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, X_train.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
X_pick = pd.DataFrame(X_pick)
X_pick.fillna(0, method=None, axis=1, inplace=True)
X_train_full = pd.concat([X_train_whithout_categ_norm, X_pick], axis=1)
X_test_full = pd.concat([X_test_without_categ_norm, X_pick], axis=1)
X_test_full = pd.DataFrame(X_test_full)
X_test_full.fillna(0, method=None, axis=1, inplace=True)
getCrossValidationResult('Кросс-валидация с числовыми признаками героев', X_train_full, y_train, X_test_full, param_grid, True)

####Ответы на вопросы по логистической регрессии

# 1.	Наилучшее качество работы алгоритма было достигнуто при параметре регуляризации С, равном 1e-06. Наилучшее качество при этом составляет 51,35%.
# Полученный результат значительно ниже, чем при градиентном бустинге, ввиду того, что в обучающей выборке присутствуют категориальные признаки,
# которые обрабатываются как числовые. Кроме того, значения признаков не масштабированы. Логистическая регрессия работает быстрее бустинга
# (кросс-валидация по 5 блокам с 15 возможными значениями параметра C занимает порядка 1 минуты).
# 2.	При удалении категориальных признаков параметр C, при котором достигается наилучшее качество, не изменился, как не изменилось и наилучшее качество модели.
# Объясняется это тем, что, во-первых, как мы выяснили в предыдущем задании, удаленные признаки не являются наиболее информативными, и во-вторых,
# линейные модели не очень хорошо работают с категориальными признаками.
# При масштабировании числовых признаков оптимальный параметр C стал равен ~0.01, качество модели при этом улучшилось до 71,65%. Связано это с тем, что
# при масштабировании значения признаков выравниваются, исчезают резкие отклонения.
# 3.	В игре 108 различных идентификаторов героев.
# 4.    При добавлении мешка слов качество работы улучшилось до 75,19% при C = 0.13894954943731361. Это объясняется тем, что мы учли все признаки модели, преобразовав категориальные
# признаки в числовые, сделав модель наиболее полной. Таким образом, данный алгоритм логистической регрессии является наилучшим среди рассмотренных (включая
# градиентный бустинг)
# 5. Максимальное значение прогноза на тестовой выборке составляет 0.99765811204, минимальное составляет 0.00475700726709
