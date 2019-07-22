import time

#profiler - для тайминга работы алгоритма
class Profiler(object):
    def __enter__(self):
        self._startTime = time.time()

    def __exit__(self, type, value, traceback):
        print(
        "Elapsed time: {:.3f} sec".format(time.time() - self._startTime))

# #Задание 3_1 - поиск наиболее весомых признаков
# data = pd.read_csv('svm-data.csv', names=['y', 'x1', 'x2'], header=None)
# y = data['y']
# X = data.iloc[:, 1:]
# clf = SVC(kernel='linear', random_state=241, C=100000)
# clf.fit(X,y)
# result = clf.support_
# with open("3-1.txt", "w") as file:
#     for item in result:
#         print(item+1, end=' ', file=file)




# Задание 3_2 - поиск наиболее весомых слов, определеющих класс текста
# newsgroups=datasets.fetch_20newsgroups(
#                     subset='all',
#                     categories=['alt.atheism', 'sci.space'])
# target = newsgroups.target
# vect = TfidfVectorizer(use_idf=True)
# idf=vect.fit_transform(newsgroups.data)
# clf = SVC(C=10000,kernel='linear', random_state=241)
# clf.fit(idf, target)
# result = []
# top10idx = np.array(clf.coef_.indices)[np.abs(np.array(clf.coef_.data)).argsort()[-10:]]
# for i in top10idx:
#     result.append(vect.get_feature_names()[i])
# result = sorted(result)
# with open("3-2.txt", "w") as file:
#     for item in result:
#         print(item, end=' ', file=file)

#####Часть задания 3-2 для подбора оптимального параметра регуляризации С
# with Profiler() as p:
#
#     newsgroups = datasets.fetch_20newsgroups(
#                         subset='all',
#                         categories=['alt.atheism', 'sci.space']
#                  )
#     grid = {'C': np.power(10.0, np.arange(-5, 6))}
#     cv = KFold(n_splits=5, shuffle=True, random_state=241)
#     clf = SVC(kernel='linear', random_state=241)
#     gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
#     vect = TfidfVectorizer(use_idf=True)
#     idf=vect.fit_transform(newsgroups.data)
#     gs.fit(idf, newsgroups.target)
#     for a in gs.grid_scores_:
#         print('validation=', a.mean_validation_score)
#         print('parameters', a.parameters)