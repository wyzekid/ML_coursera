import math
import numpy as np
import skimage.color
from skimage.io import imread
from sklearn.cluster import KMeans


#в методе нужно глянуть размерности матриц и правильно организовать цикл
def PSNR(original, clustered, n, m):
    summa = 0
    for i in range(n*m):
        for j in range(3):
            summa += (original[i][j]-clustered[i])**2
    summa /= 3*n*m
    return -10*(math.log10(summa))

image = imread('parrots.jpg')
float_image = skimage.img_as_float(image)
n = float_image.shape[0]
m = float_image.shape[1]
X = float_image.reshape(n*m, 3) #матрица признаков(интенсивность цветов RGB)
X_med = np.zeros((n*m, 3))
X_mean = np.zeros((n*m, 3))
numb_clusters = []
temp_matrix = []
temp_matrix_red = []
temp_matrix_green = []
temp_matrix_blue = []
for count_clust in range(8, 21, 1):
    k_means = KMeans(init='k-means++', random_state=241, n_clusters=count_clust).fit(X)
    y = k_means.labels_ #матрица кластеров
    for i in y:
        if i not in numb_clusters:
            numb_clusters.append(i)
    clusters_mean = dict.fromkeys(numb_clusters)
    clusters_med = dict.fromkeys(numb_clusters)
    clusters_mean_red = dict.fromkeys(numb_clusters)
    clusters_mean_green = dict.fromkeys(numb_clusters)
    clusters_mean_blue = dict.fromkeys(numb_clusters)
    clusters_med_red = dict.fromkeys(numb_clusters)
    clusters_med_green = dict.fromkeys(numb_clusters)
    clusters_med_blue = dict.fromkeys(numb_clusters)
    for j in numb_clusters:
        for i in range(len(y)):
            if y[i] == j:
                temp_matrix_red.append(X[i][0])
                temp_matrix_green.append(X[i][1])
                temp_matrix_blue.append(X[i][2])
        clusters_mean_red[j] = np.mean(temp_matrix_red)
        clusters_mean_green[j] = np.mean(temp_matrix_green)
        clusters_mean_blue[j] = np.mean(temp_matrix_blue)
        clusters_med_red[j] = np.median(temp_matrix_red)
        clusters_med_green[j] = np.median(temp_matrix_green)
        clusters_med_blue[j] = np.median(temp_matrix_blue)
        temp_matrix_red.clear()
        temp_matrix_green.clear()
        temp_matrix_blue.clear()
    for j in numb_clusters:
        for i in range(len(y)):
            if y[i] == j:
                X_med[i][0] = clusters_med_red[j]
                X_med[i][1] = clusters_med_green[j]
                X_med[i][2] = clusters_med_blue[j]
                X_mean[i][0] = clusters_mean_red[j]
                X_mean[i][1] = clusters_mean_green[j]
                X_mean[i][2] = clusters_mean_blue[j]
    A_med = X_med.reshape(n, m, 3)
    B_mean = X_mean.reshape(n, m, 3)
    print(count_clust)
    print('PSNR_med = ', skimage.measure.compare_psnr(float_image, A_med))
    print('PSNR_mean = ', skimage.measure.compare_psnr(float_image, B_mean))
    # print('PSNR_med = ', PSNR(float_image,A_med,n,m))
    # print('PSNR_mean = ', PSNR(float_image,B_mean,n,m))

