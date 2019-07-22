import matplotlib.pyplot as plt

x = [1,2,3,4]
y = [5,6,7,8]
z = [9,20,5,6]
fig = plt.figure()
plt.title('Градиентный бустинг')
plt.xlabel('Количество деревьев (n_estimators)')
plt.ylabel('Время выполнения, с')
plt.grid(True)
plt.plot(x, y)
plt.plot(x, z)
plt.show()