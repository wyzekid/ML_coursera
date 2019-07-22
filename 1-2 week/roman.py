w, h, n = map(int, input().split())
if n==1:
    sqr_x = max(w,h)
    print(sqr_x)
    quit()
S0 = w*h*n #общая площадь дипломов
sqr_x = S0**(1/2)
if (sqr_x*10) % 10 != 0:
    sqr_x = int(sqr_x+1)
a = int(sqr_x/h) # сколько прямоугольнкиов поместятся в высоту
b = int(sqr_x/w) #сколько прямоугольников поместятся в ширину
c = sqr_x - b*w #зазор по ширине
d = sqr_x - a*h #зазор по высоте
if a+b-1 != n and c < w and d<h:
    if (c==0 and d==0):
        #sqr_x = sqr_x
        print(sqr_x)
        quit()
    if c == 0:
        sqr_x = sqr_x+h-d
        print(sqr_x)
        quit()
    if d == 0:
        sqr_x = sqr_x+w-c
        print(sqr_x)
        quit()
    sqr_x = sqr_x+min(c, d)
    print(sqr_x)







'''def lcs(a, b):
    lengths = [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
    return lengths[len(a)][len(b)]

N = int(input())
pos_1 = list(map(int, input().split()))
if len(pos_1) != N:
    print("Wrong length of sequence_1")
    quit()
M = int(input())
pos_2 = list(map(int, input().split()))
if len(pos_2) != M:
    print("Wrong length of sequence_2")
    quit()
result = lcs(pos_1, pos_2)
print(result)'''


'''if len(n) != len_N:
    print("Wrong length of list")
    quit()
trans_matrix = [[0]*2]*q
buffer_list = []
for i in range(q):
    trans_matrix[i] = input().split()
for i in range(q):
    for j in range(int(trans_matrix[i][0]), int(trans_matrix[i][1])+1):
        buffer_list.append(n[j-1])
    print(max(buffer_list))
    buffer_list.clear()'''
