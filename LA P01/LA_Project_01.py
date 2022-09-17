import numpy as np

def echelon(a, n, m):
    if n > 1 and m > 0:
        flag = False
        for j in range(n):
            if a[j, 0] != 0:
                a[[0, j]] = a[[j, 0]]
                flag = True
                break
        
        if flag == True: # found a pivot in this column
            for j in range(1, n):
                if (a[j, 0] != 0):
                    a[j] -= a[0] * (a[j, 0]/a[0, 0])
            echelon(a[1:, 1:], n-1, m-1)
        else:            # no pivot was found
            echelon(a[:, 1:], n, m-1)

def reduced_echelon(a, n, m, pp):
    for i in range(n-1, -1, -1):
        for j in range(m):
            if (a[i, j] != 0):
                pp[j] = i
                a[i] /= a[i, j]
                for k in range(i):
                    a[k] -= a[k, j] * a[i]
                break

def print_matrix(a):
    for row in a:
        for col in row:
            print(round(col, 6), end=" ")
        print()

def print_x(x):
    for i in range(x.size):
        print(f'X{i+1} = {round(x[i], 6)}') 

# get matrix dimensions & create the augumented matrix
n, m = map(int, input().split())
a = np.array([input().strip().split() for _ in range(n)], float)

# row reduce to obtain echelon form
echelon(a, n, m)

# row reduce to obtain reduced echelon form
pp = np.full(m, -1)
reduced_echelon(a, n, m, pp)
print_matrix(a)

# find x
x = np.zeros(m-1, float)
for j in range(m-1):
    if pp[j] == -1:
        x[j] = 10
for j in range(m-1):
    if pp[j] != -1:
        for k in range(j+1, m):
            if k != m-1:
                x[j] -= a[pp[j], k] * x[k]
            else:
                x[j] += a[pp[j], k]
print_x(x)