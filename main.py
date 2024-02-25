import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# datos!
# exterior 10ºC
exterior = 10
# electrodo metalico 35ºC
electrodo = 35
# coeficientes
kelectrodo = 4
kinterior = 0.15


# kexterior=0.2

# malla

def create_matrix_initial(N, M):
    matrizNueva = np.ones((N + 2, M + 2), dtype=float) * 10

    inicio = int(N / 4)

    fin = N - inicio

    punta = 1
    # parte rectangular
    for i in range(inicio, fin):

        for j in range(0, N + 1):
            matrizNueva[i, j] = 35

    # parte triangular

    medio = int((inicio + fin) / 2)
    for i in range(inicio, medio):
        for j in range(N, N + 1 + punta):
            matrizNueva[i, j] = 35
        punta += 4
    punta -= 1
    for i in range(medio, fin):
        for j in range(N + 1, N + 1 + punta):
            matrizNueva[i, j] = 35
        punta -= 4
    return matrizNueva


N = 50
M = N * 3
matrix = create_matrix_initial(N, M)

# 1. perfil estacionario
# derivadaT/derivadat=gradiente^2*k*T(x,y)
# como es estacionario 0=gradiente^2*k*T(x,y)
# hay que despejar T(x,y)
# gradiente = [a/ax,a/ay]
# por diferencias finitas transofrmamos la suma de derivadas parciales, quedandonos con la siguiente ecuacion:
# T(x,y) = (T(x+h,y)+T(x,y+h)+T(x-h,y)+T(x,y-h))/4 es decir la temperatura en un punto es el promedio de temperaturas

plt.imshow(matrix, cmap='hot')
plt.colorbar()
plt.show()

x, y, t = sp.symbols('x,y,t')


# Td = []

# for it1 in range(N):
#     for it2 in range(M):
#         if (it1 == 0) | (it1 == N-1):
#             Td.append(matrix[it1][it2])
#         else:
#             if (it2 == 0) | (it2 == M-1):
#                 Td.append(matrix[it1][it2])
#             else:
#                 if matrix[it1][it2] == 35:
#                     Td.append(matrix[it1][it2])
#                 else:
#                     Td.append(sp.symbols('x%d' % (it1*M+it2)))

# print(Td)

Stencil = [[0 for col in range(N*M)] for row in range(N*M)]
Indep = [0] * (N*M)


for it1 in range(N*M):
    for it2 in range(N*M):
        if it1 == it2:
            if matrix[int(it1/M)+1][(it1 % M)+1] != 35:
                Stencil[it1][it2] = 4
                if (it2 % M) != 0:
                    if matrix[int((it1 - 1) / M) + 1][((it1 - 1) % M) + 1] != 35:
                        Stencil[it1][it2-1] = -1
                if it2 % M != M-1:
                    if matrix[int((it1 + 1) / M) + 1][((it1 + 1) % M) + 1] != 35:
                        Stencil[it1][it2+1] = -1
                if it2 > (M-1):
                    if matrix[int((it1-M) / M) + 1][((it1 - M) % M) + 1] != 35:
                        Stencil[it1][it2-M] = -1
                if it2 < (N*M)-M:
                    if matrix[int((it1 + M) / M) + 1][((it1 + M) % M) + 1] != 35:
                        Stencil[it1][it2+M] = -1



for i in range(1, N+1):
    for j in range(1, M+1):
        if i == 1:
            Indep[(i-1)*M + (j-1)] += 10
        if i == N:
            Indep[(i-1)*M + (j-1)] += 10
        if j == 1:
            Indep[(i-1)*M + (j-1)] += 10
        if j == M:
            Indep[(i-1)*M + (j-1)] += 10
        if matrix[i][j-1] == 35:
            Indep[(i-1)*M+(j-1)] += 35
        if matrix[i+1][j] == 35:
            Indep[(i-1)*M + (j-1)] += 35
        if matrix[i-1][j] == 35:
            Indep[(i-1)*M + (j-1)] += 35
        if matrix[i][j + 1] == 35:
            Indep[(i - 1) * M + (j - 1)] += 35


it1 = 0
it2 = 0

while it1 < (len(Indep)):

    cero = True
    largoColumna = len(Stencil[0])
    it2 = 0

    while it2 < largoColumna:
        if (Stencil[it1][it2]) != 0:
            cero = False
        it2 += 1
    if cero:
        Stencil.pop(it1)
        Indep.pop(it1)
        for i in range(len(Stencil)):
            Stencil[i].pop(it1)
        it1 -= 1
    it1 += 1



# for it1 in range(N):
#     for it2 in range(M):
#         if matrix[it1][it2] == 35:
#             for it3 in range(N*M):
#                 Stencil[it1*M+it2][it3] = 35



# vectorResultado = np.dot(Stencil, Td)

# largoColumna = N*M
# it1 = 0
# it2 = 0
#
# while it2 < largoColumna:
#     cero = True
#     it1 = 0
#     while it1 < (len(Indep)):
#         if (Stencil[it1][it2]) != 0:
#             cero = False
#         it1 += 1
#     if cero:
#         for i in range(len(Indep)):
#             Stencil[i].pop(it2)
#         it2 -= 1
#         largoColumna -= 1
#     it2 += 1

# for i in range(len(Stencil)):
#     print(Stencil[i])

resultado = np.linalg.solve(Stencil, Indep)


contador = 0

for i in range(1, N+1):
    for j in range(1, M+1):
            if matrix[i][j] == 10:
                matrix[i][j] = resultado[contador]
                contador += 1


plt.imshow(matrix, cmap='hot')
plt.colorbar()
plt.show()
