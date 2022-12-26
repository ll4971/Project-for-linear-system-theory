import numpy as np
import sys_Analysis

# 系统矩阵
A = np.array(
    [[0, 1, 0, 0], [0, 0, 7, 0], [0, 0, 0, 1], [18.873, 0, 0, 0]])
B = np.array([[0], [0], [0], [3.495]])
C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
D = np.array([[0], [0]])
X0 = np.array([[0.5], [0], [0.5], [0]])

T = np.linspace(0, 10, 201)  # 仿真步长
U = np.zeros(T.shape[0])  # 输入矩阵
sys_Analysis.stability(A, B, C, D, T, U, X0)  # 零输入响应

sys_Analysis.controllability(A, B)  # 能控性分析

sys_Analysis.observability(A, C)  # 能观性分析

h, i = np.linalg.eig(A)
print("系统特征根为:\n", format(h))  # 特征值
# print(control.place(A, B, P))
# P1 = np.array([-50 + 3j, -50 - 3j, -80 - 6j, -80 + 6j])
# print(control.place(A, B, P))  # get k
# L = control.place(A.transpose(), C.transpose(), P1)
# print("观测矩阵极点\n", L.transpose())  # 观测矩阵极点
