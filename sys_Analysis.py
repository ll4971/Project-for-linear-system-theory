import numpy as np
import matplotlib.pyplot as plt
import control
from control.matlab import *


def stability(A, B, C, D, U, T, X0):
    sys = ss(A, B, C, D)
    yout, t, xout = lsim(sys, T, U, X0)
    # 使用子图
    plt.figure()
    plt.suptitle('Zero Input Response', size=16)  # 整张图的总标题
    plt.subplot(3, 1, 1)  # （行，列，活跃区）
    plt.title('Ball Displacement')  # 子图的小标题
    plt.plot(t.T, yout.T[0])
    plt.subplot(3, 1, 3)
    plt.title('Rail Angle')  # 子图的小标题
    plt.plot(t.T, yout.T[1])
    plt.show(block=False)
    print("系统的极点为：\n", pole(sys))
    try:
        print(zero(sys))
    except NotImplementedError:
        print("系统无零点")


def controllability(A, B):
    print("系统能控矩阵为：\n", control.ctrb(A, B))
    print("系统能控矩阵秩为：", np.linalg.matrix_rank(control.ctrb(A, B)))
    if (np.linalg.matrix_rank(control.ctrb(A, B)) == 4):
        print("系统完全能控")
    elif (np.linalg.matrix_rank(control.ctrb(A, B)) < 4):
        print("系统不完全能控")


def observability(A, C):
    print("系统能观矩阵为：\n", control.obsv(A, C))
    print("系统能观矩阵秩为:", np.linalg.matrix_rank(control.obsv(A, C)))
    if (np.linalg.matrix_rank(control.obsv(A, C)) == 4):
        print("系统完全能观")
    elif (np.linalg.matrix_rank(control.obsv(A, C)) < 4):
        print("系统不完全能观")


if __name__ == '__main__':
    A = np.array(
        [[0, 1, 0, 0], [0, 0, 7, 0], [0, 0, 0, 1], [18.873, 0, 0, 0]])
    B = np.array([[0], [0], [0], [3.495]])
    C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    D = np.array([[0], [0]])
    T = np.linspace(0, 10, 201)
    U = np.zeros(T.shape[0])
    X0 = np.array([[0.5], [0], [0.5], [0]])
    stability(A, B, C, D, T, U, X0)
    controllability(A, B, C, D)
    observability(A, B, C, D)
