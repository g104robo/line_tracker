# -*- coding: utf-8 -*-
from control.matlab import *
import numpy as np
import math as math
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# k1 = 4.0
# k2 = 3.0
# k3 = 2.0
# Vd = 1.0

phi0 = [ math.pi/6, 0.0, 0.0]
Vd = 1.0

# def func(phi, k, Vd):
#     ret = [phi[1], phi[2], k[3]*Vd*phi[0] + k[1]*phi[1] + k[2]*phi[2]]
#     return ret

def func(phi, time, k1, k2, k3, Vd):
    ret = [phi[1], phi[2], -k3*Vd*phi[0] - k1*phi[1] - k2*phi[2]]
    return ret

def check_ctrb(A,B):
    Uc = ctrb(A,B)  #可制御性行列の計算
    Nu = np.linalg.matrix_rank(Uc)  #Ucのランクを計算
    (N,N) = np.matrix(A).shape      #正方行列Aのサイズ(N*N)
    #可制御性の判別
    if Nu == N: return 0
    else: return -1

def main():
    #システム行列の定義
    #A = np.array([[0,1,0],[0,0,1],[k3*Vd, k1, k2]])
    A = np.array([[0,1,0],[0,0,1],[0, 0, 0]])
    B = np.array([[0],[0],[1]])
    #システムが可制御でなければ終了
    if check_ctrb(A,B) == -1 : exit
    #所望の極
    poles1 = [-1,-1+1j,-1-1j]
    poles2 = [-2,-2+1j,-2-1j]
    #ゲインの設計（極配置法）
    F = place(A,B,poles1)
    #計算結果の表示
    print("gain:",F)
    print("poles:", np.linalg.eigvals(A-B*F))
    # print(F[0][0])
    # print(F[0][1])
    # print(F[0][2])
    time = np.linspace(0,50,1000)
    phi_out = odeint(func, phi0, time, args=(F[0][0],F[0][1],F[0][2],Vd))
    print(phi_out)
    plt.plot(time, phi_out)
    plt.show()

if __name__ == "__main__":
    main()
