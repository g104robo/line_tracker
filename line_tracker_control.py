# -*- coding: utf-8 -*-
from control.matlab import *
import numpy as np

# k1 = 4.0
# k2 = 3.0
# k3 = 2.0
# Vd = 1.0

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
    F = place(A,B,poles2)
    #計算結果の表示
    print("gain:",F)
    print("poles:", np.linalg.eigvals(A-B*F))

if __name__ == "__main__":
    main()
