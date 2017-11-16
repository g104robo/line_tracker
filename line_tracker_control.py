# -*- coding: utf-8 -*-
from control.matlab import *
import numpy as np
import math as math
from scipy.integrate import odeint
import matplotlib.pyplot as plt


phi0 = [math.pi/6, 0.0, 0.0]
#eta0 = [1.0, math.pi/6, 0.0, 0.0]
#eta0 = 1.0
Vd = 1.0

# def func(phi, k, Vd):
#     ret = [phi[1], phi[2], k[3]*Vd*phi[0] + k[1]*phi[1] + k[2]*phi[2]]
#     return ret

def func_phi(phi, time, k1, k2, k3, Vd):
    ret = [phi[1], phi[2], -k3*Vd*phi[0] - k1*phi[1] - k2*phi[2]]
    return ret

# def func_eta(eta, time, k1, k2, k3, Vd):
#     ret = [-Vd*eta[1], eta[2], eta[3], -k3*Vd*eta[1] - k1*eta[2] - k2*eta[3]]
#     return ret

# def func_phi(phi, time, k1, k2, k3, Vd):
#     ret = [phi[1], phi[2], -k3*Vd*phi[0] - k1*phi[1] - k2*phi[2]]
#     return ret

# def func_eta(eta, time, Vd):
#     return -Vd*eta

def check_ctrb(A,B):
    Uc = ctrb(A,B)  #可制御性行列の計算
    Nu = np.linalg.matrix_rank(Uc)  #Ucのランクを計算
    (N,N) = np.matrix(A).shape      #正方行列Aのサイズ(N*N)
    #可制御性の判別
    if Nu == N: return 0
    else: return -1

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def main():
    #システム行列の定義
    #A = np.array([[0,1,0],[0,0,1],[k3*Vd, k1, k2]])
    A = np.array([[0,1,0],[0,0,1],[0, 0, 0]])
    B = np.array([[0],[0],[1]])

    # etaA = np.array([[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,0]])
    # etaB = np.array([[0],[0],[0],[1]])

    #システムが可制御でなければ終了
    if check_ctrb(A,B) == -1 : exit
    #所望の極
    poles1 = [-1,-1+1j,-1-1j]
    poles2 = [-2,-2+1j,-2-1j]
    poles3 = [-2,-2+5j,-2-5j]
    poles4 = [-1,-5+5j,-5-5j]
    poles5 = [-5,-5+5j,-5-5j]
    
    # eta_poles1 = [-1,-1,-1+1j,-1-1j]
    # eta_poles2 = [-1,-2,-2+1j,-2-1j]
    # eta_poles3 = [-1,-2,-2+5j,-2-5j]
    # eta_poles4 = [-1,-1,-5+5j,-5-5j]

    #ゲインの設計（極配置法）
    F = place(A,B,poles1)
    #計算結果の表示
    print("gain:",F)
    print("poles:", np.linalg.eigvals(A-B*F))
    #phi0[2] = -F[0][0]*phi0[0]-F[0][2]*eta0[0]
    #eta0[3] = -F[0][0]*phi0[0]-F[0][2]*eta0[0]
    #print(phi0)
    #print(eta0)

    time = np.linspace(0,50,1000)
    phi_out = odeint(func_phi, phi0, time, args=(F[0][0],F[0][1],F[0][2],Vd))
    #eta_out = odeint(func_eta, eta0, time, args=(F[0][0],F[0][1],F[0][2],Vd))
    
    #eta_out = odeint(func_eta, eta0, time, args=(Vd,))
    #print(phi_out[:, 0])
    # plt.plot(time, phi_out[:, 0], label="phi")
    # plt.plot(time, phi_out[:, 1], label="ang_vel")
    # plt.plot(time, phi_out[:, 2], label="ang_acc")
    # #plt.plot(time, eta_out[:, 0], label="eta")
    # plt.xlabel("[sec]")
    # plt.ylabel("[rad]")
    # plt.title("line_tracker")
    # plt.legend()
    # plt.show()

    fig, phi = plt.subplots()
    fig.subplots_adjust(right=0.75)
    ang_vel = phi.twinx()
    ang_acc = phi.twinx()

    ang_acc.spines["right"].set_position(("axes",1.2))
    make_patch_spines_invisible(ang_acc)
    ang_acc.spines["right"].set_visible(True)

    phi1, = phi.plot(time, phi_out[:, 0],"b-", label="phi")
    phi2, = ang_vel.plot(time, phi_out[:, 1],"r-", label="ang_vel")
    phi3, = ang_acc.plot(time, phi_out[:, 2],"g-", label="ang_acc")

    phi.yaxis.label.set_color(phi1.get_color())
    ang_vel.yaxis.label.set_color(phi2.get_color())
    ang_acc.yaxis.label.set_color(phi3.get_color())

    tkw = dict(size=4, width=1.5)
    phi.tick_params(axis='y', colors=phi1.get_color(), **tkw)
    ang_vel.tick_params(axis='y', colors=phi2.get_color(), **tkw)
    ang_acc.tick_params(axis='y', colors=phi3.get_color(), **tkw)
    phi.tick_params(axis='x', **tkw)

    phi.set_xlabel("time[sec]")
    phi.set_ylabel("phi[rad]")
    ang_vel.set_ylabel("ang_vel[rad/s]")
    ang_acc.set_ylabel("ang_acc[rad/s2]")

    lines = [phi1, phi2, phi3]

    phi.legend(lines, [l.get_label() for l in lines])
    plt.show()





if __name__ == "__main__":
    main()
