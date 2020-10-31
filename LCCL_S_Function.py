'''
PURPOSE : A Model of LCC-S WPT System
KEYWORDS : LCC_S, WPT
'''
import numpy as np
from scipy.linalg import expm

def LCCL_S_Function(LCL_Param,LCL_Ini,t,Index):

    # ------------------------------------------------------------------------
    # 参数解析
    Freq    = LCL_Param[0]
    Us      = LCL_Param[1]
    M       = LCL_Param[2]
    R       = LCL_Param[3]
    alpha   = LCL_Param[4]
    LP      = LCL_Param[5]
    LS      = LCL_Param[6]
    Cf      = LCL_Param[7]
    RP      = LCL_Param[8]
    RT      = LCL_Param[9]
    RS      = LCL_Param[10]
    TimeGap = LCL_Param[11]

    # ------------------------------------------------------------------------
    # 初值计算
    w       = 2 * np.pi * Freq
    LT      = LP*alpha

    Cd = LCL_Param[12]
    Cp = LCL_Param[13]
    Cs = LCL_Param[14]
    # Cp      = 1/w/w/LT
    # Cd      = 1/w/w/LP/(1-alpha)
    # Cs      = 1/w/w/LS
    X0      = LCL_Ini

    # ------------------------------------------------------------------------
    # 矩阵构建
    delta=M**2-LP*LS
    A1 = np.array([ [-RP/LT,     0,              0,              1/LT,           0,          0,          0],
                    [0,          RT*LS/delta,    -RS*M/delta,    LS/delta,       -LS/delta,  -M/delta,   -M/delta],
                    [0,          -RT*M/delta,    RS*LP/delta,    -M/delta,       M/delta,    LP/delta,   LP/delta],
                    [-1/Cp,      1/Cp,           0,              0,              0,          0,          0],
                    [0,          -1/Cd,          0,              0,              0,          0,          0],
                    [0,          0,              1/Cs,           0,              0,          0,          0],
                    [0,          0,              1/Cf,           0,              0,          0,          -1/(Cf*R)]])

    A2 = np.array([ [-RP/LT,     0,              0,              1/LT,           0,          0,          0],
                    [0,          RT*LS/delta,    -RS*M/delta,    LS/delta,       -LS/delta,  -M/delta,   M/delta],
                    [0,          -RT*M/delta,    RS*LP/delta,    -M/delta,       M/delta,    LP/delta,   -LP/delta],
                    [-1/Cp,      1/Cp,           0,              0,              0,          0,          0],
                    [0,          -1/Cd,          0,              0,              0,          0,          0],
                    [0,          0,              1/Cs,           0,              0,          0,          0],
                    [0,          0,              -1/Cf,          0,              0,          0,          -1/(Cf*R)]])

    B  = np.array([1/LT,       0,              0,              0,              0,          0,          0]).reshape(7, 1)

    # ------------------------------------------------------------------------
    # 循环迭代
    #  常数预计算
    Phi1 = expm(A1*TimeGap)
    Phi2 = expm(A2*TimeGap)
    Inv = np.mod(np.fix(t[Index]/(1/Freq/2)), 2)
    if Inv == 1:
        X0 = np.matmul(Phi2, X0) + np.linalg.multi_dot((np.linalg.inv(A2), (Phi1 - np.eye(7)), B)).reshape(X0.shape) * Us
        Uin = -Us
    if Inv == 0:
        X0 = np.matmul(Phi1, X0) - np.linalg.multi_dot((np.linalg.inv(A1), (Phi1 - np.eye(7)), B)).reshape(X0.shape) * Us
        Uin = Us

    return X0, Uin
