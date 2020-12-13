'''
PURPOSE : A Model of LCC-S WPT System
KEYWORDS : LCC_S, WPT
'''
import numpy as np
from scipy.linalg import expm

# def LCCL_S_Function(LCL_Param,LCL_Ini,t,Index):

def LCCL_S_Function(LCL_Param, LCL_Ini, Phi_In, Matrix_In, t, Index):

    # ------------------------------------------------------------------------
    # 参数解析
    # Freq    = LCL_Param[0]
    # Us      = LCL_Param[1]
    # M       = LCL_Param[2]
    # R       = LCL_Param[3]
    # alpha   = LCL_Param[4]
    # LP      = LCL_Param[5]
    # LS      = LCL_Param[6]
    # Cf      = LCL_Param[7]
    # RP      = LCL_Param[8]
    # RT      = LCL_Param[9]
    # RS      = LCL_Param[10]
    # TimeGap = LCL_Param[11]

    Freq    = LCL_Param[0]
    Us      = LCL_Param[1]
    M       = LCL_Param[2]
    R       = LCL_Param[3]
    LP      = LCL_Param[4]
    LS      = LCL_Param[5]
    Cf      = LCL_Param[6]
    RP      = LCL_Param[7]
    RT      = LCL_Param[8]
    RS      = LCL_Param[9]
    LT      = LCL_Param[10]
    Cp      = LCL_Param[11]
    Cd      = LCL_Param[12]
    Cs      = LCL_Param[13]
    TimeGap = LCL_Param[14]
    flag    = LCL_Param[15]
    Tj      = LCL_Param[16]
    M_last  = LCL_Param[17]
    R_last  = LCL_Param[18]
    X0      = LCL_Ini

    A1_last = Matrix_In[0:7,0:7]
    A2_last = Matrix_In[7:14,0:7]
    B       = Matrix_In[14,0:7].T   #注意是转置矩阵
    eye     = Matrix_In[15:22,0:7]

    # # ------------------------------------------------------------------------
    # # 初值计算
    # w       = 2 * np.pi * Freq
    # LT      = LP*alpha
    #
    # Cd = LCL_Param[12]
    # Cp = LCL_Param[13]
    # Cs = LCL_Param[14]
    # # Cp      = 1/w/w/LT
    # # Cd      = 1/w/w/LP/(1-alpha)
    # # Cs      = 1/w/w/LS
    # X0      = LCL_Ini

    # ------------------------------------------------------------------------
    # 矩阵构建
    if (M_last != M) or (R_last != R):
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
        Phi1 = expm(A1 * TimeGap)
        Phi2 = expm(A2 * TimeGap)
        # B  = np.array([1/LT,       0,              0,              0,              0,          0,          0]).reshape(7, 1)
    else:
        A1   = A1_last
        A2   = A2_last
        Phi1 = Phi_In[0:7,0:7]
        Phi2 = Phi_In[7:14,0:7]
    # ------------------------------------------------------------------------
    # 循环迭代
    #  常数预计算
    # Phi1 = expm(A1*TimeGap)
    # Phi2 = expm(A2*TimeGap)
    Inv = np.mod(np.fix(t[Index]/(1/Freq/2)), 2)
    # if Inv == 1:
    #     X0 = np.matmul(Phi2, X0) + np.linalg.multi_dot((np.linalg.inv(A2), (Phi1 - np.eye(7)), B)).reshape(X0.shape) * Us
    #     Uin = -Us
    # if Inv == 0:
    #     X0 = np.matmul(Phi1, X0) - np.linalg.multi_dot((np.linalg.inv(A1), (Phi1 - np.eye(7)), B)).reshape(X0.shape) * Us
    #     Uin = Us

    U_Square = np.array([0.0, 0.0])

    if Inv == 1 and flag == 1:
        # X0 = Phi1 * X0 + (A1 ^ (-1)) * (Phi1 - eye) * B * Us;
        X0 = np.matmul(Phi1, X0) + np.linalg.multi_dot((np.linalg.inv(A1), (Phi1 - eye), B)).reshape(
             X0.shape) * Us
        U_Square[0] = Us
        U_Square[1] = -X0[6]

    if Inv == 1 and flag == 0:
        # X0 = Phi2 * X0 + (A2 ^ (-1)) * (Phi1 - eye) * B * Us;
        X0 = np.matmul(Phi2, X0) + np.linalg.multi_dot((np.linalg.inv(A2), (Phi1 - eye), B)).reshape(
             X0.shape) * Us
        U_Square[0] = Us
        U_Square[1] = -X0[6]

    if Inv == 0 and flag == 1:
        # X0 = Phi1 * X0 - (A1 ^ (-1)) * (Phi1 - eye) * B * Us
        X0 = np.matmul(Phi1, X0) - np.linalg.multi_dot((np.linalg.inv(A1), (Phi1 - eye), B)).reshape(
             X0.shape) * Us
        U_Square[0] = -Us
        U_Square[1] = X0[6]

    if Inv == 0 and flag == 0:
        # X0 = Phi2 * X0 - (A2 ^ (-1)) * (Phi1 - eye) * B * Us
        X0 = np.matmul(Phi2, X0) - np.linalg.multi_dot((np.linalg.inv(A2), (Phi1 - eye), B)).reshape(
             X0.shape) * Us
        U_Square[0] = -Us
        U_Square[1] = -X0[6]


    if X0[2] * (0.23 + Tj * 2.71e-4) + (0.97 + Tj * (-1.4e-3)) > 1.5:
        flag = 1
    elif X0[2] <= -200e-6:
        flag = 0

    Param_O = np.array([0.0, 0.0, 0.0])
    Param_O[0]         = flag
    Param_O[1]         = M
    Param_O[2]         = R
    Phi_O = np.zeros((14, 7))
    Phi_O[0:7, 0:7]    = Phi1
    Phi_O[7:14,0:7]    = Phi2
    Matrix_O = np.zeros((14, 7))
    Matrix_O[0:7, 0:7] = A1
    Matrix_O[7:14,0:7] = A2

    return X0, U_Square, Phi_O, Matrix_O, Param_O