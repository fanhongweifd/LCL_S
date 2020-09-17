'''
PURPOSE : A Model of Boost System
KEYWORDS : Boost, WPT
'''
import numpy as np
from scipy.linalg import expm

def  Boost_Function(Boost_Param,Boost_Ini,Index):

    # ------------------------------------------------------------------------
    # 参数解析
    Lb      = Boost_Param[0]
    Cb      = Boost_Param[1]
    R       = Boost_Param[2]
    Uin     = Boost_Param[3]
    TimeGap = Boost_Param[4]
    N_boost = Boost_Param[5]
    Sample  = Boost_Param[6]
    D       = Boost_Param[7]

    # ------------------------------------------------------------------------
    # 矩阵构建
    A1 = np.array([[0,      -1/Lb    ],
                   [1/Cb,   -1/(Cb*R)]])
    B =  np.array( [1/Lb,   0        ]).reshape(2, 1)

    # ------------------------------------------------------------------------
    # 循环迭代
    Phi1 = expm(A1*TimeGap)
    X0 = Boost_Ini
    Inb = np.mod(Index, N_boost*Sample)
    iL = X0[0]

    if Inb <= np.round(D*Sample*N_boost):
        X0[0] = X0[0]+TimeGap*Uin/Lb
        X0[1] = X0[1]*np.exp(-TimeGap/(R*Cb))
    elif Inb > np.round(D*Sample*N_boost):
        if iL > 0:
            X0 = np.matmul(Phi1, X0) + np.linalg.multi_dot((np.linalg.inv(A1), (Phi1 - np.eye(2)), B)).reshape(X0.shape) * Uin
        if iL <= 0:
            X0[0] = 0
            X0[1] = X0[1]*np.exp(-TimeGap/(R*Cb))
    Req = R / (1-D)**2

    return X0, Req