'''
PURPOSE : A Model of PID Controller
KEYWORDS : Boost, WPT
'''

def PID_Function(PID_Param,Index,Err_In,Inb):

    # ------------------------------------------------------------------------
    # 参数解析
    Kp      = PID_Param[0]
    Ki      = PID_Param[1]
    Kd      = PID_Param[2]
    Sample  = PID_Param[3]
    D       = PID_Param[4]


    # ------------------------------------------------------------------------
    # 计算
    if (Inb == 0) and (Index >= Sample*1000):
        err1    = Err_In[0]
        err2    = Err_In[1]
        err3    = Err_In[2]
        Delta_u = Kp*(err1-err2)+Ki*err1+Kd*(err1-2*err2+err3)
        D       = D + Delta_u

        if D >= 0.5:
            D = 0.5
        elif D <= 0:
            D = 0

    return D