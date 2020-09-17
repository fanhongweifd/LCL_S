'''
PURPOSE : A Closed-Loop LCC-S WPT System Simulation
KEYWORDS : LCC_S, WPT
REUSE ISSUES
Other : 1. The variables in the display are all in Parameter
        2. For repeating calls of the function, the initial
           variables named XBox and k should be set as the
           value at the end of last loop.
'''
import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from json import loads, dumps
from collections import defaultdict
from sys import argv, stdout, stderr
from PID_Function import PID_Function
from Boost_Function import Boost_Function
from LCCL_S_Function import LCCL_S_Function



def LCL_S_model(Freq, Us, alpha, LP, LS, Cf, RP, RT, RS, Sample, Period, Lb, Cb, fb, D, Kp, Ki, Kd, Ref, fp,
                Simulate_Time, R_Index, M_Index, N_fresh, resume=False, resume_path='', output_dir='', output_json_path=''):
    '''
    :param Freq:            系统频率（Hz）
    :param Us:              直流电压(V)
    :param alpha:           L偏移量
    :param LP:              原边初级线圈自感（H）
    :param LS:              副边线圈自感（H）
    :param Cf:              输出滤波电容（F）
    :param RP:              原边线圈内阻（Ω）
    :param RT:              原边线圈内阻（Ω）
    :param RS:              副边线圈内阻（Ω）
    :param Sample:          单周期采样点数(最好为Freq的倍数！！！,为保障系统稳定，建议精度>=120)
    :param Period:          仿真周期长度(必须为Freq的倍数！！！)
    :param Lb:              boost电感
    :param Cb:              boost电容
    :param fb:              Boost开关频率(需能被谐振频率Freq整除)
    :param D:               初始占空比
    :param Kp:              PID参数
    :param Ki:              PID参数
    :param Kd:              PID参数
    :param Ref:             参考电压
    :param fp:              PID频率
    :param Simulate_Time:   仿真时长
    :param R_Index:         负载数据
    :param M_Index:         互感数据
    :param N_fresh:         互感和负载更新频率

    :return:
    返回的结构是:
    {
        'XBox':{
            0: ...
            1: ...
        }
        'k': 10000
    }
    '''

    # ------------------------------------------------------------------------
    # 其他参数生成
    T = 1 / Freq               # 系统周期
    w = 2 * np.pi * Freq       # 角速度
    Inner_Time = Period / Freq # 内循环时长
    M = M_Index[0]          # 互感
    R = R_Index[0]          # 负载
    Loop = int(max(1, np.floor(Simulate_Time / Inner_Time)))
    N_boost = Freq / fb
    N_PID = Freq / fp
    TimeGap = 1 / Freq / Sample

    # 是否加载上次一循环的参数XBox 和 K

    if resume and os.path.exists(resume_path):
        with open(resume_path, 'rb') as f:
            param = pickle.load(f)
            XBox = param['XBox'][max(param['XBox'])]
            k = param['k']
    else:
        XBox = np.zeros([10, int(Inner_Time / TimeGap)])
        k = 0

    # ------------------------------------------------------------------------
    # 整体循环
    stdout_num = 100
    count_point = np.arange(0, Loop * int(Inner_Time / TimeGap), Loop * int(Inner_Time / TimeGap) / stdout_num).astype('int')
    count_percent = {value: idx/stdout_num for idx, value in enumerate(count_point)}
    result = defaultdict(dict)

    for j in range(Loop):
        # 初始化参数
        t = np.arange(Inner_Time * j, Inner_Time * (j+1), TimeGap)
        if j > 0:
            XBox[:, 0] = XBox[:, int(Inner_Time / TimeGap)-1]

        # LCL_S 参数
        LCL_Param = [None] * 12
        LCL_Param[0] = Freq
        LCL_Param[1] = Us
        LCL_Param[2] = M
        LCL_Param[3] = R
        LCL_Param[4] = alpha
        LCL_Param[5] = LP
        LCL_Param[6] = LS
        LCL_Param[7] = Cf
        LCL_Param[8] = RP
        LCL_Param[9] = RT
        LCL_Param[10] = RS
        LCL_Param[11] = TimeGap

        # Boost 参数
        Boost_Param = [None] * 8
        Boost_Param[0] = Lb
        Boost_Param[1] = Cb
        Boost_Param[2] = R
        Boost_Param[3] = XBox[6, 0]
        Boost_Param[4] = TimeGap
        Boost_Param[5] = N_boost
        Boost_Param[6] = Sample
        Boost_Param[7] = D

        # PID 参数
        PID_Param = [None] * 5
        PID_Param[0] = Kp
        PID_Param[1] = Ki
        PID_Param[2] = Kd
        PID_Param[3] = Sample
        PID_Param[4] = D
        Err_in = np.zeros([1, 3])

        # 循环迭代
        for i in range(int(Inner_Time / TimeGap)):
            # 索引M/R
            Inb = int(np.mod(j * Inner_Time / TimeGap + i + 1, N_PID * Sample))
            Inc = int(np.mod(j * Inner_Time / TimeGap + i + 1, N_fresh * Sample))
            if Inc == 0:
                M = M_Index[k]
                R = R_Index[k]
                if R < 0:
                    R = 2e2
                LCL_Param[2] = M
                Boost_Param[2] = R
                k = k + 1
            # 循环调用
            if (i == 0) and (j == 0):
                XBox[0: 7, i], XBox[7, i] = LCCL_S_Function(LCL_Param, XBox[0: 7, i], t, i)
                XBox[8: , i], Req = Boost_Function(Boost_Param, XBox[8: 10, i], i +1)
                err_3 = 0
                err_2 = 0
                err_1 = Ref - XBox[9, i]
            elif i == 0:
                LCL_Param[3] = Req
                XBox[0: 7, i], XBox[7, i] = LCCL_S_Function(LCL_Param, XBox[0: 7, i], t, i)
                Boost_Param[3] = XBox[6, i]
                Boost_Param[7] = D
                XBox[8: , i], Req = Boost_Function(Boost_Param, XBox[8: 10, i], i)
                if (Inb == 0) and (j * Inner_Time / TimeGap + i > 50000):
                    err_3 = err_2
                    err_2 = err_1
                    err_1 = Ref - XBox[9, i]
                    Err_in = [err_1, err_2, err_3]
                    [D] = PID_Function(PID_Param, (j - 1) * Inner_Time / TimeGap + i, Err_in, Inb)
                    PID_Param[4] = D
            else:
                LCL_Param[3] = Req
                XBox[0: 7, i], XBox[7, i] = LCCL_S_Function(LCL_Param, XBox[0: 7, i - 1], t, i)
                Boost_Param[3] = XBox[6, i]
                Boost_Param[7] = D
                XBox[8:, i], Req = Boost_Function(Boost_Param, XBox[8: 10, i - 1], i)
                if (Inb == 0) and ( j * Inner_Time / TimeGap + i > 50000):
                    err_3 = err_2
                    err_2 = err_1
                    err_1 = Ref - XBox[9, i]
                    Err_in = [err_1, err_2, err_3]
                    D = PID_Function(PID_Param, j * Inner_Time / TimeGap + i, Err_in, Inb)
                    PID_Param[4] = D

            # 输出进度
            count = j * int(Inner_Time / TimeGap) + i
            if count in count_percent:
                stdout.write(dumps({
                    'type': 'process',
                    'value': count_percent[count],
                }))
                stdout.flush()

        IP = XBox[0, :]
        IT = XBox[1, :]
        IS = XBox[2, :]
        UCP = XBox[3, :]
        UCT = XBox[4, :]
        UCS = XBox[5, :]
        Ur = XBox[6, :]
        Uinv = XBox[7, :]
        Iout = XBox[8, :]
        Vout = XBox[9, :]

        if output_dir:
            for feature in ['IP', 'IT', 'IS', 'UCP', 'UCT', 'UCS', 'Ur', 'Uinv', 'Iout', 'Vout']:
                with open(feature +str(j)+'.txt', 'w') as f:
                    data_row = eval(feature)
                    f.write('\t'.join(data_row.astype('str').tolist()))

        result['XBox'][j] = XBox
        # print('k={}, j={}, sum(Xbox)={}'.format(k, j, sum(sum(XBox))))
    result['k'] = k
    if resume_path:
        with open(resume_path, 'wb') as f:
            pickle.dump(result, f)

    if output_json_path:
        XBox = result['XBox']
        result_json = defaultdict(dict)
        for i in XBox.keys():
            XBox_i = XBox[i]
            result_json[i]['IP'] = XBox_i[0, :].tolist()
            result_json[i]['IT'] = XBox_i[1, :].tolist()
            result_json[i]['IS'] = XBox_i[2, :].tolist()
            result_json[i]['UCP'] = XBox_i[3, :].tolist()
            result_json[i]['UCT'] = XBox_i[4, :].tolist()
            result_json[i]['UCS'] = XBox_i[5, :].tolist()
            result_json[i]['Ur'] = XBox_i[6, :].tolist()
            result_json[i]['Uinv'] = XBox_i[7, :].tolist()
            result_json[i]['Iout'] = XBox_i[8, :].tolist()
            result_json[i]['Vout'] = XBox_i[9, :].tolist()

        with open(output_json_path, 'w') as file_obj:
            json.dump(result_json, file_obj, indent=4)

    return result


if __name__ == '__main__':

    R_M = np.array(open('data/R.txt', 'r').readlines()[0].strip().split('\t')).astype(np.float64)
    M_M = np.array(open('data/M.txt', 'r').readlines()[0].strip().split('\t')).astype(np.float64)

    param = {
        'Freq': 60e3,
        'Us': 750,
        'alpha': .365,
        'LP': 52.8103e-6,
        'LS': 68.2297e-6,
        'Cf': 470e-6,
        'RP': 0.03,
        'RT': 0.06,
        'RS': 0.06,
        'Sample': 120,
        'Period': 3e3,
        'Lb': 1e-3,
        'Cb': 1e-4,
        'fb': 6e3,
        'D': 0.00,
        'Kp': 0.000005,
        'Ki': 0.000025,
        'Kd': 0,
        'Ref': 650,
        'fp': 6e2,
        'Simulate_Time': 0.2,
        'R_Index': R_M,
        'M_Index': M_M,
        'N_fresh': 20,
        'resume': True
    }

    xbox = LCL_S_model(**param)


