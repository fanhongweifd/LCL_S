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
import time
import json
import pickle
import numpy as np
from json import loads, dumps
from collections import defaultdict
from sys import argv, stdout, stderr
from PID_Function import PID_Function
from Boost_Function import Boost_Function
from LCCL_S_Function import LCCL_S_Function




def LCL_S_model(Freq, Us, alpha, LP, LS, Cf, RP, RT, RS, Sample, Period, Lb, Cb, fb, D, Kp, Ki, Kd, Ref, fp, Cd, Cp, CS, Lt, round_num, Output_Interv,
                Tj, Simulate_Time, R_Index, M_Index, N_fresh, NP_RMS,  ReSample, resume=False, resume_path='', output_dir='', output_json_path='', flag=0):
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
    :param Output_Interv:   输出数据的时间间隔
    :param round_num:       保留小数点位数
    :param flag:            副边电流正负标志位
    :param Tj:              二极管温度
    :param ReSample:        保存文件的采样间隔
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
    # Output_Interv = 0.01       # 输出数据的时间间隔
    run_time = 0               # 程序运行时长
    T = 1 / Freq               # 系统周期
    w = 2 * np.pi * Freq       # 角速度
    Inner_Time = Period / Freq # 内循环时长
    M = M_Index[0]          # 互感
    R = R_Index[0]          # 负载
    Loop = int(max(1, np.floor(Simulate_Time / Inner_Time)))
    N_boost = Freq / fb
    N_PID = Freq / fp
    TimeGap = 1 / Freq / Sample
    # LT      = LP*alpha
    LT = Lt
    CP      = 1/w/w/LT
    # Cd      = 1/w/w/LP/(1-alpha)
    # Cs      = 1/w/w/LS
    # CP = Cp
    Cs = CS
    NP_RMS = NP_RMS                # 用20周期数据计算一个有效值
    t_all_index = np.arange(0, Simulate_Time, TimeGap)

    # 是否加载上次一循环的参数XBox 和 K

    if resume and os.path.exists(resume_path):
        with open(resume_path, 'rb') as f:
            param = pickle.load(f)
            XBox = param['XBox'][max(param['XBox'])]
            k = param['k']
    else:
        XBox = np.zeros([15, int(Inner_Time / TimeGap)])
        k = 0

    # ------------------------------------------------------------------------
    # 整体循环
    stdout_num = 100
    all_sample  = int(Loop * Inner_Time / TimeGap)
    count_point = np.arange(0, Loop * int(Inner_Time / TimeGap), Loop * int(Inner_Time / TimeGap) / stdout_num).astype('int')
    count_percent = {value: idx/stdout_num for idx, value in enumerate(count_point)}
    result = defaultdict(dict)

    start_time = time.time()
    last_i = 0
    last_j = 0

    # LCL_S 参数
    LCL_Param = [None] * 19
    LCL_Param[0]   = Freq
    LCL_Param[1]   = Us
    LCL_Param[2]   = M
    LCL_Param[3]   = R
    LCL_Param[4]   = LP
    LCL_Param[5]   = LS
    LCL_Param[6]   = Cf
    LCL_Param[7]   = RP
    LCL_Param[8]   = RT
    LCL_Param[9]   = RS
    LCL_Param[10]  = LT
    LCL_Param[11]  = Cp
    LCL_Param[12]  = Cd
    LCL_Param[13]  = Cs
    LCL_Param[14]  = TimeGap
    LCL_Param[15]  = flag
    LCL_Param[16]  = Tj
    LCL_Param[17]  = 0
    LCL_Param[18]  = 0

    Param = [None] * 3
    Param[0]       = 0
    Param[1]       = 0
    Param[2]       = 0

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

    Phi_In      = np.zeros((14,7))
    Matrix_In   = np.zeros((22,7))
    Matrix_In[14,0] = 1/LT
    Matrix_In[15:22,0:7] = np.eye(7)

    for j in range(Loop):
        # 循环迭代
        # time = Inner_Time * (j - 1):TimeGap: Inner_Time * j
        t = np.arange(Inner_Time*j, Inner_Time*(j+1), TimeGap)
        for i in range(int(Inner_Time / TimeGap)):
            # 索引M/R
            Inb = int(np.mod(j * Inner_Time / TimeGap + i + 1, N_PID * Sample))
            Inc = int(np.mod(j * Inner_Time / TimeGap + i + 1, N_fresh * Sample))
            if Inc == 1:
                M = M_Index[k]
                R = R_Index[k]
                LCL_Param[2] = M
                Boost_Param[2] = R
                k = k + 1
            # 循环调用
            if i == 0:
                if j == 0:
                    XBox[0: 7, i], XBox[7:9, i], Phi_O, Matrix_O, Param = LCCL_S_Function(LCL_Param, XBox[0: 7, i], Phi_In, Matrix_In, t, i)
                    LCL_Param[15]       = Param[0]
                    LCL_Param[17]       = Param[1]
                    LCL_Param[18]       = Param[2]
                    Phi_In              = Phi_O
                    Matrix_In[0:14,:]   = Matrix_O

                    XBox[9: 11, i], Req = Boost_Function(Boost_Param, XBox[9: 11, i], i)
                    err_3 = 0
                    err_2 = 0
                    err_1 = Ref - XBox[10, i]
                    if Inb == 0:
                        Err_in = [err_1, err_2, err_3]
                        [D] = PID_Function(PID_Param, i, Err_in, Inb)
                        PID_Param[4] = D
                else:
                    LCL_Param[3] = Req
                    XBox[0: 7, i], XBox[7:9, i], Phi_O, Matrix_O, Param = LCCL_S_Function(LCL_Param, XBox[0: 7, -1], Phi_In, Matrix_In, t, i)
                    LCL_Param[15]       = Param[0]
                    LCL_Param[17]       = Param[1]
                    LCL_Param[18]       = Param[2]
                    Phi_In              = Phi_O
                    Matrix_In[0:14,:]   = Matrix_O

                    Boost_Param[3] = XBox[6, i]
                    Boost_Param[7] = D
                    XBox[9: 11, i], Req = Boost_Function(Boost_Param, XBox[9: 11, -1], i)
                    if (Inb == 0) and (j * Inner_Time / TimeGap + i + 1 > 10000):
                        err_3 = err_2
                        err_2 = err_1
                        err_1 = Ref - XBox[10, i]
                        Err_in = [err_1, err_2, err_3]
                        D = PID_Function(PID_Param, i, Err_in, Inb)
                        PID_Param[4] = D
            else:
                LCL_Param[3] = Req
                XBox[0: 7, i], XBox[7:9, i], Phi_O, Matrix_O, Param = LCCL_S_Function(LCL_Param, XBox[0: 7, i - 1], Phi_In, Matrix_In, t, i)
                LCL_Param[15] = Param[0]
                LCL_Param[17] = Param[1]
                LCL_Param[18] = Param[2]
                Phi_In = Phi_O
                Matrix_In[0:14, :] = Matrix_O
                Boost_Param[3] = XBox[6, i]
                Boost_Param[7] = D
                XBox[9:11, i], Req = Boost_Function(Boost_Param, XBox[9: 11, i - 1], i)
                if (Inb == 0) and (j * Inner_Time / TimeGap + i + 1 > 10000):
                    err_3 = err_2
                    err_2 = err_1
                    err_1 = Ref - XBox[10, i]
                    Err_in = [err_1, err_2, err_3]
                    D = PID_Function(PID_Param, i, Err_in, Inb)
                    PID_Param[4] = D

            # 输出进度
            count = j * int(Inner_Time / TimeGap) + i
            end_time = time.time()
            # if (count in count_percent) and (count != 0):
            if end_time - start_time > Output_Interv:
                run_time += end_time - start_time
                temp_result = defaultdict(dict)
                if last_j != j:
                    if i != 0:

                        temp_result[j]['IP'] = np.round(XBox[0, :i], round_num).tolist()
                        temp_result[j]['IT'] = np.round(XBox[1, :i], round_num).tolist()
                        temp_result[j]['IS'] = np.round(XBox[2, :i], round_num).tolist()
                        temp_result[j]['UCP'] = np.round(XBox[3, :i], round_num).tolist()
                        temp_result[j]['UCT'] = np.round(XBox[4, :i], round_num).tolist()
                        temp_result[j]['UCS'] = np.round(XBox[5, :i], round_num).tolist()
                        temp_result[j]['Ur'] = np.round(XBox[6, :i], round_num).tolist()
                        temp_result[j]['Uinv'] = np.round(XBox[7, :i], round_num).tolist()
                        temp_result[j]['Urec'] = np.round(XBox[8, :i], round_num).tolist()
                        temp_result[j]['Iout'] = np.round(XBox[9, :i], round_num).tolist()
                        temp_result[j]['Vout'] = np.round(XBox[10, :i], round_num).tolist()
                        temp_result[j]['Vlp'] = np.round(XBox[1, :i] * w * LP, round_num).tolist()
                        temp_result[j]['ICP'] = np.round(XBox[3, :i] * w * CP, round_num).tolist()
                        temp_result[j]['VLT'] = np.round(XBox[0, :i] * w * LT, round_num).tolist()
                        temp_result[j]['VLR'] = np.round(XBox[2, :i] * w * LS, round_num).tolist()

                    temp_result[last_j]['IP'] = np.round(result['XBox'][last_j][0, last_i:], round_num).tolist()
                    temp_result[last_j]['IT'] = np.round(result['XBox'][last_j][1, last_i:], round_num).tolist()
                    temp_result[last_j]['IS'] = np.round(result['XBox'][last_j][2, last_i:], round_num).tolist()
                    temp_result[last_j]['UCP'] = np.round(result['XBox'][last_j][3, last_i:], round_num).tolist()
                    temp_result[last_j]['UCT'] = np.round(result['XBox'][last_j][4, last_i:], round_num).tolist()
                    temp_result[last_j]['UCS'] = np.round(result['XBox'][last_j][5, last_i:], round_num).tolist()
                    temp_result[last_j]['Ur'] = np.round(result['XBox'][last_j][6, last_i:], round_num).tolist()
                    temp_result[last_j]['Uinv'] = np.round(result['XBox'][last_j][7, last_i:], round_num).tolist()
                    temp_result[last_j]['Urec'] = np.round(result['XBox'][last_j][8, last_i:], round_num).tolist()
                    temp_result[last_j]['Iout'] = np.round(result['XBox'][last_j][9, last_i:], round_num).tolist()
                    temp_result[last_j]['Vout'] = np.round(result['XBox'][last_j][10, last_i:], round_num).tolist()
                    temp_result[last_j]['Vlp'] = np.round(result['XBox'][last_j][1, last_i:] * w * LP, round_num).tolist()
                    temp_result[last_j]['ICP'] = np.round(result['XBox'][last_j][3, last_i:] * w * CP, round_num).tolist()
                    temp_result[last_j]['VLT'] = np.round(result['XBox'][last_j][0, last_i:] * w * LT, round_num).tolist()
                    temp_result[last_j]['VLR'] = np.round(result['XBox'][last_j][2, last_i:] * w * LS, round_num).tolist()

                else:
                    temp_result[j]['IP'] = np.round(XBox[0, last_i:i], round_num).tolist()
                    temp_result[j]['IT'] = np.round(XBox[1, last_i:i], round_num).tolist()
                    temp_result[j]['IS'] = np.round(XBox[2, last_i:i], round_num).tolist()
                    temp_result[j]['UCP'] = np.round(XBox[3, last_i:i], round_num).tolist()
                    temp_result[j]['UCT'] = np.round(XBox[4, last_i:i], round_num).tolist()
                    temp_result[j]['UCS'] = np.round(XBox[5, last_i:i], round_num).tolist()
                    temp_result[j]['Ur'] = np.round(XBox[6, last_i:i], round_num).tolist()
                    temp_result[j]['Uinv'] = np.round(XBox[7, last_i:i], round_num).tolist()
                    temp_result[j]['Urec'] = np.round(XBox[8, last_i:i], round_num).tolist()
                    temp_result[j]['Iout'] = np.round(XBox[9, last_i:i], round_num).tolist()
                    temp_result[j]['Vout'] = np.round(XBox[10, last_i:i], round_num).tolist()
                    temp_result[j]['Vlp'] = np.round(XBox[1, last_i:i] * w * LP, round_num).tolist()
                    temp_result[j]['ICP'] = np.round(XBox[3, last_i:i] * w * CP, round_num).tolist()
                    temp_result[j]['VLT'] = np.round(XBox[0, last_i:i] * w * LT, round_num).tolist()
                    temp_result[j]['VLR'] = np.round(XBox[2, last_i:i] * w * LS, round_num).tolist()

                stdout.write(dumps({
                    'type': 'process',
                    'value': count/all_sample,
                    'result': temp_result,
                    'time': time.time(),
                    'run_time': run_time
                }))

                # {
                #     'XBox': {
                #         0: ...
                #         1: ...
                #     }
                #     'k': 10000
                # }
                stdout.flush()
                while input() != 'continue':
                   continue
                last_j = j
                last_i = i
                start_time = time.time()

            elif count == (all_sample -1):
                temp_result = defaultdict(dict)
                temp_result[j]['IP'] = np.round(XBox[0, last_i:], round_num).tolist()
                temp_result[j]['IT'] = np.round(XBox[1, last_i:], round_num).tolist()
                temp_result[j]['IS'] = np.round(XBox[2, last_i:], round_num).tolist()
                temp_result[j]['UCP'] = np.round(XBox[3, last_i:], round_num).tolist()
                temp_result[j]['UCT'] = np.round(XBox[4, last_i:], round_num).tolist()
                temp_result[j]['UCS'] = np.round(XBox[5, last_i:], round_num).tolist()
                temp_result[j]['Ur'] = np.round(XBox[6, last_i:], round_num).tolist()
                temp_result[j]['Uinv'] = np.round(XBox[7, last_i:], round_num).tolist()
                temp_result[j]['Urec'] = np.round(XBox[8, last_i:i], round_num).tolist()
                temp_result[j]['Iout'] = np.round(XBox[9, last_i:], round_num).tolist()
                temp_result[j]['Vout'] = np.round(XBox[10, last_i:], round_num).tolist()
                temp_result[j]['Vlp'] = np.round(XBox[1, last_i:] * w * LP, round_num).tolist()
                temp_result[j]['ICP'] = np.round(XBox[3, last_i:] * w * CP, round_num).tolist()
                temp_result[j]['VLT'] = np.round(XBox[0, last_i:] * w * LT, round_num).tolist()
                temp_result[j]['VLR'] = np.round(XBox[2, last_i:] * w * LS, round_num).tolist()

                stdout.write(dumps({
                    'type': 'process',
                    'value': 1,
                    'result': temp_result,
                    'time': time.time(),
                    'run_time': run_time
                }))
                stdout.flush()

                last_j = j
                last_i = i
                start_time = time.time()

            if (i+1) % (NP_RMS * Sample) == 0:
                IP_RMS = np.sqrt(np.mean(XBox[0, (i - (NP_RMS * Sample) + 1):i:int(Sample/ReSample)] ** 2))
                IS_RMS = np.sqrt(np.mean(XBox[2, (i - (NP_RMS * Sample) + 1):i:int(Sample/ReSample)] ** 2))
                Ur_RMS = np.sqrt(np.mean(XBox[6, (i - (NP_RMS * Sample) + 1):i:int(Sample/ReSample)] ** 2))
                Uinv_RMS = np.sqrt(np.mean(XBox[7, (i - (NP_RMS * Sample) + 1):i:int(Sample/ReSample)] ** 2))
                Pin_RMS = IP_RMS * Uinv_RMS
                Pout_RMS = IS_RMS * Ur_RMS * 0.9
                eff_RMS = Pout_RMS / Pin_RMS
                t_RMS = t_all_index[j*int(Inner_Time / TimeGap)+i]

                stdout.write(dumps({
                    'type': 'process',
                    'Pin_RMS': Pin_RMS,
                    'Pout_RMS': Pout_RMS,
                    'eff_RMS': eff_RMS,
                    't_RMS': t_RMS,
                    'index': j * int(Inner_Time / TimeGap) + i + 1
                }))
                # })+'\n')
                stdout.flush()

            XBox[11, i] = err_1
            XBox[12, i] = err_2
            XBox[13, i] = err_3
            XBox[14, i] = D

        if output_dir:
            for feature in ['IP', 'IT', 'IS', 'UCP', 'UCT', 'UCS', 'Ur', 'Uinv', 'Iout', 'Vout']:
                with open(feature + str(j) + '.txt', 'w') as f:
                    data_row = eval(feature)
                    f.write('\t'.join(data_row.astype('str').tolist()))

        result['XBox'][j] = XBox.copy()
        if  ReSample>0:
            result['XBox'][j] = result['XBox'][j][:, 0::int(Sample/ReSample)]

        # IP = XBox[0, :]
        # IT = XBox[1, :]
        # IS = XBox[2, :]
        # UCP = XBox[3, :]
        # UCT = XBox[4, :]
        # UCS = XBox[5, :]
        # Ur = XBox[6, :]
        # Uinv = XBox[7, :]
        # Iout = XBox[8, :]
        # Vout = XBox[9, :]

        # print('k={}, j={}, sum(Xbox)={}'.format(k, j, sum(sum(XBox))))

    result['k'] = k
    if resume_path:
        with open(resume_path, 'wb') as f:
            pickle.dump(result, f)

    if output_json_path:
        XBox = result['XBox']
        result_json = defaultdict(dict)
        for i in XBox.keys():
            XBox_i = np.round(XBox[i], round_num)
            result_json[i]['IP'] = XBox_i[0, :].tolist()
            result_json[i]['IT'] = XBox_i[1, :].tolist()
            result_json[i]['IS'] = XBox_i[2, :].tolist()
            result_json[i]['UCP'] = XBox_i[3, :].tolist()
            result_json[i]['UCT'] = XBox_i[4, :].tolist()
            result_json[i]['UCS'] = XBox_i[5, :].tolist()
            result_json[i]['Ur'] = XBox_i[6, :].tolist()
            result_json[i]['Uinv'] = XBox_i[7, :].tolist()
            result_json[i]['Urec'] = XBox_i[8, :].tolist()
            result_json[i]['Iout'] = XBox_i[9, :].tolist()
            result_json[i]['Vout'] = XBox_i[10, :].tolist()
            result_json[i]['err_1'] = XBox_i[11, :].tolist()
            result_json[i]['err_2'] = XBox_i[12, :].tolist()
            result_json[i]['err_3'] = XBox_i[13, :].tolist()
            result_json[i]['D'] = XBox_i[14, :].tolist()

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


