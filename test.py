import json
import numpy as np
from json import loads, dumps
from collections import defaultdict
from sys import argv, stdout, stderr
from LCL_S_Boost_PID import LCL_S_model


if __name__ == '__main__':
    # if len(argv) <= 1:
    #     stderr.write(dumps({
    #         'type': 'input error'
    #     }))
    #     exit()
    #
    # param = loads(argv[1])
    # param['R_Index'] = np.array(open(param['R_Index'], 'r').readlines()[0].strip().split('\t')).astype(np.float64)
    # param['M_Index'] = np.array(open(param['M_Index'], 'r').readlines()[0].strip().split('\t')).astype(np.float64)
    #
    # stdout.write(dumps({
    #     'type': 'program start'
    # }))
    # stdout.flush()


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
        'output_json_path': 'result.json',
        'Cd': 1/(2 * np.pi * 60e3)/(2 * np.pi * 60e3)/(52.8103e-6)/(1-.365),
        'Cp': 1/(2 * np.pi * 60e3)/(2 * np.pi * 60e3)/(52.8103e-6*.365),
        'CS': 1/(2 * np.pi * 60e3)/(2 * np.pi * 60e3)/(68.2297e-6),
        'Lt': 52.8103e-6*.365
    }

    xbox_k = LCL_S_model(**param)

    stdout.write(dumps({
        'type': 'program end'
    }))
    stdout.flush()

