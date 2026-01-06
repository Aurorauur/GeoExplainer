import numpy as np
import os

import torchsnooper
import datetime
import time

import sys  
np.set_printoptions(threshold=sys.maxsize)#打印内容不限长度

time_now = time.strftime("%Y%m%d-%H%M", time.localtime())

def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def read_xyz(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    num_body = seq_info['frameInfo'][0]['numBody']
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
    return data

def read_xyz_new(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    num_body = seq_info['frameInfo'][0]['numBody']
    data = np.zeros((3, seq_info['numFrame'], num_joint, num_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
    return data

# author 20220520
def plucker(file, max_body=2, num_joint=25):
    data = read_xyz(file, max_body, num_joint)
    # graph from net.utils.graph
    neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                      (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                      (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                      (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                      (22, 23), (23, 8), (24, 25), (25, 12), (21, 21)]
    neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
    s = np.zeros_like(data)                                                                                                                                                                                                                                                                                                                                                                                    
    rs = np.zeros_like(data)
    c, t, v, m = data.shape     
    for i, j in neighbor_link:
        s[:, :, i, :] = data[:, :, j, :] - data[:, :, i, :]
        rs[0, :, i, :] = data[1, :, j, :] * data[2, :, i, :] - data[2, :, j, :] * data[1, :, i, :]
        rs[1, :, i, :] = data[0, :, j, :] * data[2, :, i, :] - data[2, :, j, :] * data[0, :, i, :]
        rs[2, :, i, :] = data[0, :, j, :] * data[1, :, i, :] - data[1, :, j, :] * data[0, :, i, :]    
    ret = np.concatenate((s, rs), axis=0)
    return ret 