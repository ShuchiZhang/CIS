# -*- coding: utf-8 -*-
"""
Created on Wed July 1 14:34:04 2020
"""
import torch
import os
from datetime import datetime

PATH = 'data\\'
MAXINT = 10000
SEQ_LENGTH = 30  # x: 'START' + tokens; y: tokens + 'END'
EMB_SIZE = 32
GENERATE_NUM = 100
FILTER_SIZE = list(range(1, SEQ_LENGTH))
NUM_FILTER = ([100] + [200] * 9 + [160] * SEQ_LENGTH)[0:SEQ_LENGTH - 1]
DIS_NUM_EPOCH = 10
DIS_NUM_EPOCH_PRETRAIN = 10
GEN_NUM_EPOCH = 10
GEN_NUM_EPOCH_PRETRAIN = 10
GEN_HIDDEN_DIM = 48
ROLLOUT_ITER = 16
TOTAL_BATCH = 3
URL_AMOUNT = 100000

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NrGPU = 0


def output(data, filename='urlResult.txt'):
    fp = open(PATH + filename, 'w', encoding='utf-8')
    for url in data:
        fp.write(url + '\n')
    fp.close()


# log to record
def openLog(filename='record.txt'):
    if os.path.exists(PATH + filename):
        append_write = 'a'
    else:
        append_write = 'w'
    log = open(PATH + filename, append_write)
    return log


if DEVICE.type == 'cuda':
    NrGPU = torch.cuda.device_count()
    print('number of GPUs available:{}'.format(NrGPU))
    log = openLog('gpu.txt')
    log.write('datetime:{}, device name:{}\n'.format(
        datetime.now(), torch.cuda.get_device_name(0)))
    log.write('Memory Usage:')
    log.write('\nAllocated:' +
              str(round(torch.cuda.memory_allocated(0) / 1024**3, 1)) + 'GB')
    log.write('\nCached:   ' +
              str(round(torch.cuda.memory_cached(0) / 1024**3, 1)) + 'GB')
    log.close()
