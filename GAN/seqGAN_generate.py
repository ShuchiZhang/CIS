# -*- coding: utf-8 -*-
"""
Created on Wed July 1 14:34:04 2020
"""
import sys
import torch
from config import PATH, URL_AMOUNT, openLog
from data_processing import decode
from url_processing import read_url_data


def main(batch_size=1):
    model = torch.load(PATH + 'generator.pkl')
    reverse_vocab = torch.load(PATH + 'reverse_vocab.pkl')

    num = model.generate(batch_size=batch_size)
    log = openLog('genTxt_predict.txt')
    result = decode(num, reverse_vocab, log)
    log.close()
    read_url_data(num=URL_AMOUNT)
    return result


if __name__ == '__main__':
    try:
        batch_size = int(sys.argv[1])
    except IndexError:
        batch_size = 1

    result = main(batch_size)
    print(result)
