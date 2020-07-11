# -*- coding: utf-8 -*-
"""
Created on Wed July 1 14:34:04 2020
"""
import torch
import string
import random as rd
import pandas as pd
from wordseg import delSpace
from data_processing import gen_label
from config import SEQ_LENGTH, GENERATE_NUM, DEVICE, PATH, URL_AMOUNT, output


# ignore specific characters
def ifIgnored(x, ignored=string.punctuation):
    if x not in ignored:
        return False
    else:
        return True


def processing(characters, num=None):
    # cut out a part of URL
    begin = rd.randint(7, 15)
    end = rd.randint(16, 30)
    length = len(characters)
    begin = min(begin, length)
    end = min(end, length)
    chars = characters[begin:end]
    if num is None:
        num = len(chars)
    for i in range(len(chars)):
        char = chars[i]
        if ifIgnored(char):
            continue
        elif char.isdigit():
            char = str(int(char) + rd.randint(1, 10))
        elif char.isalpha():
            randNum = ord(char) + rd.randint(-10, 10)
            if randNum > 126 or randNum == 125 or randNum == 123 or randNum == 96:
                continue
            else:
              char = chr(randNum)
        chars[i] = char
    characters[begin:end] = chars
    return characters


def read_url_data(file='FinalDataIX.txt', num=None):
    lineList_all = list()
    with open(PATH + file, 'r', encoding='utf-8-sig') as f:
        for line in f:
            # delete blank space in the beginning or end of every URL
            line.strip()
            lineList_all.append(line)
    if num is not None:
        num = min(num, len(lineList_all))
        data = lineList_all[0:num]
    else:
        data = lineList_all
    # delete blank space in every URL
    data = [delSpace(x) for x in data if len(x) > 5]
    for i in range(len(data)):
        characters = list(data[i])
        characters = processing(characters)
        data[i] = "".join(characters)
    output(filename='urlResult.txt', data=data)


if __name__ == "__main__":
    read_url_data(num=URL_AMOUNT)
