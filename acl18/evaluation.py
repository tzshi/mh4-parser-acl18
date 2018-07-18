#!/usr/bin/env python
# encoding: utf-8


def POSCorrect(prediction, gold):
    ret = 0
    for i in range(1, len(prediction)):
        if gold[i] == prediction[i]:
            ret += 1
    return ret
