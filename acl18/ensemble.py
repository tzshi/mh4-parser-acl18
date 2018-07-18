#!/usr/bin/env python
# encoding: utf-8


def most_common(lst):
    s = set(lst)
    return max(s, key=lambda x: lst.count(x))


def POSEnsemble(lst):
    ret = []
    for i in range(len(lst[0])):
        ret.append(most_common([l[i] for l in lst]))

    return ret
