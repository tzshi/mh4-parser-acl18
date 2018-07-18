#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from collections import Counter
import re
import random


def normalize(word):
    return re.sub(r"\d", "0", word).lower()


def buildVocab(graphs, cutoff=1):
    wordsCount = Counter()
    charsCount = Counter()
    uposCount = Counter()
    xposCount = Counter()
    relCount = Counter()
    featCount = Counter()

    for graph in graphs:
        wordsCount.update([node.norm for node in graph.nodes[1:]])
        for node in graph.nodes[1:]:
            charsCount.update(list(node.word))
            featCount.update(node.feats_set)
        uposCount.update([node.upos for node in graph.nodes[1:]])
        xposCount.update([node.xupos for node in graph.nodes[1:]])
        relCount.update([rel for rel in graph.rels[1:]])

    wordsCount = Counter({w: i for w, i in wordsCount.items() if i >= cutoff})
    print("Vocab containing {} words".format(len(wordsCount)))
    print("Charset containing {} chars".format(len(charsCount)))
    print("UPOS containing {} tags".format(len(uposCount)), uposCount)
    print("XPOS containing {} tags".format(len(xposCount)), xposCount)
    print("Rels containing {} tags".format(len(relCount)), relCount)
    print("Feats containing {} tags".format(len(featCount)))

    ret = {
        "vocab": list(wordsCount.keys()),
        "wordfreq": wordsCount,
        "charset": list(charsCount.keys()),
        "charfreq": charsCount,
        "upos": list(uposCount.keys()),
        "xpos": list(xposCount.keys()),
        "rels": list(relCount.keys()),
        "feats": list(featCount.keys()),
    }

    return ret


def shuffled_stream(data):
    len_data = len(data)
    while True:
        for d in random.sample(data, len_data):
            yield d


def shuffled_balanced_stream(data):
    for ds in zip(*[shuffled_stream(s) for s in data]):
        ds = list(ds)
        random.shuffle(ds)
        for d in ds:
            yield d
