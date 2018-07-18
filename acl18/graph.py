#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from .utils import normalize


def parse_dict(features):
    if features is None or features == "_":
        return {}

    ret = {}
    lst = features.split("|")
    for l in lst:
        k, v = l.split("=")
        ret[k] = v
    return ret


def parse_features(features):
    if features is None or features == "_":
        return set()

    return features.lower().split("|")


class Word:

    def __init__(self, word, upos, lemma=None, xpos=None, feats=None, misc=None):
        self.word = word
        self.norm = normalize(word)
        self.lemma = lemma if lemma else "_"
        self.upos = upos
        self.xpos = xpos if xpos else "_"
        self.xupos = self.upos + "|" + self.xpos
        self.feats = feats if feats else "_"
        self.feats_set = parse_features(self.feats)
        self.misc = misc if misc else "_"

    def cleaned(self):
        return Word(self.word, "_")

    def clone(self):
        return Word(self.word, self.upos, self.lemma, self.xpos, self.feats, self.misc)

    def __repr__(self):
        return "{}_{}".format(self.word, self.upos)


class DependencyGraph(object):

    def __init__(self, words, tokens=None):
        #  Token is a tuple (start, end, form)
        if tokens is None:
            tokens = []
        self.nodes = np.array([Word("*ROOT*", "*ROOT*")] + list(words))
        self.tokens = tokens
        self.heads = np.array([-1] * len(self.nodes))
        self.rels = np.array(["_"] * len(self.nodes), dtype=object)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.nodes = self.nodes
        result.tokens = self.tokens
        result.heads = self.heads.copy()
        result.rels = self.rels.copy()
        return result

    def cleaned(self, node_level=True):
        if node_level:
            return DependencyGraph([node.cleaned() for node in self.nodes[1:]], self.tokens)
        else:
            return DependencyGraph([node.clone() for node in self.nodes[1:]], self.tokens)

    def attach(self, head, tail, rel):
        self.heads[tail] = head
        self.rels[tail] = rel

    def __repr__(self):
        return "\n".join(["{} ->({})  {} ({})".format(str(self.nodes[i]), self.rels[i], self.heads[i], self.nodes[self.heads[i]]) for i in range(len(self.nodes))])
