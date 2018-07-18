#!/usr/bin/env python
# encoding: utf-8

import argparse
import sys

args = argparse.ArgumentParser()

args.add_argument('input', type=str)
args.add_argument('output', type=str)
args.add_argument('model', type=str)
args.add_argument('mode', type=str)
args.add_argument('lan', type=str)

args = args.parse_args()

from acl18.cdparser import CDParser
from acl18.io import read_conll, write_conll

graphs = read_conll(args.input)

parser = CDParser()
parser.load_model(args.model, verbose=False)

m = args.mode
dic = {"cle": "npmst", "ah": "ahdp", "mh4": "mh4t"}

if m in dic:
    m = dic[m]

mode = {}
mode[m] = True
mode["label"] = True
parser.predict(graphs, **mode)

write_conll(args.output, graphs)
