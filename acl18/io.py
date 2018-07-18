#!/usr/bin/env python
# encoding: utf-8

from .const import ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
from .graph import DependencyGraph, Word


def read_conll(filename):
    def get_word(columns):
        return Word(columns[FORM], columns[UPOS], lemma=columns[LEMMA], xpos=columns[XPOS], feats=columns[FEATS], misc=columns[MISC])

    def get_graph(graphs, words, tokens, edges, comments):
        graph = DependencyGraph(words, tokens)
        for (h, d, r) in edges:
            graph.attach(h, d, r)
        graph.comments = comments
        graphs.append(graph)

    file = open(filename, "r", encoding="utf-8")

    graphs = []
    words = []
    tokens = []
    edges = []
    comments = []

    sentence_start = False
    while True:
        line = file.readline()
        if not line:
            if len(words) > 0:
                get_graph(graphs, words, tokens, edges, comments)
                words, tokens, edges, comments = [], [], [], []
            break
        line = line.rstrip("\r\n")

        # Handle sentence start boundaries
        if not sentence_start:
            # Skip comments
            if line.startswith("#"):
                comments.append(line)
                continue
            # Start a new sentence
            sentence_start = True
        if not line:
            sentence_start = False
            if len(words) > 0:
                get_graph(graphs, words, tokens, edges, comments)
                words, tokens, edges, comments = [], [], [], []
            continue

        # Read next token/word
        columns = line.split("\t")

        # Skip empty nodes
        if "." in columns[ID]:
            continue

        # Handle multi-word tokens to save word(s)
        if "-" in columns[ID]:
            start, end = map(int, columns[ID].split("-"))
            tokens.append((start, end + 1, columns[FORM]))

            for _ in range(start, end + 1):
                word_line = file.readline().rstrip("\r\n")
                word_columns = word_line.split("\t")
                words.append(get_word(word_columns))
                if word_columns[HEAD].isdigit():
                    head = int(word_columns[HEAD])
                else:
                    head = -1
                edges.append((head, int(word_columns[ID]), word_columns[DEPREL].split(":")[0]))
        # Basic tokens/words
        else:
            words.append(get_word(columns))
            if columns[HEAD].isdigit():
                head = int(columns[HEAD])
            else:
                head = -1
            edges.append((head, int(columns[ID]), columns[DEPREL].split(":")[0]))

    file.close()

    return graphs


def write_conll(filename, graphs, append=False):
    if append:
        file = open(filename, "a", encoding="utf-8")
    else:
        file = open(filename, "w", encoding="utf-8")

    for j in range(len(graphs)):
        graph = graphs[j]
        curtoken = 0
        for i in range(1, len(graph.nodes)):
            if curtoken < len(graph.tokens) and i == graph.tokens[curtoken][0]:
                file.write("{}-{}\t{}\t_\t_\t_\t_\t_\t_\t_\t_\n".format(graph.tokens[curtoken][0], graph.tokens[curtoken][1] - 1, graph.tokens[curtoken][2]))
                curtoken += 1

            file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t_\t{}\n".format(
                i, graph.nodes[i].word, graph.nodes[i].lemma, graph.nodes[i].upos, graph.nodes[i].xpos,
                graph.nodes[i].feats, graph.heads[i], graph.rels[i], graph.nodes[i].misc))

        file.write("\n")

    file.close()


def read_text(filename):
    file = open(filename, "r")
    text = file.read()
    file.close()

    documents = text.split("\n\n")
    ret = [" ".join(x.split("\n")).strip() for x in documents]

    return ret
