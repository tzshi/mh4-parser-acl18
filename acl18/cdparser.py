#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function

import numpy as np
from collections import defaultdict
import fire, json, random, time, math, os, lzma
from dynet import *
from .utils import buildVocab, normalize, shuffled_balanced_stream
from .modules import UPOSTagger, XPOSTagger, MSTParser, NPMSTParser, MH4Parser, AHDPParser, AEDPParser
from .modules import MH4TParser, OneECParser

from . import pyximportcpp; pyximportcpp.install()
from .calgorithm import projectivize, is_projective
from .mh4 import mh4ize
from .ec import parse_1ec_o3
from .asbeamconf import ASBeamConf
from .evaluation import POSCorrect

from .io import read_conll, write_conll
from .layers import MultiLayerPerceptron, Dense, identity, Biaffine


PROFILE=False

class ComputationCarrier(object):

    def __copy__(self):
        result = object.__new__(ComputationCarrier)
        result.__dict__.update(self.__dict__)
        return result


class CDParser:

    def __init__(self):
        pass

    def create_parser(self, **kwargs):
        self._verbose = kwargs.get("verbose", True)
        if self._verbose:
            print("Parameters (others default):")
            for k in sorted(kwargs):
                print(k, kwargs[k])
        self._args = kwargs

        self._learning_rate = kwargs.get("learning_rate", 0.001)
        self._beta1 = kwargs.get("beta1", 0.9)
        self._beta2 = kwargs.get("beta2", 0.999)
        self._epsilon = kwargs.get("epsilon", 1e-8)
        self._edecay = kwargs.get("edecay", 0.)
        self._clip = kwargs.get("clip", 5.)
        self._sparse_updates = kwargs.get("sparse_updates", False)

        self._optimizer = kwargs.get("optimizer", "adam")

        self._batch_size = kwargs.get("batch_size", 50)
        self._anneal_base = kwargs.get("anneal_base", 1.0)
        self._anneal_steps = kwargs.get("anneal_steps", 1000)

        self._word_smooth = kwargs.get("word_smooth", 0.25)
        self._char_smooth = kwargs.get("char_smooth", 0.25)

        self._wdims = kwargs.get("wdims", 128)
        self._bidirectional = kwargs.get("bidirectional", True)
        self._bilstm_dims = kwargs.get("bilstm_dims", 128)
        self._bilstm_layers = kwargs.get("bilstm_layers", 2)
        self._bilstm_dropout = kwargs.get("bilstm_dropout", 0.0)

        self._pdims = kwargs.get("pdims", 0)
        self._fdims = kwargs.get("fdims", 0)

        self._feature_dropout = kwargs.get("feature_dropout", 0.0)

        self._block_dropout = kwargs.get("block_dropout", 0.)
        self._char_dropout = kwargs.get("char_dropout", 0.)

        self._cdims = kwargs.get("cdims", 32)
        self._char_lstm_dims = kwargs.get("char_lstm_dims", 128)
        self._char_lstm_layers = kwargs.get("char_lstm_layers", 2)
        self._char_lstm_dropout = kwargs.get("char_lstm_dropout", 0.0)

        self._char_repr_method = kwargs.get("char_repr_method", "pred")

        self._activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}

        self._utagger_num = kwargs.get("utagger_num", 0)
        self._utagger_weight = kwargs.get("utagger_weight", 1.0)
        self._utaggers = [UPOSTagger(self, id="UPOS-{}".format(i+1), **self._args) for i in range(self._utagger_num)]

        self._xtagger_num = kwargs.get("xtagger_num", 0)
        self._xtagger_weight = kwargs.get("xtagger_weight", 1.0)
        self._xtaggers = [XPOSTagger(self, id="XPOS-{}".format(i+1), **self._args) for i in range(self._xtagger_num)]

        self._parsers = []

        self._ahdp_num = kwargs.get("ahdp_num", 0)
        self._ahdp_weight = kwargs.get("ahdp_weight", 1.0)
        self._ahdp_parsers = [AHDPParser(self, id="AHDP-{}".format(i+1), **self._args) for i in range(self._ahdp_num)]
        self._parsers.extend(self._ahdp_parsers)

        self._aedp_num = kwargs.get("aedp_num", 0)
        self._aedp_weight = kwargs.get("aedp_weight", 1.0)
        self._aedp_parsers = [AEDPParser(self, id="AEDP-{}".format(i+1), **self._args) for i in range(self._aedp_num)]
        self._parsers.extend(self._aedp_parsers)

        self._as_mlp_activation = self._activations[kwargs.get('as_mlp_activation', 'relu')]
        self._as_mlp_dims = kwargs.get("as_mlp_dims", 128)
        self._as_mlp_layers = kwargs.get("as_mlp_layers", 2)
        self._as_mlp_dropout = kwargs.get("as_mlp_dropout", 0.0)
        self._as_stack_features = kwargs.get("as_stack_features", 2)
        self._as_buffer_features = kwargs.get("as_buffer_features", 1)
        self._as_weight = kwargs.get("as_weight", 1.0)

        self._mst_num = kwargs.get("mst_num", 0)
        self._mst_weight = kwargs.get("mst_weight", 1.0)
        self._mst_parsers = [MSTParser(self, id="MST-{}".format(i+1), **self._args) for i in range(self._mst_num)]
        self._parsers.extend(self._mst_parsers)

        self._npmst_num = kwargs.get("npmst_num", 0)
        self._npmst_weight = kwargs.get("npmst_weight", 1.0)
        self._npmst_parsers = [NPMSTParser(self, id="NPMST-{}".format(i+1), **self._args) for i in range(self._npmst_num)]
        self._parsers.extend(self._npmst_parsers)

        self._mh4_num = kwargs.get("mh4_num", 0)
        self._mh4_weight = kwargs.get("mh4_weight", 1.0)
        self._mh4_parsers = [MH4Parser(self, id="MH4-{}".format(i+1), **self._args) for i in range(self._mh4_num)]
        self._parsers.extend(self._mh4_parsers)

        self._mh4t_num = kwargs.get("mh4t_num", 0)
        self._mh4t_weight = kwargs.get("mh4t_weight", 1.0)
        self._mh4t_parsers = [MH4TParser(self, id="MH4T-{}".format(i+1), **self._args) for i in range(self._mh4t_num)]
        self._parsers.extend(self._mh4t_parsers)

        self._oneec_num = kwargs.get("oneec_num", 0)
        self._oneec_weight = kwargs.get("oneec_weight", 1.0)
        self._oneec_parsers = [OneECParser(self, id="OneEC-{}".format(i+1), **self._args) for i in range(self._oneec_num)]
        self._parsers.extend(self._oneec_parsers)

        self._label_mlp_activation = self._activations[kwargs.get('label_mlp_activation', 'relu')]
        self._label_mlp_dims = kwargs.get("label_mlp_dims", 128)
        self._label_mlp_layers = kwargs.get("label_mlp_layers", 1)
        self._label_mlp_dropout = kwargs.get("label_mlp_dropout", 0.0)
        self._label_concat_dims = kwargs.get("label_concat_dims", 128)
        self._label_concat_layers = kwargs.get("label_concat_layers", 1)
        self._label_weight = kwargs.get("label_weight", 1.0)
        self._label_discrim = kwargs.get("label_discrim", False)
        self._label_biaffine = kwargs.get("label_biaffine", False)

        return self

    def _load_vocab(self, vocab):
        self._fullvocab = vocab
        self._upos = {p: i for i, p in enumerate(vocab["upos"])}
        self._iupos = vocab["upos"]
        self._xpos = {p: i for i, p in enumerate(vocab["xpos"])}
        self._ixpos = vocab["xpos"]
        self._vocab = {w: i + 3 for i, w in enumerate(vocab["vocab"])}
        self._wordfreq = vocab["wordfreq"]
        self._charset = {c: i + 3 for i, c in enumerate(vocab["charset"])}
        self._charfreq = vocab["charfreq"]
        self._rels = {r: i for i, r in enumerate(vocab["rels"])}
        self._irels = vocab["rels"]
        self._feats = {f: i + 1 for i, f in enumerate(vocab["feats"])}

    def load_vocab(self, filename):
        with open(filename, "r") as f:
            vocab = json.load(f)
        self._load_vocab(vocab)
        return self

    def save_vocab(self, filename):
        with open(filename, "w") as f:
            json.dump(self._fullvocab, f)
        return self

    def build_vocab(self, filename, savefile=None, cutoff=1):
        if isinstance(filename, str):
            graphs = read_conll(filename)
        elif isinstance(filename, list):
            graphs = []
            for f in filename:
                graphs.extend(read_conll(f))

        self._fullvocab= buildVocab(graphs, cutoff)

        if savefile:
            self.save_vocab(savefile)
        self._load_vocab(self._fullvocab)
        return self

    def save_model(self, filename):
        self.save_vocab(filename + ".vocab")
        with open(filename + ".params", "w") as f:
            json.dump(self._args, f)
        self._model.save(filename + ".model")
        return self

    def load_model(self, filename, **kwargs):
        self.load_vocab(filename + ".vocab")
        with open(filename + ".params", "r") as f:
            args = json.load(f)
            args.update(kwargs)
            self.create_parser(**args)
        self.init_model()
        self._model.populate(filename + ".model")
        return self


    def init_model(self):
        self._model = Model()
        if self._optimizer == "adam":
            self._trainer = AdamTrainer(self._model, alpha=self._learning_rate, beta_1 = self._beta1, beta_2=self._beta2, eps=self._epsilon)
        elif self._optimizer == "sgd":
            self._trainer = SimpleSGDTrainer(self._model, self._learning_rate)

        self._trainer.set_sparse_updates(self._sparse_updates)
        self._trainer.set_clip_threshold(self._clip)

        input_dims = 0
        if self._cdims > 0 and self._char_lstm_dims > 0:
            if self._char_lstm_dims > 0:
                self._char_lookup = self._model.add_lookup_parameters((len(self._charset) + 3, self._cdims))
                self._char_lstm = BiRNNBuilder(self._char_lstm_layers, self._cdims, self._char_lstm_dims, self._model, CoupledLSTMBuilder)
                if self._char_repr_method == "concat":
                    input_dims += self._char_lstm_dims

            if self._char_repr_method == "pred":
                self._char_to_word = Dense(self._char_lstm_dims, self._wdims, tanh, self._model)

        if self._wdims > 0:
            self._word_lookup = self._model.add_lookup_parameters((len(self._vocab) + 3, self._wdims))
            input_dims += self._wdims

        if self._pdims > 0:
            self._upos_lookup = self._model.add_lookup_parameters((len(self._upos) + 1, self._pdims))
            input_dims += self._pdims

        if self._fdims > 0:
            self._feats_lookup = self._model.add_lookup_parameters((len(self._feats) + 1, self._fdims))
            input_dims += self._fdims


        if input_dims <= 0:
            print("Input to LSTM is empty! You need to use at least one of word embeddings or character embeddings.")
            return

        if self._bidirectional:
            self._bilstm = BiRNNBuilder(self._bilstm_layers, input_dims, self._bilstm_dims, self._model, CoupledLSTMBuilder)
        else:
            self._bilstm = CoupledLSTMBuilder(self._bilstm_layers, input_dims, self._bilstm_dims, self._model)

        self._root_repr = self._model.add_parameters(input_dims)
        self._bos_repr = self._model.add_parameters(input_dims)
        self._eos_repr = self._model.add_parameters(input_dims)


        for utagger in self._utaggers:
            utagger.init_params()

        for xtagger in self._xtaggers:
            xtagger.init_params()

        for parser in self._parsers:
            parser.init_params()

        self._as_pad_repr = [self._model.add_parameters(self._bilstm_dims) for i in range(self._as_stack_features + self._as_buffer_features)]
        self._as_mlp = MultiLayerPerceptron([(self._bilstm_dims) * (self._as_stack_features + self._as_buffer_features)] + [self._as_mlp_dims] * self._as_mlp_layers, self._as_mlp_activation, self._model)
        self._as_final = Dense(self._as_mlp_dims, 3, identity, self._model)

        if self._label_biaffine:
            self._label_head_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._label_mlp_dims] * self._label_mlp_layers, self._label_mlp_activation, self._model)
            self._label_mod_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._label_mlp_dims] * self._label_mlp_layers, self._label_mlp_activation, self._model)
            self._label_scorer = Biaffine(self._label_mlp_dims, len(self._rels), self._model)
        else:
            self._label_mlp = MultiLayerPerceptron([(self._bilstm_dims) * 2] + [self._label_mlp_dims] * self._label_mlp_layers, self._label_mlp_activation, self._model)
            self._label_final = Dense(self._label_mlp_dims, len(self._rels), identity, self._model)

        return self

    def load_embeddings(self, filename, xz=False):
        if not os.path.isfile(filename):
            print(filename, "does not exist")
            return self

        if xz:
            f = lzma.open(filename, "rt", encoding="utf-8", errors="ignore")
        else:
            f = open(filename, "r", encoding="utf-8", errors="ignore")
        found_set = set()
        for line in f:
            l = line.strip().split()
            word = normalize(l[0])
            vec = np.array([float(x) for x in l[1:]])
            if word in self._vocab and len(vec) == self._wdims:
                found_set.add(word)
                self._word_lookup.init_row(self._vocab[word], vec)
        f.close()
        print("Loaded embeddings from", filename)
        print(len(found_set), "hits with vocab size of", len(self._vocab))

        return self

    def _next_epoch(self):
        self._epoch += 1
        return self


    def _get_lstm_features(self, sentence, train=False):
        carriers = [ComputationCarrier() for i in range(len(sentence))]
        carriers[0].vec = parameter(self._root_repr)
        carriers[0].word_id = 0
        carriers[0].pos_id = 0

        for entry, cc in zip(sentence[1:], carriers[1:]):
            cc.word_id = self._vocab.get(entry.norm, 0)
            cc.pos_id = self._upos.get(entry.upos, len(self._upos))
            vecs = []

            word_flag = False
            if self._wdims > 0:
                c = float(self._wordfreq.get(entry.norm, 0))
                word_flag = c > 0 and (not train or (random.random() < (c / (self._word_smooth + c))))

                wvec = lookup(self._word_lookup, int(self._vocab.get(entry.norm, 0)) if word_flag else 0)
                if self._char_repr_method == "concat" or word_flag:
                    vecs.append(wvec)

            if self._cdims > 0 and self._char_lstm_dims > 0:
                if not (self._char_repr_method == "pred" and word_flag):
                    char_vecs = []
                    char_vecs.append(lookup(self._char_lookup, 1))
                    for ch in entry.word:
                        c = float(self._charfreq.get(ch, 0))
                        keep_flag = not train or (random.random() < (c / (self._char_smooth + c)))
                        charvec = lookup(self._char_lookup, int(self._charset.get(ch, 0)) if keep_flag else 0)
                        if self._char_dropout > 0.:
                            char_vecs.append(block_dropout(charvec, self._char_dropout))
                        else:
                            char_vecs.append(charvec)
                    char_vecs.append(lookup(self._char_lookup, 2))

                    char_vecs = self._char_lstm.add_inputs(char_vecs)

                    cvec = concatenate([char_vecs[0][-1].output(), char_vecs[-1][0].output()])

                    if self._char_repr_method == "concat":
                        vecs.append(cvec)
                    elif not word_flag:
                        vecs.append(self._char_to_word(cvec))

            if self._pdims > 0:
                vecs.append(lookup(self._upos_lookup, int(self._upos.get(entry.upos, len(self._upos)))))


            if self._fdims > 0:
                feats = []
                for f in entry.feats_set:
                    if f in self._feats and ((not train) or random.random() > self._feature_dropout):
                        feats.append(lookup(self._feats_lookup, int(self._feats[f])))

                if len(feats) == 0:
                    feats.append(lookup(self._feats_lookup, 0))

                vecs.append(emax(feats))

            cc.vec = concatenate(vecs)

            if train and self._block_dropout > 0.:
                cc.vec = block_dropout(cc.vec, self._block_dropout)

        if self._bidirectional:
            ret = self._bilstm.transduce([x.vec for x in carriers[:]])
        else:
            ret = self._bilstm.initial_state().transduce([x.vec for x in carriers[:]])

        for vec, cc in zip(ret[:], carriers[:]):
            cc.vec = vec

        return carriers


    def _minibatch_update(self, loss, num_tokens):
        if len(loss) == 0:
            self._init_cg(train=True)
            return 0.

        loss = esum(loss) * (1. / self._batch_size)
        ret = loss.scalar_value()
        loss.backward()
        self._trainer.update()
        self._steps += 1
        self._init_cg(train=True)

        return ret * self._batch_size

    def _as_conf_eval(self, features, carriers):
        vecs = [carriers[f].vec if f >= 0 else parameter(self._as_pad_repr[i]) for i, f in enumerate(features)]

        exprs = self._as_final(self._as_mlp(concatenate(vecs)))

        return exprs.value(), exprs

    def _as_sent_eval(self, graph, carriers):
        gold_heads = graph.proj_heads

        loss = []
        beamconf = ASBeamConf(len(graph.nodes), 1, np.array(gold_heads), self._as_stack_features, self._as_buffer_features)
        beamconf.init_conf(0, True)
        total = 0
        wrong = 0

        while not beamconf.is_complete(0):
            valid = beamconf.valid_transitions(0)
            if np.count_nonzero(valid) < 1:
                break

            scores, exprs = self._as_conf_eval(beamconf.extract_features(0), carriers)
            best = beamconf.static_oracle(0)
            rest = tuple((i, s) for i, s in enumerate(scores) if i != best)
            total += 1
            if len(rest) > 0:
                second, secondScore = max(rest, key=lambda x: x[1])

                if scores[best] < scores[second] + 1.0:
                    loss.append(exprs[second] - exprs[best] + 1.)
                    wrong += 1

            beamconf.make_transition(0, best)

        return (total - wrong) / total * (len(graph.nodes) - 1), loss

    def ast(self, graph, carriers):
        beamconf = ASBeamConf(len(graph.nodes), 1, np.array(graph.heads), self._as_stack_features, self._as_buffer_features)
        beamconf.init_conf(0, True)

        while not beamconf.is_complete(0):
            valid = beamconf.valid_transitions(0)
            scores, exprs = self._as_conf_eval(beamconf.extract_features(0), carriers)
            action, _ = max(((i, s) for i, s in enumerate(scores) if valid[i]), key=lambda x: x[1])
            beamconf.make_transition(0, action)

        graph.heads = list(beamconf.get_heads(0))

        return self

    def _label_arc_eval(self, carriers, head, mod):
        if self._label_biaffine:
            expr = self._label_scorer(self._label_head_mlp(carriers[head].vec), self._label_mod_mlp(carriers[mod].vec))
        else:
            expr = self._label_final(self._label_mlp(concatenate([carriers[head].vec, carriers[mod].vec])))

        return expr.value(), expr

    def _label_sent_eval(self, graph, carriers):
        correct = 0
        loss = []
        for mod, head in enumerate(graph.heads):
            if mod > 0 and head >= 0:
                scores, exprs= self._label_arc_eval(carriers, head, mod)
                if not graph.rels[mod] in self._rels:
                    continue
                answer = self._rels[graph.rels[mod]]
                if np.argmax(scores) == answer:
                    correct += 1

                if self._label_discrim:
                    wrong_pred = max(((i, score) for i, score in enumerate(scores) if i != answer), key=lambda x: x[1])[0]
                    if scores[answer] < scores[wrong_pred] + 1.:
                        loss.append((exprs[wrong_pred] - exprs[answer] + 1.))
                else:
                    loss.append(pickneglogsoftmax(exprs, answer))
        return correct, loss

    def label(self, graph, carriers):
        for mod, head in enumerate(graph.heads):
            if mod > 0 and head >= 0:
                scores, exprs= self._label_arc_eval(carriers, head, mod)
                graph.rels[mod] = self._irels[np.argmax(scores)]

        return self

    def predict(self, graphs, **kwargs):
        ahdp = kwargs.get("ahdp", False)
        aedp = kwargs.get("aedp", False)
        ast = kwargs.get("ast", False)
        mst = kwargs.get("mst", False)
        npmst = kwargs.get("npmst", False)
        mh4 = kwargs.get("mh4", False)
        mh4t = kwargs.get("mh4t", False)
        oneec = kwargs.get("oneec", False)
        label = kwargs.get("label", False)

        parsers = []
        if mst: parsers.extend(self._mst_parsers)
        if npmst: parsers.extend(self._npmst_parsers)
        if mh4: parsers.extend(self._mh4_parsers)
        if mh4t: parsers.extend(self._mh4t_parsers)
        if oneec: parsers.extend(self._oneec_parsers)
        if ahdp: parsers.extend(self._ahdp_parsers)
        if aedp: parsers.extend(self._aedp_parsers)

        for graph in graphs:
            self._init_cg(train=False)
            carriers = self._get_lstm_features(graph.nodes, train=False)

            for parser in parsers:
                parser.predict(graph, carriers)

            if ast:
                self.ast(graph, carriers)

            if label:
                self.label(graph, carriers)

        return graphs


    def test(self, graphs=None, filename=None, **kwargs):
        utag = kwargs.get("utag", False)
        xtag = kwargs.get("xtag", False)
        ahdp = kwargs.get("ahdp", False)
        aedp = kwargs.get("aedp", False)
        ast = kwargs.get("ast", False)
        mst = kwargs.get("mst", False)
        npmst = kwargs.get("npmst", False)
        mh4 = kwargs.get("mh4", False)
        mh4t = kwargs.get("mh4t", False)
        oneec = kwargs.get("oneec", False)
        label = kwargs.get("label", False)
        save_prefix = kwargs.get("save_prefix", None)

        if graphs is None:
            graphs = read_conll(filename)

        total = 0
        correct_counts = defaultdict(int)
        as_uas_correct = 0
        as_las_correct = 0
        label_correct = 0
        ret = 0.

        parsers = []
        if mst: parsers.extend(self._mst_parsers)
        if npmst: parsers.extend(self._npmst_parsers)
        if mh4: parsers.extend(self._mh4_parsers)
        if mh4t: parsers.extend(self._mh4t_parsers)
        if oneec: parsers.extend(self._oneec_parsers)
        if ahdp: parsers.extend(self._ahdp_parsers)
        if aedp: parsers.extend(self._aedp_parsers)

        for gold_graph in graphs:
            self._init_cg(train=False)
            graph = gold_graph.cleaned(node_level=False)
            total += len(graph.nodes) - 1
            carriers = self._get_lstm_features(graph.nodes, train=False)

            if utag:
                gold_upos = [x.upos for x in graph.nodes]
                for utagger in self._utaggers:
                    utagger.predict(graph, carriers)
                    predicted = [x.upos for x in graph.nodes]
                    correct_counts["{} Accuracy".format(utagger.id)] += POSCorrect(predicted, gold_upos)

            if xtag:
                gold_xpos = [x.xpos for x in graph.nodes]
                for xtagger in self._xtaggers:
                    xtagger.predict(graph, carriers)
                    predicted = [x.xpos for x in graph.nodes]
                    correct_counts["{} Accuracy".format(xtagger.id)] += POSCorrect(predicted, gold_xpos)

            for parser in parsers:
                parser.predict(graph, carriers)
                if label:
                    self.label(graph, carriers)

                for i in range(1, len(graph.nodes)):
                    if gold_graph.heads[i] == graph.heads[i]:
                        correct_counts["{}-UAS".format(parser.id)] += 1
                        if gold_graph.rels[i].split(":")[0] == graph.rels[i].split(":")[0]:
                            correct_counts["{}-LAS".format(parser.id)] += 1
                if save_prefix:
                    write_conll("{}_{}_{}.conllu".format(save_prefix, self._epoch, parser.id), [graph], append=True)

            if ast:
                self.ast(graph, carriers)
                if label:
                    self.label(graph, carriers)

                for i in range(1, len(graph.nodes)):
                    if gold_graph.heads[i] == graph.heads[i]:
                        as_uas_correct += 1
                        if gold_graph.rels[i].split(":")[0] == graph.rels[i].split(":")[0]:
                            as_las_correct += 1
                if save_prefix:
                    write_conll("{}_{}_as.conllu".format(save_prefix, self._epoch), [graph], append=True)

            if len(parsers) == 0 and not ast and label:
                graph.heads = np.copy(gold_graph.heads)
                self.label(graph, carriers)
                for i in range(1, len(graph.nodes)):
                    if gold_graph.rels[i].split(":")[0] == graph.rels[i].split(":")[0]:
                        label_correct += 1


        for id in sorted(correct_counts):
            print(id, correct_counts[id] / total)
            if label and "-LAS" in id:
                ret = max(ret, correct_counts[id])
            if not label and "-UAS" in id:
                ret = max(ret, correct_counts[id])

        if ast:
            if label:
                print("AS-UAS", as_uas_correct / total)
                print("AS-LAS", as_las_correct / total)
                ret = max(ret, as_las_correct)
            else:
                print("AS-UAS", as_uas_correct / total)
                ret = max(ret, as_uas_correct)

        if len(parsers) == 0 and not ast and label:
            print("LA", label_correct / total)
            ret = max(ret, label_correct)

        return ret / total


    def fine_tune(self, filename, ratio=0.95, max_steps=1000, eval_steps=100, decay_evals=5, decay_times=0, dev=None, **kwargs):
        graphs = read_conll(filename)
        graphs_list = [list(random.sample(graphs, int(len(graphs) * ratio)))]

        return self.train("", max_steps, eval_steps, decay_evals, decay_times, dev, graphs_list, **kwargs)

    def train_small(self, filename, split_ratio=0.9, **kwargs):
        save_prefix = kwargs.get("save_prefix", None)
        graphs = read_conll(filename)
        random.shuffle(graphs)
        train_len = int(len(graphs) * split_ratio)
        train_graphs = [graphs[:train_len]]
        dev_graphs = graphs[train_len:]

        if save_prefix is not None:
            write_conll("{}_train.conllu".format(save_prefix), train_graphs[0])
            write_conll("{}_dev.conllu".format(save_prefix), dev_graphs)

        return self.train("", graphs_list=train_graphs, dev_graphs=dev_graphs, **kwargs)


    def train(self, filename, max_steps=1000, eval_steps=100, decay_evals=5, decay_times=0, dev=None, graphs_list=None, dev_portion=0.8, dev_graphs=None, **kwargs):
        if graphs_list is None:
            if isinstance(filename, str):
                graphs_list = [read_conll(filename)]
            elif isinstance(filename, list):
                graphs_list = [read_conll(f) for f in filename]

        total = 0
        proj_count = 0
        total_trees = 0
        oneec = kwargs.get("oneec", False)
        for graphs in graphs_list:
            total_trees += len(graphs)
            for g in graphs:
                total += len(g.nodes) - 1
                if is_projective(g.heads):
                    g.proj_heads = g.heads
                    g.mh4_heads = g.heads
                    proj_count += 1
                else:
                    g.proj_heads = projectivize(g.heads)
                    if len(g.nodes) < 100:
                        g.mh4_heads = mh4ize(g.heads)
                if oneec and len(g.nodes) < 100:
                    scores = np.zeros((len(g.nodes) + 1, len(g.nodes) + 1, 4))
                    for m, h in enumerate(g.heads):
                        scores[h + 1, m + 1, 1] += 1
                        scores[h + 1, m + 1, 3] += 1
                    g.oneec_heads, traces = parse_1ec_o3(scores)
                    g.oneec_traces = {(int(i), int(j), int(r)) for i, j, r in traces if r >= 0}

        print("Training set projective ratio", proj_count / total_trees)

        train_set_steps = total / self._batch_size

        eval_steps = max(int(train_set_steps * 0.25), eval_steps)

        save_prefix = kwargs.get("save_prefix", None)

        if dev is not None and dev_graphs is None:
            dev_graphs = read_conll(dev)
            dev_samples = int(len(dev_graphs) * dev_portion)
            dev_graphs = list(random.sample(dev_graphs, dev_samples))
            dev_graphs = [g for g in dev_graphs if len(g.nodes) < 100]
            if save_prefix is not None:
                write_conll("{}_dev.conllu".format(save_prefix), dev_graphs)


        utag = kwargs.get("utag", False)
        xtag = kwargs.get("xtag", False)
        ahdp = kwargs.get("ahdp", False)
        aedp = kwargs.get("aedp", False)
        ast = kwargs.get("ast", False)
        mst = kwargs.get("mst", False)
        npmst = kwargs.get("npmst", False)
        mh4 = kwargs.get("mh4", False)
        mh4t = kwargs.get("mh4t", False)
        oneec = kwargs.get("oneec", False)
        label = kwargs.get("label", False)

        self._steps = 0
        self._epoch = 0
        self._base_lr = 1.
        max_dev = 0.
        max_dev_ep = 0

        i = 0
        t0 = time.time()

        self._init_cg(train=True)
        loss = []
        loss_sum = 0.0
        total_tokens = 0
        num_tokens = 0
        correct_counts = defaultdict(float)

        as_correct = 0
        label_correct = 0

        for graph in shuffled_balanced_stream(graphs_list):
            i += 1
            if i % 100 == 0:
                print(i, "{0:.2f}s".format(time.time() - t0), end=" ")
                sys.stdout.flush()
                t0 = time.time()

            carriers = self._get_lstm_features(graph.nodes, train=True)
            num_tokens += len(graph.nodes) - 1
            total_tokens += len(graph.nodes) - 1

            if utag:
                for utagger in self._utaggers:
                    c, l = utagger.sent_loss(graph, carriers)
                    correct_counts[utagger.id] += c
                    if len(l) > 0:
                        loss.append(esum(l) * (self._utagger_weight / self._utagger_num))

            if xtag:
                for xtagger in self._xtaggers:
                    c, l = xtagger.sent_loss(graph, carriers)
                    correct_counts[xtagger.id] += c
                    if len(l) > 0:
                        loss.append(esum(l) * (self._xtagger_weight / self._xtagger_num))

            if ast:
                c, l = self._as_sent_eval(graph, carriers)
                as_correct += c
                if len(l) > 0:
                    loss.append(esum(l) * self._as_weight)

            if mst:# and len(graph.nodes) < 100:
                for mstparser in self._mst_parsers:
                    c, l = mstparser.sent_loss(graph, carriers)
                    correct_counts[mstparser.id] += c
                    if len(l) > 0:
                        loss.append(esum(l) * (self._mst_weight / self._mst_num))

            if npmst:# and len(graph.nodes) < 100:
                for npmstparser in self._npmst_parsers:
                    c, l = npmstparser.sent_loss(graph, carriers)
                    correct_counts[npmstparser.id] += c
                    if len(l) > 0:
                        loss.append(esum(l) * (self._npmst_weight / self._npmst_num))

            if mh4 and len(graph.nodes) < 100:
                for mh4parser in self._mh4_parsers:
                    c, l = mh4parser.sent_loss(graph, carriers)
                    correct_counts[mh4parser.id] += c
                    if len(l) > 0:
                        loss.append(esum(l) * (self._mh4_weight / self._mh4_num))

            if mh4t and len(graph.nodes) < 100:
                for mh4tparser in self._mh4t_parsers:
                    c, l = mh4tparser.sent_loss(graph, carriers)
                    correct_counts[mh4tparser.id] += c
                    if len(l) > 0:
                        loss.append(esum(l) * (self._mh4t_weight / self._mh4t_num))

            if oneec and len(graph.nodes) < 100:
                for oneecparser in self._oneec_parsers:
                    c, l = oneecparser.sent_loss(graph, carriers)
                    correct_counts[oneecparser.id] += c
                    if len(l) > 0:
                        loss.append(esum(l) * (self._oneec_weight / self._oneec_num))

            if ahdp:# and len(graph.nodes) < 100:
                for ahdpparser in self._ahdp_parsers:
                    c, l = ahdpparser.sent_loss(graph, carriers)
                    correct_counts[ahdpparser.id] += c
                    if len(l) > 0:
                        loss.append(esum(l) * (self._ahdp_weight / self._ahdp_num))

            if aedp:# and len(graph.nodes) < 100:
                for aedpparser in self._aedp_parsers:
                    c, l = aedpparser.sent_loss(graph, carriers)
                    correct_counts[aedpparser.id] += c
                    if len(l) > 0:
                        loss.append(esum(l) * (self._aedp_weight / self._aedp_num))

            if label:
                c, l = self._label_sent_eval(graph, carriers)
                label_correct += c
                if len(l) > 0:
                    loss.append(esum(l) * self._label_weight)

            if num_tokens >= self._batch_size:
                loss_sum += self._minibatch_update(loss, num_tokens)
                loss = []
                num_tokens = 0

                if self._steps % eval_steps == 0:

                    self._next_epoch()
                    print()
                    self._trainer.status()

                    print()
                    print("Total Loss", loss_sum, "Avg", loss_sum / total_tokens)
                    for id in sorted(correct_counts):
                        print("Train {} Acc".format(id), correct_counts[id] / total_tokens)
                    if ast:
                        print("Train AS Acc", as_correct / total_tokens)
                    if label:
                        print("Train Label Acc", label_correct / total_tokens)

                    loss_sum = 0.0
                    total_tokens = 0
                    num_tokens = 0
                    correct_counts = defaultdict(float)
                    as_correct = 0
                    label_correct = 0

                    if self._steps >= max_steps:
                        break

                    if dev_graphs is not None:
                        performance = self.test(graphs=dev_graphs, **kwargs)
                        self._init_cg(train=True)

                        if performance >= max_dev:
                            max_dev = performance
                            max_dev_ep = 0
                            if save_prefix:
                                self.save_model("{}_{}_model".format(save_prefix, max_dev_ep))
                        else:
                            max_dev_ep += 1

                        if max_dev_ep >= decay_evals:
                            if decay_times > 0:
                                decay_times -= 1
                                max_dev_ep = 0
                                self._base_lr /= 4.
                                self._trainer.restart(self._learning_rate * self._base_lr)
                                print("Learning rate decayed!")
                                print("Current decay ratio", self._base_lr * math.pow(self._anneal_base, self._steps / self._anneal_steps))
                            else:
                                break

        return self

    def _init_cg(self, train=False):
        renew_cg()
        if train:
            self._bilstm.set_dropout(self._bilstm_dropout)
            if self._cdims > 0 and self._char_lstm_dims > 0:
                self._char_lstm.set_dropout(self._char_lstm_dropout)

            self._as_mlp.set_dropout(self._as_mlp_dropout)
            if self._label_biaffine:
                self._label_head_mlp.set_dropout(self._label_mlp_dropout)
                self._label_mod_mlp.set_dropout(self._label_mlp_dropout)
            else:
                self._label_mlp.set_dropout(self._label_mlp_dropout)
        else:
            self._bilstm.set_dropout(0.)
            if self._cdims > 0 and self._char_lstm_dims > 0:
                self._char_lstm.set_dropout(0.)

            self._as_mlp.set_dropout(0.)
            if self._label_biaffine:
                self._label_head_mlp.set_dropout(0.)
                self._label_mod_mlp.set_dropout(0.)
            else:
                self._label_mlp.set_dropout(0.)
        for utagger in self._utaggers:
            utagger.init_cg(train)
        for xtagger in self._xtaggers:
            xtagger.init_cg(train)

        for parser in self._parsers:
            parser.init_cg(train)

    def finish(self, **kwargs):
        print()


if __name__ == '__main__':
    fire.Fire(CDParser)
