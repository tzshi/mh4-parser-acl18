#!/usr/bin/env python
# encoding: utf-8

from dynet import *
from collections import Counter
import numpy as np

from . import pyximportcpp; pyximportcpp.install()
from .calgorithm import parse_proj, parse_ah_dp_mst, parse_ae_dp_mst
from .chu_liu_edmonds import chu_liu_edmonds
from .mh4 import parse_mh4
from .mh4t import parse_mh4t, parse_mh4t_sh, mh4t_combine_scores
from .ec import parse_1ec_o3
from .ahbeamconf import AHBeamConf
from .aebeamconf import AEBeamConf
from .mh4beamconf import MH4BeamConf

from .layers import MultiLayerPerceptron, Dense, Bilinear, identity, BiaffineBatch


class UPOSTagger:

    def __init__(self, parser, id="UPOSTagger", **kwargs):
        self._parser = parser
        self.id = id

        self._activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}

        self._bilstm_dims = kwargs.get("bilstm_dims", 128)
        self._utagger_mlp_activation = self._activations[kwargs.get('utagger_mlp_activation', 'relu')]
        self._utagger_mlp_dims = kwargs.get("utagger_mlp_dims", 128)
        self._utagger_mlp_layers = kwargs.get("utagger_mlp_layers", 2)
        self._utagger_mlp_dropout = kwargs.get("utagger_mlp_dropout", 0.0)
        self._utagger_discrim = kwargs.get("utagger_discrim", False)

    def init_params(self):
        self._utagger_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._utagger_mlp_dims] * self._utagger_mlp_layers, self._utagger_mlp_activation, self._parser._model)
        self._utagger_final = Dense(self._utagger_mlp_dims, len(self._parser._upos), identity, self._parser._model)

    def init_cg(self, train=False):
        if train:
            self._utagger_mlp.set_dropout(self._utagger_mlp_dropout)
        else:
            self._utagger_mlp.set_dropout(0.)

    def sent_loss(self, graph, carriers):
        ret = []
        correct = 0

        for node, c in zip(graph.nodes[1:], carriers[1:]):
            potentials = self._utagger_final(self._utagger_mlp(c.vec))
            pred = np.argmax(potentials.value())
            answer = self._parser._upos[node.upos]

            if (pred == answer):
                correct += 1

            if self._utagger_discrim:
                potential_values = potentials.value()
                best_wrong = max([(i, val) for i, val in enumerate(potential_values) if i != answer], key=lambda x: x[1])

                if best_wrong[1] + 1. > potential_values[answer]:
                    ret.append((potentials[best_wrong[0]] - potentials[answer] + 1.))
            else:
                ret.append(pickneglogsoftmax(potentials, answer))
        return correct, ret

    def predict(self, graph, carriers):
        for node, c in zip(graph.nodes[1:], carriers[1:]):
            potentials = self._utagger_final(self._utagger_mlp(c.vec))
            pred = np.argmax(potentials.value())
            node.upos = self._parser._iupos[pred]

        return self


class XPOSTagger:

    def __init__(self, parser, id="XPOSTagger", **kwargs):
        self._parser = parser
        self.id = id

        self._activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}

        self._bilstm_dims = kwargs.get("bilstm_dims", 128)
        self._xtagger_mlp_activation = self._activations[kwargs.get('xtagger_mlp_activation', 'relu')]
        self._xtagger_mlp_dims = kwargs.get("xtagger_mlp_dims", 128)
        self._xtagger_mlp_layers = kwargs.get("xtagger_mlp_layers", 2)
        self._xtagger_mlp_dropout = kwargs.get("xtagger_mlp_dropout", 0.0)
        self._xtagger_discrim = kwargs.get("xtagger_discrim", False)

    def init_params(self):
        self._xtagger_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._xtagger_mlp_dims] * self._xtagger_mlp_layers, self._xtagger_mlp_activation, self._parser._model)
        self._xtagger_final = Dense(self._xtagger_mlp_dims, len(self._parser._xpos), identity, self._parser._model)

    def init_cg(self, train=False):
        if train:
            self._xtagger_mlp.set_dropout(self._xtagger_mlp_dropout)
        else:
            self._xtagger_mlp.set_dropout(0.)

    def sent_loss(self, graph, carriers):
        ret = []
        correct = 0

        for node, c in zip(graph.nodes[1:], carriers[1:]):
            potentials = self._xtagger_final(self._xtagger_mlp(c.vec))
            pred = np.argmax(potentials.value())
            answer = self._parser._xpos[node.xupos]

            if (pred == answer):
                correct += 1

            if self._xtagger_discrim:
                potential_values = potentials.value()
                best_wrong = max([(i, val) for i, val in enumerate(potential_values) if i != answer], key=lambda x: x[1])

                if best_wrong[1] + 1. > potential_values[answer]:
                    ret.append(potentials[best_wrong[0]] - potentials[answer] + 1.)
            else:
                ret.append(pickneglogsoftmax(potentials, answer))
        return correct, ret

    def predict(self, graph, carriers):
        for node, c in zip(graph.nodes[1:], carriers[1:]):
            potentials = self._xtagger_final(self._xtagger_mlp(c.vec))
            pred = np.argmax(potentials.value())
            node.xpos = self._parser._ixpos[pred].split("|")[1]

        return self


class MSTParser:

    def __init__(self, parser, id="MSTParser", **kwargs):
        self._parser = parser
        self.id = id

        self._activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}

        self._bilstm_dims = kwargs.get("bilstm_dims", 128)
        self._mst_mlp_activation = self._activations[kwargs.get('mst_mlp_activation', 'relu')]
        self._mst_mlp_dims = kwargs.get("mst_mlp_dims", 128)
        self._mst_mlp_layers = kwargs.get("mst_mlp_layers", 2)
        self._mst_mlp_dropout = kwargs.get("mst_mlp_dropout", 0.0)

    def init_params(self):
        self._mst_head_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._mst_mlp_dims] * self._mst_mlp_layers, self._mst_mlp_activation, self._parser._model)
        self._mst_mod_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._mst_mlp_dims] * self._mst_mlp_layers, self._mst_mlp_activation, self._parser._model)
        self._mst_bilinear = Bilinear(self._mst_mlp_dims, self._parser._model)
        self._mst_head_bias = Dense(self._mst_mlp_dims, 1, identity, self._parser._model)

    def init_cg(self, train=False):
        if train:
            self._mst_head_mlp.set_dropout(self._mst_mlp_dropout)
            self._mst_mod_mlp.set_dropout(self._mst_mlp_dropout)
        else:
            self._mst_head_mlp.set_dropout(0.)
            self._mst_mod_mlp.set_dropout(0.)

    def _mst_arcs_eval(self, carriers):
        head_vecs = [self._mst_head_mlp(c.vec) for c in carriers]
        mod_vecs = [self._mst_mod_mlp(c.vec) for c in carriers]

        head_vecs = concatenate(head_vecs, 1)
        mod_vecs = concatenate(mod_vecs, 1)

        exprs = colwise_add(self._mst_bilinear(head_vecs, mod_vecs), reshape(self._mst_head_bias(head_vecs), (len(carriers),)))

        scores = exprs.value()

        exprs = np.array([[exprs[i][j] for j in range(len(carriers))] for i in range(len(carriers))])

        return scores, exprs

    def sent_loss(self, graph, carriers):
        gold_heads = graph.proj_heads

        scores, exprs = self._mst_arcs_eval(carriers)

        # Cost Augmentation
        for m, h in enumerate(gold_heads):
            scores[h, m] -= 1.

        heads = parse_proj(scores)

        correct = sum([1 for (h, g) in zip(heads[1:], gold_heads[1:]) if h == g])
        loss = [exprs[int(h)][int(i)] - exprs[int(g)][int(i)] + 1. for i, (h, g) in enumerate(zip(heads, gold_heads)) if h != g]

        return correct, loss

    def predict(self, graph, carriers):
        scores, exprs = self._mst_arcs_eval(carriers)
        graph.heads = parse_proj(scores)

        return self


class NPMSTParser:

    def __init__(self, parser, id="NPMSTParser", **kwargs):
        self._parser = parser
        self.id = id

        self._activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}

        self._bilstm_dims = kwargs.get("bilstm_dims", 128)
        self._npmst_mlp_activation = self._activations[kwargs.get('npmst_mlp_activation', 'relu')]
        self._npmst_mlp_dims = kwargs.get("npmst_mlp_dims", 128)
        self._npmst_mlp_layers = kwargs.get("npmst_mlp_layers", 2)
        self._npmst_mlp_dropout = kwargs.get("npmst_mlp_dropout", 0.0)

    def init_params(self):
        self._npmst_head_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._npmst_mlp_dims] * self._npmst_mlp_layers, self._npmst_mlp_activation, self._parser._model)
        self._npmst_mod_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._npmst_mlp_dims] * self._npmst_mlp_layers, self._npmst_mlp_activation, self._parser._model)
        self._npmst_bilinear = Bilinear(self._npmst_mlp_dims, self._parser._model)
        self._npmst_head_bias = Dense(self._npmst_mlp_dims, 1, identity, self._parser._model)

    def init_cg(self, train=False):
        if train:
            self._npmst_head_mlp.set_dropout(self._npmst_mlp_dropout)
            self._npmst_mod_mlp.set_dropout(self._npmst_mlp_dropout)
        else:
            self._npmst_head_mlp.set_dropout(0.)
            self._npmst_mod_mlp.set_dropout(0.)

    def _npmst_arcs_eval(self, carriers):
        head_vecs = [self._npmst_head_mlp(c.vec) for c in carriers]
        mod_vecs = [self._npmst_mod_mlp(c.vec) for c in carriers]

        head_vecs = concatenate(head_vecs, 1)
        mod_vecs = concatenate(mod_vecs, 1)

        exprs = colwise_add(self._npmst_bilinear(head_vecs, mod_vecs), reshape(self._npmst_head_bias(head_vecs), (len(carriers),)))

        scores = exprs.value()

        exprs = np.array([[exprs[i][j] for j in range(len(carriers))] for i in range(len(carriers))])

        return scores, exprs

    def sent_loss(self, graph, carriers):
        gold_heads = graph.heads
        scores, exprs = self._npmst_arcs_eval(carriers)

        # Cost Augmentation
        for m, h in enumerate(gold_heads):
            scores[h, m] -= 1.

        heads, tree_score = chu_liu_edmonds(scores.T)
        correct = sum([1 for (h, g) in zip(heads[1:], gold_heads[1:]) if h == g])
        loss = [exprs[int(h)][int(i)] - exprs[int(g)][int(i)] + 1. for i, (h, g) in enumerate(zip(heads, gold_heads)) if h != g]

        return correct, loss

    def predict(self, graph, carriers):
        scores, exprs = self._npmst_arcs_eval(carriers)
        graph.heads, tree_score = chu_liu_edmonds(scores.T)

        return self


class MH4Parser:

    def __init__(self, parser, id="MH4Parser", **kwargs):
        self._parser = parser
        self.id = id

        self._activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}

        self._bilstm_dims = kwargs.get("bilstm_dims", 128)
        self._mh4_mlp_activation = self._activations[kwargs.get('mh4_mlp_activation', 'relu')]
        self._mh4_mlp_dims = kwargs.get("mh4_mlp_dims", 128)
        self._mh4_mlp_layers = kwargs.get("mh4_mlp_layers", 2)
        self._mh4_mlp_dropout = kwargs.get("mh4_mlp_dropout", 0.0)

    def init_params(self):
        self._mh4_head_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._mh4_mlp_dims] * self._mh4_mlp_layers, self._mh4_mlp_activation, self._parser._model)
        self._mh4_mod_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._mh4_mlp_dims] * self._mh4_mlp_layers, self._mh4_mlp_activation, self._parser._model)
        self._mh4_bilinear = Bilinear(self._mh4_mlp_dims, self._parser._model)
        self._mh4_head_bias = Dense(self._mh4_mlp_dims, 1, identity, self._parser._model)

    def init_cg(self, train=False):
        if train:
            self._mh4_head_mlp.set_dropout(self._mh4_mlp_dropout)
            self._mh4_mod_mlp.set_dropout(self._mh4_mlp_dropout)
        else:
            self._mh4_head_mlp.set_dropout(0.)
            self._mh4_mod_mlp.set_dropout(0.)

    def _mh4_arcs_eval(self, carriers):
        head_vecs = [self._mh4_head_mlp(c.vec) for c in carriers]
        mod_vecs = [self._mh4_mod_mlp(c.vec) for c in carriers]

        head_vecs = concatenate(head_vecs, 1)
        mod_vecs = concatenate(mod_vecs, 1)

        exprs = colwise_add(self._mh4_bilinear(head_vecs, mod_vecs), reshape(self._mh4_head_bias(head_vecs), (len(carriers),)))

        scores = exprs.value()

        exprs = np.array([[exprs[i][j] for j in range(len(carriers))] for i in range(len(carriers))])

        return scores, exprs

    def sent_loss(self, graph, carriers):
        gold_heads = graph.mh4_heads
        scores, exprs = self._mh4_arcs_eval(carriers)

        # Cost Augmentation
        for m, h in enumerate(gold_heads):
            scores[h, m] -= 1.

        heads = parse_mh4(scores)
        correct = sum([1 for (h, g) in zip(heads[1:], gold_heads[1:]) if h == g])
        loss = [exprs[int(h)][int(i)] - exprs[int(g)][int(i)] + 1. for i, (h, g) in enumerate(zip(heads, gold_heads)) if h != g]

        return correct, loss

    def predict(self, graph, carriers):
        scores, exprs = self._mh4_arcs_eval(carriers)
        graph.heads = parse_mh4(scores)

        return self


class MH4TParser:

    def __init__(self, parser, id="MH4TParser", **kwargs):
        self._parser = parser
        self.id = id

        self._activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}

        self._bilstm_dims = kwargs.get("bilstm_dims", 128)
        self._mh4t_mlp_activation = self._activations[kwargs.get('mh4t_mlp_activation', 'relu')]
        self._mh4t_mlp_dims = kwargs.get("mh4t_mlp_dims", 128)
        self._mh4t_mlp_layers = kwargs.get("mh4t_mlp_layers", 2)
        self._mh4t_mlp_dropout = kwargs.get("mh4t_mlp_dropout", 0.0)
        self._mh4t_mode = kwargs.get("mh4t_mode", "two")
        self._mh4t_stack_features = kwargs.get("mh4t_stack_features", 1)
        self._mh4t_buffer_features = kwargs.get("mh4t_buffer_features", 1)

    def init_params(self):
        if self._mh4t_mode == "local":
            self._mh4t_pad_repr = [self._parser._model.add_parameters(self._bilstm_dims) for i in range(self._mh4t_stack_features + self._mh4t_buffer_features)]
            self._mh4t_mlp = MultiLayerPerceptron([self._bilstm_dims * (self._mh4t_stack_features + self._mh4t_buffer_features)] + [self._mh4t_mlp_dims] * self._mh4t_mlp_layers, self._mh4t_mlp_activation, self._parser._model)
            self._mh4t_final = Dense(self._mh4t_mlp_dims, 7, identity, self._parser._model)
        elif self._mh4t_mode == "two":
            self._mh4t_pad_repr = self._parser._model.add_parameters(self._bilstm_dims)
            self._mh4t_stack_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._mh4t_mlp_dims] * self._mh4t_mlp_layers, self._mh4t_mlp_activation, self._parser._model)
            self._mh4t_buffer_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._mh4t_mlp_dims] * self._mh4t_mlp_layers, self._mh4t_mlp_activation, self._parser._model)
            self._mh4t_biaffine = BiaffineBatch(self._mh4t_mlp_dims, 7, self._parser._model)
        elif self._mh4t_mode == "hybrid":
            self._mh4t_pad_repr = self._parser._model.add_parameters(self._bilstm_dims)
            self._mh4t_s1_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._mh4t_mlp_dims] * self._mh4t_mlp_layers, self._mh4t_mlp_activation, self._parser._model)
            self._mh4t_s0_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._mh4t_mlp_dims] * self._mh4t_mlp_layers, self._mh4t_mlp_activation, self._parser._model)
            self._mh4t_b0_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._mh4t_mlp_dims] * self._mh4t_mlp_layers, self._mh4t_mlp_activation, self._parser._model)
            self._mh4t_shift_biaffine = BiaffineBatch(self._mh4t_mlp_dims, 1, self._parser._model)
            self._mh4t_reduce_biaffine1 = BiaffineBatch(self._mh4t_mlp_dims, 6, self._parser._model)
            self._mh4t_reduce_biaffine2 = BiaffineBatch(self._mh4t_mlp_dims, 6, self._parser._model)
        elif self._mh4t_mode == "b0":
            self._mh4t_pad_repr = self._parser._model.add_parameters(self._bilstm_dims)
            self._mh4t_b0_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._mh4t_mlp_dims] * self._mh4t_mlp_layers + [7], self._mh4t_mlp_activation, self._parser._model)

    def init_cg(self, train=False):
        if self._mh4t_mode == "local":
            if train:
                self._mh4t_mlp.set_dropout(self._mh4t_mlp_dropout)
            else:
                self._mh4t_mlp.set_dropout(0.)
        if self._mh4t_mode == "two":
            if train:
                self._mh4t_stack_mlp.set_dropout(self._mh4t_mlp_dropout)
                self._mh4t_buffer_mlp.set_dropout(self._mh4t_mlp_dropout)
            else:
                self._mh4t_stack_mlp.set_dropout(0.)
                self._mh4t_buffer_mlp.set_dropout(0.)
        elif self._mh4t_mode == "hybrid":
            if train:
                self._mh4t_s1_mlp.set_dropout(self._mh4t_mlp_dropout)
                self._mh4t_s0_mlp.set_dropout(self._mh4t_mlp_dropout)
                self._mh4t_b0_mlp.set_dropout(self._mh4t_mlp_dropout)
            else:
                self._mh4t_s1_mlp.set_dropout(0.)
                self._mh4t_s0_mlp.set_dropout(0.)
                self._mh4t_b0_mlp.set_dropout(0.)
        elif self._mh4t_mode == "b0":
            if train:
                self._mh4t_b0_mlp.set_dropout(self._mh4t_mlp_dropout)
            else:
                self._mh4t_b0_mlp.set_dropout(0.)

    def _mh4t_conf_eval(self, features, carriers):
        vecs = [carriers[int(f)].vec if f >= 0 else parameter(self._mh4t_pad_repr[i]) for i, f in enumerate(features)]
        exprs = self._mh4t_final(self._mh4t_mlp(concatenate(vecs)))

        return exprs.value(), exprs

    def _mh4t_eval(self, carriers):
        if self._mh4t_mode == "two":
            stack_vecs = [self._mh4t_stack_mlp(c.vec) for c in carriers] + [parameter(self._mh4t_pad_repr)]
            buffer_vecs = [self._mh4t_buffer_mlp(c.vec) for c in carriers] + [parameter(self._mh4t_pad_repr)]

            stack_vecs = concatenate(stack_vecs, 1)
            buffer_vecs = concatenate(buffer_vecs, 1)

            exprs = self._mh4t_biaffine(stack_vecs, buffer_vecs)

            scores = exprs.value()

            return scores, exprs
        elif self._mh4t_mode == "b0":
            b0_vecs = concatenate([c.vec for c in carriers] + [parameter(self._mh4t_pad_repr)], 1)
            b0_vecs = transpose(self._mh4t_b0_mlp(b0_vecs))

            exprs = [b0_vecs] * (len(carriers) + 1)
            scores = np.array([b0_vecs.value()] * (len(carriers) + 1))

            return scores, exprs
        elif self._mh4t_mode == "hybrid":
            s1_vecs = [self._mh4t_s1_mlp(c.vec) for c in carriers] + [parameter(self._mh4t_pad_repr)]
            s0_vecs = [self._mh4t_s0_mlp(c.vec) for c in carriers] + [parameter(self._mh4t_pad_repr)]
            b0_vecs = [self._mh4t_b0_mlp(c.vec) for c in carriers] + [parameter(self._mh4t_pad_repr)]

            s1_vecs = concatenate(s1_vecs, 1)
            s0_vecs = concatenate(s0_vecs, 1)
            b0_vecs = concatenate(b0_vecs, 1)

            sh_exprs = self._mh4t_shift_biaffine(s0_vecs, b0_vecs)
            sh_scores = sh_exprs.value()

            re_exprs1 = self._mh4t_reduce_biaffine1(s1_vecs, s0_vecs)
            re_exprs2 = self._mh4t_reduce_biaffine2(s0_vecs, b0_vecs)
            re_scores1 = re_exprs1.value()
            re_scores2 = re_exprs2.value()

            return sh_scores, mh4t_combine_scores(re_scores1, re_scores2), sh_exprs, re_exprs1, re_exprs2

    def sent_loss(self, graph, carriers):
        if self._mh4t_mode == "local":
            loss = []
            gold_heads = graph.mh4_heads

            beamconf = MH4BeamConf(len(graph.nodes), 1, self._mh4t_stack_features, self._mh4t_buffer_features)
            beamconf.init_conf(0)

            scores = np.zeros((len(gold_heads) + 1, len(gold_heads) + 1, 7))
            mst_scores = np.ones((len(gold_heads), len(gold_heads)))
            mst_scores += -np.inf
            for m, h in enumerate(gold_heads):
                mst_scores[h, m] = 0.

            gscore, _, gold_traces = parse_mh4t(scores, mst_scores)

            total = 0
            wrong = 0
            for gi, gj, gr in gold_traces:
                valid = beamconf.valid_transitions(0)
                if np.count_nonzero(valid) < 1:
                    break

                scores, exprs = self._mh4t_conf_eval(beamconf.extract_features(0), carriers)
                best = int(gr)
                rest = tuple((i, s) for i, s in enumerate(scores) if i != best)
                total += 1
                if len(rest) > 0:
                    second, secondScore = max(rest, key=lambda x: x[1])

                    if scores[best] < scores[second] + 1.0:
                        loss.append(exprs[second] - exprs[best] + 1.)
                        wrong += 1

                beamconf.make_transition(0, best)

            return (total - wrong) / total * (len(graph.nodes) - 1), loss

        elif self._mh4t_mode == "two" or self._mh4t_mode == "b0":
            gold_heads = graph.mh4_heads

            scores, exprs = self._mh4t_eval(carriers)

            mst_scores = np.ones((len(gold_heads), len(gold_heads)))

            # Cost Augmentation
            for m, h in enumerate(gold_heads):
                mst_scores[h, m] -= 1.

            pscore, heads, traces = parse_mh4t(scores, mst_scores)

            mst_scores += -np.inf
            for m, h in enumerate(gold_heads):
                mst_scores[h, m] = 0.
            gscore, _, gold_traces = parse_mh4t(scores, mst_scores)

            correct = sum([1 for (h, g) in zip(heads[1:], gold_heads[1:]) if h == g])
            traces_set = Counter((i, j, r) for i, j, r in traces)
            gold_traces_set = Counter((i, j, r) for i, j, r in gold_traces)
            loss = [exprs[int(i)][int(j)][int(r)] for i, j, r in (traces_set - gold_traces_set).elements()]
            loss.extend([-exprs[int(i)][int(j)][int(r)] for i, j, r in (gold_traces_set - traces_set).elements()])

            return correct, loss

        if self._mh4t_mode == "hybrid":
            gold_heads = graph.mh4_heads

            sh_scores, re_scores, sh_exprs, re_exprs1, re_exprs2 = self._mh4t_eval(carriers)

            mst_scores = np.ones((len(gold_heads), len(gold_heads)))

            # Cost Augmentation
            for m, h in enumerate(gold_heads):
                mst_scores[h, m] -= 1.

            pscore, heads, traces = parse_mh4t_sh(sh_scores, re_scores, mst_scores)

            mst_scores += -np.inf
            for m, h in enumerate(gold_heads):
                mst_scores[h, m] = 0.
            gscore, _, gold_traces = parse_mh4t_sh(sh_scores, re_scores, mst_scores)

            correct = sum([1 for (h, g) in zip(heads[1:], gold_heads[1:]) if h == g])

            traces_set = Counter((n, i, j, k, r) for n, i, j, k, r in traces)
            gold_traces_set = Counter((n, i, j, k, r) for n, i, j, k, r in gold_traces)
            loss = []
            for n, i, j, k, r in (traces_set - gold_traces_set).elements():
                if n == 2:
                    loss.append(sh_exprs[int(i)][int(j)])
                elif n == 3:
                    loss.append(re_exprs1[int(i)][int(j)][int(r)] + re_exprs2[int(j)][int(k)][int(r)])

            for n, i, j, k, r in (gold_traces_set - traces_set).elements():
                if n == 2:
                    loss.append(-sh_exprs[int(i)][int(j)])
                elif n == 3:
                    loss.append(-re_exprs1[int(i)][int(j)][int(r)] - re_exprs2[int(j)][int(k)][int(r)])

            return correct, loss

    def predict(self, graph, carriers):
        if self._mh4t_mode != "local" and len(graph.nodes) >= 100:
            graph.heads = -np.ones(len(graph.nodes), dtype=int)
            return self

        if self._mh4t_mode == "local":
            beamconf = MH4BeamConf(len(graph.nodes), 1, self._mh4t_stack_features, self._mh4t_buffer_features)
            beamconf.init_conf(0)

            while not beamconf.is_complete(0):
                valid = beamconf.valid_transitions(0)
                scores, exprs = self._mh4t_conf_eval(beamconf.extract_features(0), carriers)
                action, _ = max(((i, s) for i, s in enumerate(scores) if valid[i]), key=lambda x: x[1])
                beamconf.make_transition(0, action)

            graph.heads = list(beamconf.get_heads(0))

            return self

        elif self._mh4t_mode == "two" or self._mh4t_mode == "b0":
            mst_scores = np.ones((len(carriers), len(carriers)))
            scores, exprs = self._mh4t_eval(carriers)
            _, graph.heads, _ = parse_mh4t(scores, mst_scores)

            return self

        elif self._mh4t_mode == "hybrid":
            mst_scores = np.ones((len(carriers), len(carriers)))
            sh_scores, re_scores, sh_exprs, re_exprs1, re_exprs2 = self._mh4t_eval(carriers)
            _, graph.heads, _ = parse_mh4t_sh(sh_scores, re_scores, mst_scores)

            return self


class AHDPParser:

    def __init__(self, parser, id="AHDPParser", **kwargs):
        self._parser = parser
        self.id = id

        self._activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}

        self._bilstm_dims = kwargs.get("bilstm_dims", 128)
        self._ah_mlp_activation = self._activations[kwargs.get('ah_mlp_activation', 'relu')]
        self._ah_mlp_dims = kwargs.get("ah_mlp_dims", 128)
        self._ah_mlp_layers = kwargs.get("ah_mlp_layers", 2)
        self._ah_mlp_dropout = kwargs.get("ah_mlp_dropout", 0.0)

        self._ah_global = kwargs.get("ah_global", True)

    def init_params(self):
        self._ah_pad_repr = [self._parser._model.add_parameters(self._bilstm_dims) for i in range(2)]
        self._ah_stack_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._ah_mlp_dims] * self._ah_mlp_layers, self._ah_mlp_activation, self._parser._model)
        self._ah_buffer_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._ah_mlp_dims] * self._ah_mlp_layers, self._ah_mlp_activation, self._parser._model)
        self._ah_scorer = BiaffineBatch(self._ah_mlp_dims, 3, self._parser._model)

    def init_cg(self, train=False):
        if train:
            self._ah_stack_mlp.set_dropout(self._ah_mlp_dropout)
            self._ah_buffer_mlp.set_dropout(self._ah_mlp_dropout)
        else:
            self._ah_stack_mlp.set_dropout(0.)
            self._ah_buffer_mlp.set_dropout(0.)

    def _ah_confs_eval(self, carriers):
        rows = list(range(-1, len(carriers))) + [-2]
        vecs = [carriers[f].vec if f >= 0 else parameter(self._ah_pad_repr[f]) for i, f in enumerate(rows)]
        vecs = concatenate(vecs, 1)
        exprs = self._ah_scorer(self._ah_stack_mlp(vecs), self._ah_buffer_mlp(vecs))
        scores = exprs.value()
        exprs = np.array([[[exprs[i][j][b] for b in (0, 1, 2)] for j in range(len(rows))] for i in range(len(rows))])

        return scores, exprs

    def _ah_seq_loss(self, correctseq, wrongseq, beamconf, loss, carriers, exprs, loc=0):
        commonprefix = 0
        for i in range(min(len(correctseq), len(wrongseq))):
            if wrongseq[i] == correctseq[i]:
                commonprefix = i + 1
            else:
                break

        beamconf.init_conf(loc, True)
        for i in range(commonprefix):
            beamconf.make_transition(loc, wrongseq[i])
        for i in range(commonprefix, len(wrongseq)):
            s, b = beamconf.extract_features(loc)
            if b < 0:
                b = len(carriers)
            loss.append(exprs[int(s+1)][int(b+1)][int(wrongseq[i])])
            beamconf.make_transition(loc, wrongseq[i])
        beamconf.init_conf(loc, True)
        for i in range(commonprefix):
            beamconf.make_transition(loc, correctseq[i])
        for i in range(commonprefix, len(correctseq)):
            s, b = beamconf.extract_features(loc)
            if b < 0:
                b = len(carriers)
            loss.append(-exprs[int(s+1)][int(b+1)][int(correctseq[i])])
            beamconf.make_transition(loc, correctseq[i])

    def sent_loss(self, graph, carriers):
        gold_heads = graph.proj_heads

        loss = []
        beamconf = AHBeamConf(len(graph.nodes), 1, np.array(gold_heads), 1, 1)
        beamconf.init_conf(0, True)

        scores, exprs = self._ah_confs_eval(carriers)

        if self._ah_global:
            # cost augmentation
            mst_scores = np.ones((len(graph.nodes), len(graph.nodes)))
            for m, h in enumerate(gold_heads):
                mst_scores[h, m] -= 1.
            pred_transitions, pred_heads = parse_ah_dp_mst(scores, mst_scores)
            true_transitions = beamconf.gold_transitions(0, True)

            self._ah_seq_loss(true_transitions, pred_transitions, beamconf, loss, carriers, exprs, loc=0)

            correct = sum([1 if gold_heads[i] == pred_heads[i] else 0 for i in range(1, len(graph.nodes))])
        else:
            true_transitions = beamconf.gold_transitions(0, True)

            beamconf.init_conf(0, True)
            beamconf.make_transition(0, 0)

            correct = 0

            for step in range(1, len(true_transitions)):
                s, b = beamconf.extract_features(0)
                if b < 0:
                    b = len(carriers)

                g = true_transitions[step]

                valid_transitions = beamconf.valid_transitions(0)
                valid_set = [i for i in range(3) if valid_transitions[i]]
                valid_scores = [scores[s+1, b+1, i] if i == g else scores[s+1, b+1, i] + 1. for i in valid_set]

                m = valid_set[np.argmax(valid_scores)]

                if m == true_transitions[step]:
                    correct += 1
                else:
                    loss.append(exprs[s+1, b+1, m] + 1. - exprs[s+1, b+1, g])

                beamconf.make_transition(0, true_transitions[step])

        return correct, loss

    def predict(self, graph, carriers):
        beamconf = AHBeamConf(len(graph.nodes), 1, np.array(graph.heads), 1, 1)
        beamconf.init_conf(0, True)

        scores, exprs = self._ah_confs_eval(carriers)

        if self._ah_global:
            mst_scores = np.zeros((len(graph.nodes), len(graph.nodes)))
            transitions, heads = parse_ah_dp_mst(scores, mst_scores)
            graph.heads = heads
        else:
            while not beamconf.is_complete(0):
                valid_transitions = beamconf.valid_transitions(0)
                valid_set = [i for i in range(3) if valid_transitions[i]]

                if len(valid_set) == 0:
                    break

                s, b = beamconf.extract_features(0)
                if b < 0:
                    b = len(carriers)

                valid_scores = [scores[s+1, b+1, i] for i in valid_set]
                beamconf.make_transition(0, valid_set[np.argmax(valid_scores)])
            graph.heads = beamconf.get_heads(0)

        return self


class AEDPParser:

    def __init__(self, parser, id="AEDPParser", **kwargs):
        self._parser = parser
        self.id = id

        self._activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}

        self._bilstm_dims = kwargs.get("bilstm_dims", 128)
        self._ae_mlp_activation = self._activations[kwargs.get('ae_mlp_activation', 'relu')]
        self._ae_mlp_dims = kwargs.get("ae_mlp_dims", 128)
        self._ae_mlp_layers = kwargs.get("ae_mlp_layers", 2)
        self._ae_mlp_dropout = kwargs.get("ae_mlp_dropout", 0.0)

        self._ae_global = kwargs.get("ae_global", True)

    def init_params(self):
        self._ae_pad_repr = [self._parser._model.add_parameters(self._bilstm_dims) for i in range(2)]
        self._ae_stack_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._ae_mlp_dims] * self._ae_mlp_layers, self._ae_mlp_activation, self._parser._model)
        self._ae_buffer_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._ae_mlp_dims] * self._ae_mlp_layers, self._ae_mlp_activation, self._parser._model)
        self._ae_scorer = BiaffineBatch(self._ae_mlp_dims, 4, self._parser._model)

    def init_cg(self, train=False):
        if train:
            self._ae_stack_mlp.set_dropout(self._ae_mlp_dropout)
            self._ae_buffer_mlp.set_dropout(self._ae_mlp_dropout)
        else:
            self._ae_stack_mlp.set_dropout(0.)
            self._ae_buffer_mlp.set_dropout(0.)

    def _ae_confs_eval(self, carriers):
        rows = list(range(-1, len(carriers))) + [-2]

        vecs = [carriers[f].vec if f >= 0 else parameter(self._ae_pad_repr[f]) for i, f in enumerate(rows)]
        vecs = concatenate(vecs, 1)

        exprs = self._ae_scorer(self._ae_stack_mlp(vecs), self._ae_buffer_mlp(vecs))
        scores = exprs.value()
        exprs = np.array([[[exprs[i][j][b] for b in (0, 1, 2, 3)] for j in range(len(rows))] for i in range(len(rows))])

        return scores, exprs

    def _ae_seq_loss(self, correctseq, wrongseq, beamconf, loss, carriers, exprs, loc=0):
        commonprefix = 0
        for i in range(min(len(correctseq), len(wrongseq))):
            if wrongseq[i] == correctseq[i]:
                commonprefix = i + 1
            else:
                break

        beamconf.init_conf(loc, True)
        for i in range(commonprefix):
            beamconf.make_transition(loc, wrongseq[i])
        for i in range(commonprefix, len(wrongseq)):
            s, b = beamconf.extract_features(loc)
            if b < 0:
                b = len(carriers)
            loss.append(exprs[int(s+1)][int(b+1)][int(wrongseq[i])])
            beamconf.make_transition(loc, wrongseq[i])
        beamconf.init_conf(loc, True)
        for i in range(commonprefix):
            beamconf.make_transition(loc, correctseq[i])
        for i in range(commonprefix, len(correctseq)):
            s, b = beamconf.extract_features(loc)
            if b < 0:
                b = len(carriers)
            loss.append(-exprs[int(s+1)][int(b+1)][int(correctseq[i])])
            beamconf.make_transition(loc, correctseq[i])

    def sent_loss(self, graph, carriers):
        gold_heads = graph.proj_heads

        loss = []
        beamconf = AEBeamConf(len(graph.nodes), 1, np.array(gold_heads), 1, 1)
        beamconf.init_conf(0, True)

        scores, exprs = self._ae_confs_eval(carriers)

        if self._ae_global:
            # cost augmentation
            mst_scores = np.ones((len(graph.nodes), len(graph.nodes)))
            for m, h in enumerate(gold_heads):
                mst_scores[h, m] -= 1.
            pred_transitions, pred_heads = parse_ae_dp_mst(scores, mst_scores)
            true_transitions = beamconf.gold_transitions(0, True)

            self._ae_seq_loss(true_transitions, pred_transitions, beamconf, loss, carriers, exprs, loc=0)

            correct = sum([1 if gold_heads[i] == pred_heads[i] else 0 for i in range(1, len(graph.nodes))])

        else:
            true_transitions = beamconf.gold_transitions(0, True)

            beamconf.init_conf(0, True)
            beamconf.make_transition(0, 0)

            correct = 0

            for step in range(1, len(true_transitions)):
                s, b = beamconf.extract_features(0)
                if b < 0:
                    b = len(carriers)

                g = true_transitions[step]

                valid_transitions = beamconf.valid_transitions(0)
                valid_set = [i for i in range(4) if valid_transitions[i]]
                valid_scores = [scores[s+1, b+1, i] if i == g else scores[s+1, b+1, i] + 1. for i in valid_set]

                m = valid_set[np.argmax(valid_scores)]

                if m == true_transitions[step]:
                    correct += 1
                else:
                    loss.append(exprs[s+1, b+1, m] + 1. - exprs[s+1, b+1, g])

                beamconf.make_transition(0, true_transitions[step])

            correct /= len(true_transitions) - 1
            correct *= len(graph.nodes) - 1

        return correct, loss

    def predict(self, graph, carriers):
        beamconf = AEBeamConf(len(graph.nodes), 1, np.array(graph.heads), 1, 1)
        beamconf.init_conf(0, True)

        scores, exprs = self._ae_confs_eval(carriers)
        if self._ae_global:
            mst_scores = np.zeros((len(graph.nodes), len(graph.nodes)))
            transitions, heads = parse_ae_dp_mst(scores, mst_scores)
        else:
            while not beamconf.is_complete(0):
                valid_transitions = beamconf.valid_transitions(0)
                valid_set = [i for i in range(4) if valid_transitions[i]]

                if len(valid_set) == 0:
                    break

                s, b = beamconf.extract_features(0)
                if b < 0:
                    b = len(carriers)

                valid_scores = [scores[s+1, b+1, i] for i in valid_set]
                beamconf.make_transition(0, valid_set[np.argmax(valid_scores)])
            heads = beamconf.get_heads(0)

        graph.heads = heads

        return self


class OneECParser:
    def __init__(self, parser, id="OneECParser", **kwargs):
        self._parser = parser
        self.id = id

        self._activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}

        self._bilstm_dims = kwargs.get("bilstm_dims", 128)
        self._oneec_mlp_activation = self._activations[kwargs.get('oneec_mlp_activation', 'relu')]
        self._oneec_mlp_dims = kwargs.get("oneec_mlp_dims", 128)
        self._oneec_mlp_layers = kwargs.get("oneec_mlp_layers", 2)
        self._oneec_mlp_dropout = kwargs.get("oneec_mlp_dropout", 0.0)

    def init_params(self):
        self._oneec_grand_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._oneec_mlp_dims] * self._oneec_mlp_layers, self._oneec_mlp_activation, self._parser._model)
        self._oneec_head_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._oneec_mlp_dims] * self._oneec_mlp_layers, self._oneec_mlp_activation, self._parser._model)
        self._oneec_mod_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._oneec_mlp_dims] * self._oneec_mlp_layers, self._oneec_mlp_activation, self._parser._model)
        self._oneec_sib_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._oneec_mlp_dims] * self._oneec_mlp_layers, self._oneec_mlp_activation, self._parser._model)
        self._oneec_grand_mod_biaffine = BiaffineBatch(self._oneec_mlp_dims, 1, self._parser._model)
        self._oneec_head_mod_biaffine = BiaffineBatch(self._oneec_mlp_dims, 1, self._parser._model)
        self._oneec_sib_mod_biaffine = BiaffineBatch(self._oneec_mlp_dims, 1, self._parser._model)
        self._oneec_non_proj_head_mod_biaffine = BiaffineBatch(self._oneec_mlp_dims, 1, self._parser._model)
        self._oneec_grand_pad = self._parser._model.add_parameters(self._oneec_mlp_dims)
        self._oneec_head_pad = self._parser._model.add_parameters(self._oneec_mlp_dims)
        self._oneec_mod_pad = self._parser._model.add_parameters(self._oneec_mlp_dims)
        self._oneec_sib_pad = self._parser._model.add_parameters(self._oneec_mlp_dims)

    def init_cg(self, train=False):
        if train:
            self._oneec_grand_mlp.set_dropout(self._oneec_mlp_dropout)
            self._oneec_head_mlp.set_dropout(self._oneec_mlp_dropout)
            self._oneec_mod_mlp.set_dropout(self._oneec_mlp_dropout)
            self._oneec_sib_mlp.set_dropout(self._oneec_mlp_dropout)
        else:
            self._oneec_grand_mlp.set_dropout(0.)
            self._oneec_head_mlp.set_dropout(0.)
            self._oneec_mod_mlp.set_dropout(0.)
            self._oneec_sib_mlp.set_dropout(0.)

    def _oneec_arcs_eval(self, carriers):
        grand_vecs = [parameter(self._oneec_grand_pad)] + [self._oneec_grand_mlp(c.vec) for c in carriers]
        head_vecs = [parameter(self._oneec_head_pad)] + [self._oneec_head_mlp(c.vec) for c in carriers]
        mod_vecs = [parameter(self._oneec_mod_pad)] + [self._oneec_mod_mlp(c.vec) for c in carriers]
        sib_vecs = [parameter(self._oneec_sib_pad)] + [self._oneec_sib_mlp(c.vec) for c in carriers]

        grand_vecs = concatenate(grand_vecs, 1)
        head_vecs = concatenate(head_vecs, 1)
        mod_vecs = concatenate(mod_vecs, 1)
        sib_vecs = concatenate(sib_vecs, 1)

        grand_mod = self._oneec_grand_mod_biaffine(grand_vecs, mod_vecs)
        head_mod = self._oneec_head_mod_biaffine(head_vecs, mod_vecs)
        sib_mod = self._oneec_sib_mod_biaffine(sib_vecs, mod_vecs)
        nonproj_mod = self._oneec_non_proj_head_mod_biaffine(head_vecs, mod_vecs)

        exprs = concatenate([grand_mod, head_mod, sib_mod, nonproj_mod], 2)

        scores = exprs.value()

        return scores, exprs

    def sent_loss(self, graph, carriers):
        gold_heads, gold_traces = graph.oneec_heads, graph.oneec_traces
        scores, exprs = self._oneec_arcs_eval(carriers)

        # Cost Augmentation
        for m, h in enumerate(gold_heads):
            scores[h + 1, m + 1, 1] -= 1.
            scores[h + 1, m + 1, 3] -= 1.

        heads, traces = parse_1ec_o3(scores)
        correct = sum([1 for (h, g) in zip(heads[1:], gold_heads[1:]) if h == g])

        pred_traces = {(int(i), int(j), int(r)) for i, j, r in traces if r >= 0}

        loss = [exprs[i][j][r] for i, j, r in pred_traces - gold_traces]
        loss.extend([-exprs[i][j][r] for i, j, r in gold_traces - pred_traces])

        return correct, loss

    def predict(self, graph, carriers):
        if len(graph.nodes) >= 100:
            graph.heads = -np.ones(len(graph.nodes), dtype=int)
            return self
        scores, exprs = self._oneec_arcs_eval(carriers)
        graph.heads, _ = parse_1ec_o3(scores)

        return self
