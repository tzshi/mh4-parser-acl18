#!/usr/bin/env python
# encoding: utf-8


from dynet import parameter, transpose, dropout, rectify
from dynet import layer_norm, affine_transform
from dynet import concatenate, zeroes, dot_product
from dynet import GlorotInitializer, ConstInitializer, SaxeInitializer
import math


class Dense(object):
    def __init__(self, indim, outdim, activation, model, ln=False):
        self.model = model
        self.activation = activation
        self.ln = ln
        if activation == rectify:
            self.W = model.add_parameters((outdim, indim), init=GlorotInitializer(gain=math.sqrt(2.)))
        else:
            self.W = model.add_parameters((outdim, indim))
        self.b = model.add_parameters(outdim, init=ConstInitializer(0.))
        if ln:
            self.ln_s = model.add_parameters(outdim, ConstInitializer(1.))
        self.spec = (indim, outdim, activation, ln)

    def __call__(self, x):
        if self.ln:
            return self.activation(layer_norm(parameter(self.W) * x, parameter(self.ln_s), parameter(self.b)))
        else:
            return self.activation(affine_transform([parameter(self.b), parameter(self.W), x]))

    def param_collection(self):
        return self.model

    @staticmethod
    def from_spec(spec, model):
        (indim, outdim, activation, ln) = spec
        return Dense(indim, outdim, activation, model, ln)


class MultiLayerPerceptron(object):
    def __init__(self, dims, activation, model, ln=False):
        self.model = model
        self.layers = []
        self.dropout = 0.
        self.outdim = []
        for indim, outdim in zip(dims, dims[1:]):
            self.layers.append(Dense(indim, outdim, activation, model, ln))
            self.outdim.append(outdim)
        self.spec = (indim, outdim, activation, ln)

    def __call__(self, x):
        for layer, dim in zip(self.layers, self.outdim):
            x = layer(x)
            if self.dropout > 0.:
                x = dropout(x, self.dropout)
        return x

    def set_dropout(self, droprate):
        self.dropout = droprate

    def param_collection(self):
        return self.model

    @staticmethod
    def from_spec(spec, model):
        (indim, outdim, activation, ln) = spec
        return MultiLayerPerceptron(indim, outdim, activation, model, ln)


class Bilinear(object):
    def __init__(self, dim, model):
        self.U = model.add_parameters((dim, dim), init=SaxeInitializer())

    def __call__(self, x, y):
        U = parameter(self.U)
        return transpose(x) * U * y

    def get_components(self):
        return [self.U]

    def restore_components(self, components):
        [self.U] = components


class Biaffine(object):
    def __init__(self, indim, model):
        self.model = model
        self.U = Bilinear(indim, model)
        self.x_bias = model.add_parameters((indim))
        self.y_bias = model.add_parameters((indim))
        self.bias = model.add_parameters(1)
        self.spec = (indim,)

    def __call__(self, x, y):
        x_bias = parameter(self.x_bias)
        y_bias = parameter(self.y_bias)
        bias = parameter(self.bias)

        return bias + dot_product(x_bias, x) + dot_product(y_bias, y) + self.U(x, y)

    def param_collection(self):
        return self.model

    @staticmethod
    def from_spec(spec, model):
        (indim, ) = spec
        return Biaffine(indim, model)


class BiaffineBatch(object):
    def __init__(self, indim, outdim, model):
        self.model = model
        self.U = [Bilinear(indim + 1, model) for i in range(outdim)]
        self.spec = (indim, outdim)

    def __call__(self, x, y):
        x = concatenate([x, zeroes((1, x.dim()[0][1],)) + 1.])
        y = concatenate([y, zeroes((1, y.dim()[0][1],)) + 1.])

        if self.spec[1] == 1:
            return self.U[0](x, y)
        else:
            return concatenate([u(x, y) for u in self.U], 2)

    def get_components(self):
        return self.U

    def restore_components(self, components):
        self.U = components[:-3]

    def param_collection(self):
        return self.model

    @staticmethod
    def from_spec(spec, model):
        (indim, outdim) = spec
        return BiaffineBatch(indim, outdim, model)


def identity(x):
    return x
