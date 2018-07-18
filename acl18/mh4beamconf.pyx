#!/usr/bin/env python
# encoding: utf-8

cimport cython
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free

np.import_array()

cdef np.float64_t NEGINF = -np.inf
cdef np.float64_t INF = np.inf
cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline np.float64_t float64_max(np.float64_t a, np.float64_t b): return a if a >= b else b
cdef inline np.float64_t float64_min(np.float64_t a, np.float64_t b): return a if a <= b else b

cdef int STACK_LOC = 0, BUFFER_LOC = 1, RESOLVED = 2
cdef int SHIFT = 0, S2S1 = 1, S0S1 = 2, B0S1 = 3, S2S0 = 4, S1S0 = 5, B0S0 = 6


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class MH4BeamConf:

    cdef int size
    cpdef int length
    cdef int *step
    cdef int *transitions
    cdef int *heads
    cdef int *stack
    cdef int *buffer
    cdef int *stack_head
    cdef int *buffer_head
    cdef int *location
    cdef int stack_features
    cdef int buffer_features

    def __cinit__(self, int length, int size, int stack_features, int buffer_features):
        self.size = size
        self.length = length
        self.stack_features = stack_features
        self.buffer_features = buffer_features
        self.step = <int *>malloc(size*sizeof(int))
        self.transitions = <int *>malloc(size*length*2*sizeof(int))
        self.heads = <int *>malloc(size*length*sizeof(int))
        self.stack = <int *>malloc(size*length*sizeof(int))
        self.buffer = <int *>malloc(size*length*sizeof(int))
        self.stack_head = <int *>malloc(size*sizeof(int))
        self.buffer_head = <int *>malloc(size*sizeof(int))
        self.location = <int *>malloc(size*length*sizeof(int))

    def __dealloc__(self):
        free(self.step)
        free(self.transitions)
        free(self.heads)
        free(self.stack)
        free(self.buffer)
        free(self.stack_head)
        free(self.buffer_head)
        free(self.location)

    cpdef void copy(self, int source, int target):
        cdef int i
        cdef int length = self.length
        self.step[target] = self.step[source]
        self.stack_head[target] = self.stack_head[source]
        self.buffer_head[target] = self.buffer_head[source]
        for i in range(self.length):
            self.heads[target*length+i] = self.heads[source*length+i]
            self.stack[target*length+i] = self.stack[source*length+i]
            self.buffer[target*length+i] = self.buffer[source*length+i]
            self.location[target*length+i] = self.location[source*length+i]
        for i in range(self.length*2):
            self.transitions[target*length*2+i] = self.transitions[source*length*2+i]

    cpdef void init_conf(self, int pos):
        cdef int i
        cdef int length = self.length
        self.step[pos] = 0
        self.stack_head[pos] = 1
        self.buffer_head[pos] = length - 1
        for i in range(length):
            self.heads[pos*length+i] = -1
            self.location[pos*length+i] = BUFFER_LOC
        self.location[0] = STACK_LOC

        for i in range(length - 1):
            self.buffer[pos*length+i] = length - i - 1
        self.stack[pos*length] = 0

    cpdef int stack_length(self, int pos):
        return self.stack_head[pos]

    def full_info(self, int pos):
        cdef int i
        cdef int offset = self.length * pos
        return (self.step[pos],
                [self.heads[offset+i] for i in range(self.length)],
                [self.stack[offset+i] for i in range(self.stack_head[pos])],
                [self.buffer[offset+i] for i in range(self.buffer_head[pos])])

    def get_transitions(self, int pos):
        cdef int offset = self.length * 2 * pos
        cdef int i
        cdef int step = self.step[pos]
        cdef np.ndarray[ndim=1, dtype=np.npy_intp] ret
        ret = np.empty(step, dtype=int)
        for i in range(step):
            ret[i] = self.transitions[offset + i]
        return ret

    def get_heads(self, int pos):
        cdef int offset = self.length * pos
        cdef int i
        cdef np.ndarray[ndim=1, dtype=np.npy_intp] ret
        ret = np.empty(self.length, dtype=int)
        for i in range(self.length):
            ret[i] = self.heads[offset + i]
        return ret

    def extract_features(self, int pos):
        cdef np.ndarray[ndim=1, dtype=np.npy_intp] ret
        cdef int total, i
        cdef int offset = self.length * pos
        cdef int nostack = self.stack_features, nobuffer = self.buffer_features

        total = nostack + nobuffer

        ret = np.full(total, -1, dtype=int)
        for i in range(nostack):
            if self.stack_head[pos] > i:
                ret[i] = self.stack[offset+self.stack_head[pos]-i-1]
        for i in range(nobuffer):
            if self.buffer_head[pos] > i:
                ret[nostack+i] = self.buffer[offset+self.buffer_head[pos]-i-1]

        return ret

    def is_complete(self, int pos):
        if self.buffer_head[pos] == 0 and self.stack_head[pos] == 1:
            return True
        else:
            return False

    cpdef np.ndarray[ndim=1, dtype=np.npy_intp] valid_transitions(self, int pos):
        cdef int offset = self.length * pos
        cdef np.ndarray[ndim=1, dtype=np.npy_intp] ret
        ret = np.zeros(7, dtype=int)

        if self.buffer_head[pos] > 0:
            ret[SHIFT] = 1
        if self.stack_head[pos] > 2 and self.stack[offset+self.stack_head[pos]-2] != 0:
            ret[S2S1] = 1
        if self.stack_head[pos] > 1 and self.stack[offset+self.stack_head[pos]-2] != 0:
            ret[S0S1] = 1
        if self.stack_head[pos] > 1 and self.stack[offset+self.stack_head[pos]-2] != 0 and self.buffer_head[pos] > 0:
            ret[B0S1] = 1
        if self.stack_head[pos] > 2 and self.stack[offset+self.stack_head[pos]-1] != 0:
            ret[S2S0] = 1
        if self.stack_head[pos] > 1 and self.stack[offset+self.stack_head[pos]-1] != 0:
            ret[S1S0] = 1
        if self.stack_head[pos] > 0 and self.stack[offset+self.stack_head[pos]-1] != 0 and self.buffer_head[pos] > 0:
            ret[B0S0] = 1

        return ret

    cpdef void make_transition(self, int pos, int transition):
        if transition == SHIFT:
            self.shift(pos)
        elif transition == S2S1:
            self.s2s1(pos)
        elif transition == S0S1:
            self.s0s1(pos)
        elif transition == B0S1:
            self.b0s1(pos)
        elif transition == S2S0:
            self.s2s0(pos)
        elif transition == S1S0:
            self.s1s0(pos)
        elif transition == B0S0:
            self.b0s0(pos)

    cpdef void shift(self, int pos):
        cdef int offset = self.length * pos
        cdef int b0 = self.buffer[offset+self.buffer_head[pos]-1]
        self.location[offset+b0] = STACK_LOC
        self.stack[offset+self.stack_head[pos]] = b0
        self.buffer_head[pos] -= 1
        self.stack_head[pos] += 1
        self.transitions[offset*2+self.step[pos]] = SHIFT
        self.step[pos] += 1

    cpdef void s2s1(self, int pos):
        cdef int offset = self.length * pos
        cdef int s0 = self.stack[offset+self.stack_head[pos]-1]
        cdef int s1 = self.stack[offset+self.stack_head[pos]-2]
        cdef int s2 = self.stack[offset+self.stack_head[pos]-3]
        self.location[offset+s1] = RESOLVED
        self.heads[offset+s1] = s2
        self.stack[offset+self.stack_head[pos]-2] = s0
        self.stack_head[pos] -= 1
        self.transitions[offset*2+self.step[pos]] = S2S1
        self.step[pos] += 1

    cpdef void s0s1(self, int pos):
        cdef int offset = self.length * pos
        cdef int s0 = self.stack[offset+self.stack_head[pos]-1]
        cdef int s1 = self.stack[offset+self.stack_head[pos]-2]
        self.location[offset+s1] = RESOLVED
        self.heads[offset+s1] = s0
        self.stack[offset+self.stack_head[pos]-2] = s0
        self.stack_head[pos] -= 1
        self.transitions[offset*2+self.step[pos]] = S0S1
        self.step[pos] += 1

    cpdef void b0s1(self, int pos):
        cdef int offset = self.length * pos
        cdef int s0 = self.stack[offset+self.stack_head[pos]-1]
        cdef int s1 = self.stack[offset+self.stack_head[pos]-2]
        cdef int b0 = self.buffer[offset+self.buffer_head[pos]-1]
        self.location[offset+s1] = RESOLVED
        self.heads[offset+s1] = b0
        self.stack[offset+self.stack_head[pos]-2] = s0
        self.stack_head[pos] -= 1
        self.transitions[offset*2+self.step[pos]] = B0S1
        self.step[pos] += 1

    cpdef void s2s0(self, int pos):
        cdef int offset = self.length * pos
        cdef int s0 = self.stack[offset+self.stack_head[pos]-1]
        cdef int s2 = self.stack[offset+self.stack_head[pos]-3]
        self.location[offset+s0] = RESOLVED
        self.heads[offset+s0] = s2
        self.stack_head[pos] -= 1
        self.transitions[offset*2+self.step[pos]] = S2S0
        self.step[pos] += 1

    cpdef void s1s0(self, int pos):
        cdef int offset = self.length * pos
        cdef int s0 = self.stack[offset+self.stack_head[pos]-1]
        cdef int s1 = self.stack[offset+self.stack_head[pos]-2]
        self.location[offset+s0] = RESOLVED
        self.heads[offset+s0] = s1
        self.stack_head[pos] -= 1
        self.transitions[offset*2+self.step[pos]] = S1S0
        self.step[pos] += 1

    cpdef void b0s0(self, int pos):
        cdef int offset = self.length * pos
        cdef int s0 = self.stack[offset+self.stack_head[pos]-1]
        cdef int b0 = self.buffer[offset+self.buffer_head[pos]-1]
        self.location[offset+s0] = RESOLVED
        self.heads[offset+s0] = b0
        self.stack_head[pos] -= 1
        self.transitions[offset*2+self.step[pos]] = B0S0
        self.step[pos] += 1
