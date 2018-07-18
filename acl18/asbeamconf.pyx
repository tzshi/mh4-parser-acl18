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
cdef int SHIFT = 0, LEFTARC = 1, RIGHTARC = 2


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class ASBeamConf:

    cdef int size
    cdef int length
    cdef int *step
    cdef int *transitions
    cdef int *heads
    cdef int *stack
    cdef int *buffer
    cdef int *stack_head
    cdef int *buffer_head
    cdef int *location
    cdef float *transition_score
    cdef int *cost
    cdef int *gold_heads
    cdef int *root_first
    cdef int stack_features
    cdef int buffer_features

    def __cinit__(self, int length, int size, np.ndarray[ndim=1, dtype=np.npy_intp] gold_heads, int stack_features, int buffer_features):
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
        self.transition_score = <float *>malloc(size*sizeof(float))
        self.cost = <int *>malloc(size*sizeof(int))
        self.gold_heads = <int *>malloc(length*sizeof(int))
        self.root_first = <int *>malloc(size*sizeof(int))

        cdef int i
        for i in range(length):
            self.gold_heads[i] = gold_heads[i]

    def __dealloc__(self):
        free(self.step)
        free(self.transitions)
        free(self.heads)
        free(self.stack)
        free(self.buffer)
        free(self.stack_head)
        free(self.buffer_head)
        free(self.location)
        free(self.transition_score)
        free(self.cost)
        free(self.gold_heads)
        free(self.root_first)

    cpdef void copy(self, int source, int target):
        cdef int i
        cdef int length = self.length
        self.step[target] = self.step[source]
        self.root_first[target] = self.root_first[source]
        self.stack_head[target] = self.stack_head[source]
        self.buffer_head[target] = self.buffer_head[source]
        self.transition_score[target] = self.transition_score[source]
        self.cost[target] = self.cost[source]
        for i in range(self.length):
            self.heads[target*length+i] = self.heads[source*length+i]
            self.stack[target*length+i] = self.stack[source*length+i]
            self.buffer[target*length+i] = self.buffer[source*length+i]
            self.location[target*length+i] = self.location[source*length+i]
        for i in range(self.length*2):
            self.transitions[target*length*2+i] = self.transitions[source*length*2+i]

    cpdef void init_conf(self, int pos, int rootfirst=0):
        cdef int i
        cdef int length = self.length
        self.step[pos] = 0
        self.stack_head[pos] = 0
        self.buffer_head[pos] = length
        self.transition_score[pos] = 0.0
        self.cost[pos] = 0
        self.root_first[pos] = rootfirst
        for i in range(length):
            self.heads[pos*length+i] = -1
            if rootfirst:
                self.buffer[pos*length+i] = length - i - 1
            else:
                self.buffer[pos*length+i] = length - i
            self.location[pos*length+i] = BUFFER_LOC
        if not rootfirst:
            self.buffer[pos*length] = 0

    cpdef np.ndarray[ndim=1, dtype=np.npy_intp] gold_transitions(self, int pos, int rootfirst=0):
        cdef np.ndarray[ndim=1, dtype=np.npy_intp] transitions
        cdef np.ndarray[ndim=1, dtype=np.npy_intp] costs
        cdef int i, j

        transitions = np.full(2 * self.length - 1, 0, dtype=int)

        self.init_conf(pos, rootfirst)
        for i in range(0, 2 * self.length - 1):
            costs = self.transition_costs(pos)
            for j in range(0, 3):
                if costs[j] == 0:
                    transitions[i] = j
                    self.make_transition(pos, j)
                    break
        return transitions

    cpdef int stack_length(self, int pos):
        return self.stack_head[pos]

    def full_info(self, int pos):
        cdef int i
        cdef int offset = self.length * pos
        return (self.step[pos], self.transition_score[pos], self.cost[pos],
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

    cpdef float get_score(self, int pos):
        return self.transition_score[pos]

    cpdef void transition_score_dev(self, int pos, float dev):
        self.transition_score[pos] += dev

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
        ret = np.zeros(3, dtype=int)
        if self.buffer_head[pos] > 0:
            ret[SHIFT] = 1
        if self.stack_head[pos] > 1 and self.stack[offset + self.stack_head[pos] - 1] != 0 and (self.stack[offset + self.stack_head[pos] - 2] != 0 or self.stack_head[pos] == 2):
            ret[RIGHTARC] = 1
        if self.stack_head[pos] > 1 and self.stack[offset + self.stack_head[pos] - 2] != 0 and (self.stack[offset + self.stack_head[pos] - 1] != 0 or self.stack_head[pos] == 2):
            ret[LEFTARC] = 1
        return ret

    cpdef void make_transition(self, int pos, int transition):
        if transition == SHIFT:
            self.shift(pos)
        elif transition == LEFTARC:
            self.leftarc(pos)
        elif transition == RIGHTARC:
            self.rightarc(pos)

    cpdef void shift(self, int pos):
        cdef int offset = self.length * pos
        cdef int b0 = self.buffer[offset+self.buffer_head[pos]-1]
        self.location[offset+b0] = STACK_LOC
        self.stack[offset+self.stack_head[pos]] = b0
        self.buffer_head[pos] -= 1
        self.stack_head[pos] += 1
        self.transitions[offset*2+self.step[pos]] = SHIFT
        self.step[pos] += 1

    cpdef void leftarc(self, int pos):
        cdef int offset = self.length * pos
        cdef int s0 = self.stack[offset+self.stack_head[pos]-1]
        cdef int s1 = self.stack[offset+self.stack_head[pos]-2]
        self.location[offset+s1] = RESOLVED
        self.heads[offset+s1] = s0
        self.stack_head[pos] -= 1
        self.stack[offset+self.stack_head[pos] - 1] = s0
        self.transitions[offset*2+self.step[pos]] = LEFTARC
        self.step[pos] += 1

    cpdef void rightarc(self, int pos):
        cdef int offset = self.length * pos
        cdef int s0 = self.stack[offset+self.stack_head[pos]-1]
        cdef int s1 = self.stack[offset+self.stack_head[pos]-2]
        self.location[offset+s0] = RESOLVED
        self.heads[offset+s0] = s1
        self.stack_head[pos] -= 1
        self.transitions[offset*2+self.step[pos]] = RIGHTARC
        self.step[pos] += 1

    cpdef np.ndarray[ndim=1, dtype=np.npy_intp] transition_costs(self, int pos):
        cdef int offset = self.length * pos
        cdef np.ndarray[ndim=1, dtype=np.npy_intp] ret
        cdef np.ndarray[ndim=1, dtype=np.npy_intp] valid = self.valid_transitions(pos)
        cdef int cost, i, parent
        cdef int s0=-1, b0=-1
        if self.stack_head[pos] > 0:
            s0 = self.stack[offset+self.stack_head[pos]-1]
        if self.buffer_head[pos] > 0:
            b0 = self.buffer[offset+self.buffer_head[pos]-1]
        ret = np.full(3, -1, dtype=int)

        if valid[SHIFT]:
            cost = 0
            ret[SHIFT] = cost

        if valid[LEFTARC]:
            cost = 0
            ret[LEFTARC] = cost

        if valid[RIGHTARC]:
            cost = 0
            ret[RIGHTARC] = cost

        return ret

    cpdef int static_oracle(self, int pos):
        cdef int offset = self.length * pos
        cdef int s0, s1, i
        cdef int noOtherChild = 1

        if self.stack_head[pos] > 1:
            s0 = self.stack[offset+self.stack_head[pos]-1]
            s1 = self.stack[offset+self.stack_head[pos]-2]
            if self.gold_heads[s1] == s0:
                return LEFTARC
            if self.gold_heads[s0] == s1:
                for i in range(self.length):
                    if self.gold_heads[i] == s0 and self.heads[offset+i] != s0:
                        noOtherChild = 0
                if noOtherChild:
                    return RIGHTARC
        return SHIFT
