#!/usr/bin/env python
# encoding: utf-8

cimport cython

import numpy as np
cimport numpy as np
np.import_array()

from cpython cimport bool

cdef np.float64_t NEGINF = -np.inf
cdef np.float64_t INF = np.inf
cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline np.float64_t float64_max(np.float64_t a, np.float64_t b): return a if a >= b else b
cdef inline np.float64_t float64_min(np.float64_t a, np.float64_t b): return a if a <= b else b


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int get_deplen(int head, int mod):
    if head == 0:
        return 0
    if head < mod:
        if mod - head < 10:
            return mod - head
        else:
            return 10
    else:
        if head - mod < 10:
            return head - mod + 10
        else:
            return 20


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.npy_intp, ndim=1] parse_proj(np.ndarray[np.float64_t, ndim=2] scores):
    cdef int nr, nc, N, i, k, s, t, r, maxidx
    cdef np.float64_t tmp, cand
    cdef np.ndarray[np.float64_t, ndim=2] complete_0
    cdef np.ndarray[np.float64_t, ndim=2] complete_1
    cdef np.ndarray[np.float64_t, ndim=2] incomplete_0
    cdef np.ndarray[np.float64_t, ndim=2] incomplete_1
    cdef np.ndarray[np.npy_intp, ndim=3] complete_backtrack
    cdef np.ndarray[np.npy_intp, ndim=3] incomplete_backtrack
    cdef np.ndarray[np.npy_intp, ndim=1] heads

    nr, nc = np.shape(scores)

    N = nr - 1 # Number of words (excluding root).

    complete_0 = np.zeros((nr, nr)) # s, t, direction (right=1).
    complete_1 = np.zeros((nr, nr)) # s, t, direction (right=1).
    incomplete_0 = np.zeros((nr, nr)) # s, t, direction (right=1).
    incomplete_1 = np.zeros((nr, nr)) # s, t, direction (right=1).

    complete_backtrack = -np.ones((nr, nr, 2), dtype=int) # s, t, direction (right=1).
    incomplete_backtrack = -np.ones((nr, nr, 2), dtype=int) # s, t, direction (right=1).

    for i in range(nr):
        incomplete_0[i, 0] = NEGINF

    for k in range(1, nr):
        for s in range(nr - k):
            t = s + k
            tmp = NEGINF
            maxidx = s
            for r in range(s, t):
                cand = complete_1[s, r] + complete_0[r+1, t]
                if cand > tmp:
                    tmp = cand
                    maxidx = r
                if s == 0 and r == 0:
                    break
            incomplete_0[t, s] = tmp + scores[t, s]
            incomplete_1[s, t] = tmp + scores[s, t]
            incomplete_backtrack[s, t, 0] = maxidx
            incomplete_backtrack[s, t, 1] = maxidx

            tmp = NEGINF
            maxidx = s
            for r in range(s, t):
                cand = complete_0[s, r] + incomplete_0[t, r]
                if cand > tmp:
                    tmp = cand
                    maxidx = r
            complete_0[s, t] = tmp
            complete_backtrack[s, t, 0] = maxidx

            tmp = NEGINF
            maxidx = s + 1
            for r in range(s+1, t+1):
                cand = incomplete_1[s, r] + complete_1[r, t]
                if cand > tmp:
                    tmp = cand
                    maxidx = r
            complete_1[s, t] = tmp
            complete_backtrack[s, t, 1] = maxidx

    heads = -np.ones(N + 1, dtype=int)
    backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, N, 1, 1, heads)

    return heads


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void backtrack_eisner(np.ndarray[np.npy_intp, ndim=3] incomplete_backtrack,
        np.ndarray[np.npy_intp, ndim=3]complete_backtrack,
        int s, int t, int direction, int complete, np.ndarray[np.npy_intp, ndim=1] heads):
    cdef int r
    if s == t:
        return
    if complete:
        r = complete_backtrack[s, t, direction]
        if direction:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 0, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1, 1, heads)
            return
        else:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0, 0, heads)
            return
    else:
        r = incomplete_backtrack[s, t, direction]
        if direction:
            heads[t] = s
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r + 1, t, 0, 1, heads)
            return
        else:
            heads[s] = t
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r + 1, t, 0, 1, heads)
            return

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bool is_projective(np.ndarray[np.npy_intp, ndim=1] heads):
    cdef int n_len, i, j, cur
    cdef int edge1_0, edge1_1, edge2_0, edge2_1
    n_len = heads.shape[0]
    for i in range(n_len):
        if heads[i] < 0:
            continue
        for j in range(i + 1, n_len):
            if heads[j] < 0:
                continue
            edge1_0 = int_min(i, heads[i])
            edge1_1 = int_max(i, heads[i])
            edge2_0 = int_min(j, heads[j])
            edge2_1 = int_max(j, heads[j])
            if edge1_0 == edge2_0:
                if edge1_1 == edge2_1:
                    return False
                else:
                    continue
            if edge1_0 < edge2_0 and not (edge2_0 >= edge1_1 or edge2_1 <= edge1_1):
                return False
            if edge1_0 > edge2_0 and not (edge1_0 >= edge2_1 or edge1_1 <= edge2_1):
                return False
    return True

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.npy_intp, ndim=1] projectivize(np.ndarray[np.npy_intp, ndim=1] heads):
    if is_projective(heads):
        return heads

    cdef int n_len, h, m
    cdef np.ndarray[np.float64_t, ndim=2] scores

    n_len = heads.shape[0]
    scores = np.zeros((n_len, n_len))
    for m in range(1, n_len):
        h = heads[m]
        scores[h, m] = 1.
    return parse_proj(scores)

@cython.boundscheck(False)
@cython.wraparound(False)
def parse_transition_dp_mst(np.ndarray[np.float64_t, ndim=3] transition_scores, np.ndarray[np.float64_t, ndim=2] mst_scores):
    cdef int nr, nc, _, N, i, j, k, x, r, r1, r2, maxidx
    cdef np.ndarray[np.float64_t, ndim=3] table
    cdef np.float64_t tmp, cand
    cdef np.ndarray[np.npy_intp, ndim=3] backtrack
    cdef np.ndarray[np.npy_intp, ndim=3] backtrack_type
    cdef np.ndarray[np.npy_intp, ndim=1] transitions
    cdef np.ndarray[np.npy_intp, ndim=1] pred_heads

    nr, nc, _ = np.shape(transition_scores)

    table = np.full((nr, nr, nr), NEGINF)
    backtrack = np.full((nr, nr, nr), 0, dtype=int)
    backtrack_type = np.full((nr, nr, nr), 0, dtype=int)
    transitions = np.full(2*(nr-2)-1, -1, dtype=int)
    pred_heads = np.full(nr - 2, -1, dtype=int)

    for i in range(0, nr - 2):
        for j in range(i + 1, nr - 1):
            table[i, j, j + 1] = transition_scores[i, j, 0]

    for r in range(3, nr):
        for x in range(0, nr - r):
            j = x + r
            for i in range(x + 1, j - 1):
                for k in range(i + 1, j):
                    if j < nr - 1:
                        tmp = table[x, i, k] + table[i, k, j] + transition_scores[k, j, 1] + mst_scores[j-1, k-1]
                        if tmp > table[x, i, j]:
                            table[x, i, j] = tmp
                            backtrack[x, i, j] = k
                            backtrack_type[x, i, j] = 1
                    if i > 0:
                        tmp = table[x, i, k] + table[i, k, j] + transition_scores[k, j, 2] + mst_scores[i-1, k-1]
                        if tmp > table[x, i, j]:
                            table[x, i, j] = tmp
                            backtrack[x, i, j] = k
                            backtrack_type[x, i, j] = 2

    backtrack_dp_mst(backtrack, backtrack_type, 0, 1, nr-1, 0, transitions, pred_heads)
    return transitions, pred_heads

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int backtrack_dp_mst(np.ndarray[np.npy_intp, ndim=3] backtrack, np.ndarray[np.npy_intp, ndim=3] backtrack_type,
        int x, int i, int j, int cur_pos, np.ndarray[np.npy_intp, ndim=1] transitions, np.ndarray[np.npy_intp, ndim=1] heads):
    cdef int new_pos, k
    if j == i + 1:
        transitions[cur_pos] = 0
        return cur_pos + 1
    else:
        k = backtrack[x, i, j]
        new_pos = backtrack_dp_mst(backtrack, backtrack_type, x, i, k, cur_pos, transitions, heads)
        new_pos = backtrack_dp_mst(backtrack, backtrack_type, i, k, j, new_pos, transitions, heads)
        transitions[new_pos] = backtrack_type[x, i, j]
        if backtrack_type[x, i, j] == 1:
            heads[k-1] = j-1
        elif backtrack_type[x, i, j] == 2:
            heads[k-1] = i-1
        new_pos += 1
        return new_pos


@cython.boundscheck(False)
@cython.wraparound(False)
def parse_ah_dp_mst(np.ndarray[ndim=3, dtype=np.float64_t] transition_scores, np.ndarray[ndim=2, dtype=np.float64_t] mst_scores):
    cdef int nr, nc, _, N, i, j, k, r
    cdef np.ndarray[ndim=2, dtype=np.float64_t] table
    cdef np.float64_t tmp, cand
    cdef np.ndarray[ndim=2, dtype=np.npy_intp] backtrack
    cdef np.ndarray[ndim=2, dtype=np.npy_intp] backtrack_type
    cdef np.ndarray[ndim=1, dtype=np.npy_intp] transitions
    cdef np.ndarray[ndim=1, dtype=np.npy_intp] pred_heads

    nr, nc, _ = np.shape(transition_scores)

    table = np.full((nr, nr), NEGINF)
    backtrack = np.full((nr, nr), 0, dtype=int)
    backtrack_type = np.full((nr, nr), 0, dtype=int)
    transitions = np.full(2*(nr-2)-1, -1, dtype=int)
    pred_heads = np.full(nr - 2, -1, dtype=int)

    for j in range(0, nr - 1):
        table[j, j + 1] = 0.

    for r in range(2, nr):
        for i in range(0, nr - r):
            j = i + r
            for k in range(i + 1, j):
                if j < nr - 1:
                    tmp = table[i, k] + table[k, j] + transition_scores[i, k, 0] + transition_scores[k, j, 1] + mst_scores[j-1, k-1]
                    if tmp > table[i, j]:
                        table[i, j] = tmp
                        backtrack[i, j] = k
                        backtrack_type[i, j] = 1
                if i > 1 or (i == 1 and j == nr - 1):
                    tmp = table[i, k] + table[k, j] + transition_scores[i, k, 0] + transition_scores[k, j, 2] + mst_scores[i-1, k-1]
                    if tmp > table[i, j]:
                        table[i, j] = tmp
                        backtrack[i, j] = k
                        backtrack_type[i, j] = 2

    backtrack_ah_dp_mst(backtrack, backtrack_type, 1, nr-1, 0, transitions, pred_heads)
    return transitions, pred_heads


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int backtrack_ah_dp_mst(np.ndarray[ndim=2, dtype=np.npy_intp] backtrack, np.ndarray[ndim=2, dtype=np.npy_intp] backtrack_type, int i, int j, int cur_pos, np.ndarray[ndim=1, dtype=np.npy_intp] transitions, np.ndarray[ndim=1, dtype=np.npy_intp] heads):
    cdef int new_pos, k
    if j == i + 1:
        transitions[cur_pos] = 0
        return cur_pos + 1
    else:
        k = backtrack[i, j]
        new_pos = backtrack_ah_dp_mst(backtrack, backtrack_type, i, k, cur_pos, transitions, heads)
        new_pos = backtrack_ah_dp_mst(backtrack, backtrack_type, k, j, new_pos, transitions, heads)
        transitions[new_pos] = backtrack_type[i, j]
        if backtrack_type[i, j] == 1:
            heads[k - 1] = j - 1
        elif backtrack_type[i, j] == 2:
            heads[k - 1] = i - 1
        new_pos += 1
        return new_pos


@cython.boundscheck(False)
@cython.wraparound(False)
def parse_ae_dp_mst(np.ndarray[ndim=3, dtype=np.float64_t] transition_scores, np.ndarray[ndim=2, dtype=np.float64_t] mst_scores):
    cdef int nr, nc, _, N, i, j, k, b, r
    cdef np.float64_t tmp, cand
    cdef np.ndarray[ndim=3, dtype=np.float64_t] table
    cdef np.ndarray[ndim=3, dtype=np.npy_intp] backtrack
    cdef np.ndarray[ndim=3, dtype=np.npy_intp] backtrack_type
    cdef np.ndarray[ndim=1, dtype=np.npy_intp] transitions
    cdef np.ndarray[ndim=1, dtype=np.npy_intp] pred_heads

    nr, nc, _ = np.shape(transition_scores)

    table = np.full((nr, 2, nr), NEGINF)
    backtrack = np.full((nr, 2, nr), 0, dtype=int)
    backtrack_type = np.full((nr, 2, nr), 0, dtype=int)
    transitions = np.full(2 * (nr - 2) -1, -1, dtype=int)
    pred_heads = np.full(nr - 2, -1, dtype=int)

    for j in range(0, nr - 1):
        table[j, 0, j + 1] = 0.
        table[j, 1, j + 1] = 0.

    for r in range(2, nr):
        for i in range(0, nr - r):
            j = i + r
            for b in range(0, 2):
                for k in range(i + 1, j):
                    if j < nr - 1:
                        # First shift then a left arc
                        tmp = table[i, b, k] + table[k, 0, j] + transition_scores[i, k, 0] + transition_scores[k, j, 1] + mst_scores[j - 1, k - 1]
                        if tmp > table[i, b, j]:
                            table[i, b, j] = tmp
                            backtrack[i, b, j] = k
                            backtrack_type[i, b, j] = 1

                    if i > 1 or (i == 1 and j == nr - 1):
                        # First right arc then reduce
                        tmp = table[i, b, k] + table[k, 1, j] + transition_scores[i, k, 2] + transition_scores[k, j, 3] + mst_scores[i - 1, k - 1]
                        if tmp > table[i, b, j]:
                            table[i, b, j] = tmp
                            backtrack[i, b, j] = k
                            backtrack_type[i, b, j] = 3

    backtrack_ae_dp_mst(backtrack, backtrack_type, 1, 0, nr - 1, 0, transitions, pred_heads)
    return transitions, pred_heads


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int backtrack_ae_dp_mst(np.ndarray[ndim=3, dtype=np.npy_intp] backtrack, np.ndarray[ndim=3, dtype=np.npy_intp] backtrack_type, int i, int b, int j, int cur_pos, np.ndarray[ndim=1, dtype=np.npy_intp] transitions, np.ndarray[ndim=1, dtype=np.npy_intp] heads):
    cdef int new_pos, k, t
    if j == i + 1 and b == 0:
        # shift
        transitions[cur_pos] = 0
        return cur_pos + 1
    if j == i + 1 and b == 1:
        # rightarc
        transitions[cur_pos] = 2
        return cur_pos + 1
    else:
        k = backtrack[i, b, j]
        t = backtrack_type[i, b, j]
        new_pos = backtrack_ae_dp_mst(backtrack, backtrack_type, i, b, k, cur_pos, transitions, heads)
        if t == 1:
            # leftarc
            new_pos = backtrack_ae_dp_mst(backtrack, backtrack_type, k, 0, j, new_pos, transitions, heads)
            transitions[new_pos] = 1
            heads[k - 1] = j - 1
        elif t == 3:
            # reduce
            new_pos = backtrack_ae_dp_mst(backtrack, backtrack_type, k, 1, j, new_pos, transitions, heads)
            transitions[new_pos] = 3
            heads[k - 1] = i - 1

        new_pos += 1
        return new_pos
