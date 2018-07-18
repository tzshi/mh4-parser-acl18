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
def parse_mh4(np.ndarray[ndim=2, dtype=np.float64_t] mst_scores):
    cdef int nr, nc, N, i, j, k, l, r1, r12, r13, r2, r3
    cdef np.float64_t tmp, cand

    cdef np.ndarray[ndim=2, dtype=np.float64_t] table2
    cdef np.ndarray[ndim=3, dtype=np.float64_t] table3
    cdef np.ndarray[ndim=4, dtype=np.float64_t] table4
    cdef np.ndarray[ndim=2, dtype=np.float64_t] guarded_mst_scores

    cdef np.ndarray[np.npy_intp, ndim=2] type2_bt
    cdef np.ndarray[np.npy_intp, ndim=3] type3_bt
    cdef np.ndarray[np.npy_intp, ndim=4] type4_bt
    cdef np.ndarray[np.npy_intp, ndim=2] bt2
    cdef np.ndarray[np.npy_intp, ndim=3] bt3
    cdef np.ndarray[np.npy_intp, ndim=1] heads

    nr, nc = np.shape(mst_scores)
    N = nr + 1

    table2 = np.full((nr, N), NEGINF, dtype=np.float)
    for i in range(nr):
        table2[i, i + 1] = 0.
    table3 = np.full((nr, nr, N), NEGINF, dtype=np.float)
    for i in range(nr - 1):
        table3[i, i + 1, i + 2] = 0.
    table4 = np.full((nr, nr, nr, N), NEGINF, dtype=np.float)
    for i in range(nr - 2):
        table4[i, i + 1, i + 2, i + 3] = 0.
    guarded_mst_scores = np.full((N, N), NEGINF, dtype=np.float)

    type2_bt = np.zeros((N, N), dtype=int)
    type3_bt = np.zeros((N, N, N), dtype=int)
    type4_bt = np.zeros((N, N, N, N), dtype=int)
    bt2 = np.zeros((N, N), dtype=int)
    bt3 = np.zeros((N, N, N), dtype=int)

    heads = -np.ones(nr, dtype=int)

    for i in range(0, nr):
        for j in range(0, nc):
            guarded_mst_scores[i, j] = mst_scores[i, j]

    for r13 in range(2, N):
        for r1 in range(1, r13):
            for r2 in range(1, r13 - r1 + 1):
                r3 = r13 - r1 - r2
                for i in range(0, N - r13):
                    j = i + r1
                    k = j + r2
                    l = k + r3

                    # combine rule # 1
                    # type3 = 1
                    tmp = table2[i, j] + table2[j, l]
                    if tmp > table3[i, j, l]:
                        table3[i, j, l] = tmp
                        type3_bt[i, j, l] = 1

                    if k != l:
                        # combine rule # 2
                        # type4 = 1
                        tmp = table3[i, j, k] + table2[k, l]
                        if tmp > table4[i, j, k, l]:
                            table4[i, j, k, l] = tmp
                            type4_bt[i, j, k, l] = 1

                        # combine rule # 3
                        # type4 = 2
                        tmp = table2[i, j] + table3[j, k, l]
                        if tmp > table4[i, j, k, l]:
                            table4[i, j, k, l] = tmp
                            type4_bt[i, j, k, l] = 2

                    if k != l:
                        # link rule # 5
                        # type3 = 2
                        tmp = table4[i, j, k, l] + guarded_mst_scores[i, k]
                        if tmp > table3[i, j, l]:
                            table3[i, j, l] = tmp
                            type3_bt[i, j, l] = 2
                            bt3[i, j, l] = k

                        # type3 = 3
                        tmp = table4[i, j, k, l] + guarded_mst_scores[j, k]
                        if tmp > table3[i, j, l]:
                            table3[i, j, l] = tmp
                            type3_bt[i, j, l] = 3
                            bt3[i, j, l] = k

                        # type3 = 4
                        tmp = table4[i, j, k, l] + guarded_mst_scores[l, k]
                        if tmp > table3[i, j, l]:
                            table3[i, j, l] = tmp
                            type3_bt[i, j, l] = 4
                            bt3[i, j, l] = k

                        # link rule # 6
                        # type3 = 5
                        tmp = table4[i, j, k, l] + guarded_mst_scores[i, j]
                        if tmp > table3[i, k, l]:
                            table3[i, k, l] = tmp
                            type3_bt[i, k, l] = 5
                            bt3[i, k, l] = j

                        # type3 = 6
                        tmp = table4[i, j, k, l] + guarded_mst_scores[k, j]
                        if tmp > table3[i, k, l]:
                            table3[i, k, l] = tmp
                            type3_bt[i, k, l] = 6
                            bt3[i, k, l] = j

                        # type3 = 7
                        tmp = table4[i, j, k, l] + guarded_mst_scores[l, j]
                        if tmp > table3[i, k, l]:
                            table3[i, k, l] = tmp
                            type3_bt[i, k, l] = 7
                            bt3[i, k, l] = j

                    # link rule # 4
                    # type2 = 1
                    tmp = table3[i, j, l] + guarded_mst_scores[i, j]
                    if tmp > table2[i, l]:
                        table2[i, l] = tmp
                        type2_bt[i, l] = 1
                        bt2[i, l] = j

                    # type2 = 2
                    tmp = table3[i, j, l] + guarded_mst_scores[l, j]
                    if tmp > table2[i, l]:
                        table2[i, l] = tmp
                        type2_bt[i, l] = 2
                        bt2[i, l] = j

    backtrack_mh4(2, 0, N-1, 0, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads)
    # print(table2[0, N-1])

    return heads

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void backtrack_mh4(int n, int i, int j, int k, int l,
        np.ndarray[np.npy_intp, ndim=2] type2_bt,
        np.ndarray[np.npy_intp, ndim=3] type3_bt,
        np.ndarray[np.npy_intp, ndim=4] type4_bt,
        np.ndarray[np.npy_intp, ndim=2] bt2,
        np.ndarray[np.npy_intp, ndim=3] bt3,
        np.ndarray[np.npy_intp, ndim=1] heads):
    cdef int t, r
    if n == 2:
        if j == i + 1:
            return
        t = type2_bt[i, j]
        r = bt2[i, j]
        if t == 1:
            heads[r] = i
            backtrack_mh4(3, i, r, j, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads)
            return
        elif t == 2:
            heads[r] = j
            backtrack_mh4(3, i, r, j, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads)
            return
    elif n == 3:
        if j == i + 1 and k == j + 1:
            return
        t = type3_bt[i, j, k]
        r = bt3[i, j, k]
        if t == 1:
            backtrack_mh4(2, i, j, 0, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads)
            backtrack_mh4(2, j, k, 0, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads)
            return
        elif t == 2:
            heads[r] = i
            backtrack_mh4(4, i, j, r, k, type2_bt, type3_bt, type4_bt, bt2, bt3, heads)
            return
        elif t == 3:
            heads[r] = j
            backtrack_mh4(4, i, j, r, k, type2_bt, type3_bt, type4_bt, bt2, bt3, heads)
            return
        elif t == 4:
            heads[r] = k
            backtrack_mh4(4, i, j, r, k, type2_bt, type3_bt, type4_bt, bt2, bt3, heads)
            return
        elif t == 5:
            heads[r] = i
            backtrack_mh4(4, i, r, j, k, type2_bt, type3_bt, type4_bt, bt2, bt3, heads)
            return
        elif t == 6:
            heads[r] = j
            backtrack_mh4(4, i, r, j, k, type2_bt, type3_bt, type4_bt, bt2, bt3, heads)
            return
        elif t == 7:
            heads[r] = k
            backtrack_mh4(4, i, r, j, k, type2_bt, type3_bt, type4_bt, bt2, bt3, heads)
            return
    elif n == 4:
        if j == i + 1 and k == j + 1 and l == k + 1:
            return
        t = type4_bt[i, j, k, l]
        if t == 1:
            backtrack_mh4(3, i, j, k, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads)
            backtrack_mh4(2, k, l, 0, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads)
            return
        elif t == 2:
            backtrack_mh4(2, i, j, 0, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads)
            backtrack_mh4(3, j, k, l, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads)
            return

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.npy_intp, ndim=1] mh4ize(np.ndarray[np.npy_intp, ndim=1] heads):
    cdef int n_len, h, m
    cdef np.ndarray[np.float64_t, ndim=2] scores

    n_len = heads.shape[0]
    scores = np.zeros((n_len, n_len))
    for m in range(1, n_len):
        h = heads[m]
        scores[h, m] = 1.
    return parse_mh4(scores)
