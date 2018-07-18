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
def mh4t_combine_scores(np.ndarray[ndim=3, dtype=np.float64_t] scores1, np.ndarray[ndim=3, dtype=np.float64_t] scores2):
    cdef int nr, nc, i, j, k, na, a
    cdef np.ndarray[ndim=4, dtype=np.float64_t] ret

    nr, nc, na = np.shape(scores1)
    ret = np.full((nr, nr, nr, na), NEGINF, dtype=np.float)

    for i in range(nr):
        for j in range(i + 1, nr):
            for k in range(j + 1, nr):
                for a in range(na):
                    ret[i, j, k, a] = scores1[i, j, a] + scores2[j, k, a]

    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
def parse_mh4t(np.ndarray[ndim=3, dtype=np.float64_t] tran_scores, np.ndarray[ndim=2, dtype=np.float64_t] mst_scores):
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
    cdef np.ndarray[np.npy_intp, ndim=2] traces

    nr, nc = np.shape(mst_scores)
    N = nr + 1

    table2 = np.full((nr, N), NEGINF, dtype=np.float)
    for i in range(nr):
        table2[i, i + 1] = 0.
    table3 = np.full((nr, nr, N), NEGINF, dtype=np.float)
    table4 = np.full((nr, nr, nr, N), NEGINF, dtype=np.float)
    guarded_mst_scores = np.full((N, N), NEGINF, dtype=np.float)

    type2_bt = np.zeros((N, N), dtype=int)
    type3_bt = np.zeros((N, N, N), dtype=int)
    type4_bt = np.zeros((N, N, N, N), dtype=int)
    bt2 = np.zeros((N, N), dtype=int)
    bt3 = np.zeros((N, N, N), dtype=int)

    heads = -np.ones(nr, dtype=int)
    traces = -np.ones((2 * (nr - 1), 3), dtype=int)

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
                    tmp = table2[i, j] + table2[j, l] + tran_scores[i, j, 0]
                    if tmp > table3[i, j, l]:
                        table3[i, j, l] = tmp
                        type3_bt[i, j, l] = 1

                    if k != l:
                        # combine rule # 2
                        # type4 = 1
                        tmp = table3[i, j, k] + table2[k, l] + tran_scores[j, k, 0]
                        if tmp > table4[i, j, k, l]:
                            table4[i, j, k, l] = tmp
                            type4_bt[i, j, k, l] = 1

                        # combine rule # 3
                        # type4 = 2
                        tmp = table2[i, j] + table3[j, k, l] + tran_scores[i, j, 0]
                        if tmp > table4[i, j, k, l]:
                            table4[i, j, k, l] = tmp
                            type4_bt[i, j, k, l] = 2

                    if k != l:
                        # link rule # 5
                        # type3 = 2
                        tmp = table4[i, j, k, l] + guarded_mst_scores[i, k] + tran_scores[k, l, 4]
                        if tmp > table3[i, j, l]:
                            table3[i, j, l] = tmp
                            type3_bt[i, j, l] = 2
                            bt3[i, j, l] = k

                        # type3 = 3
                        tmp = table4[i, j, k, l] + guarded_mst_scores[j, k] + tran_scores[k, l, 5]
                        if tmp > table3[i, j, l]:
                            table3[i, j, l] = tmp
                            type3_bt[i, j, l] = 3
                            bt3[i, j, l] = k

                        # type3 = 4
                        tmp = table4[i, j, k, l] + guarded_mst_scores[l, k] + tran_scores[k, l, 6]
                        if tmp > table3[i, j, l]:
                            table3[i, j, l] = tmp
                            type3_bt[i, j, l] = 4
                            bt3[i, j, l] = k

                        # link rule # 6
                        # type3 = 5
                        tmp = table4[i, j, k, l] + guarded_mst_scores[i, j] + tran_scores[k, l, 1]
                        if tmp > table3[i, k, l]:
                            table3[i, k, l] = tmp
                            type3_bt[i, k, l] = 5
                            bt3[i, k, l] = j

                        # type3 = 6
                        tmp = table4[i, j, k, l] + guarded_mst_scores[k, j] + tran_scores[k, l, 2]
                        if tmp > table3[i, k, l]:
                            table3[i, k, l] = tmp
                            type3_bt[i, k, l] = 6
                            bt3[i, k, l] = j

                        # type3 = 7
                        tmp = table4[i, j, k, l] + guarded_mst_scores[l, j] + tran_scores[k, l, 3]
                        if tmp > table3[i, k, l]:
                            table3[i, k, l] = tmp
                            type3_bt[i, k, l] = 7
                            bt3[i, k, l] = j

                    # link rule # 4
                    # type2 = 1
                    tmp = table3[i, j, l] + guarded_mst_scores[i, j] + tran_scores[j, l, 5]
                    if tmp > table2[i, l]:
                        table2[i, l] = tmp
                        type2_bt[i, l] = 1
                        bt2[i, l] = j

                    # type2 = 2
                    tmp = table3[i, j, l] + guarded_mst_scores[l, j] + tran_scores[j, l, 6]
                    if tmp > table2[i, l]:
                        table2[i, l] = tmp
                        type2_bt[i, l] = 2
                        bt2[i, l] = j

    backtrack_mh4t(2, 0, N-1, 0, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, 0)

    return table2[0, N-1], heads, traces

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int backtrack_mh4t(int n, int i, int j, int k, int l,
        np.ndarray[np.npy_intp, ndim=2] type2_bt,
        np.ndarray[np.npy_intp, ndim=3] type3_bt,
        np.ndarray[np.npy_intp, ndim=4] type4_bt,
        np.ndarray[np.npy_intp, ndim=2] bt2,
        np.ndarray[np.npy_intp, ndim=3] bt3,
        np.ndarray[np.npy_intp, ndim=1] heads,
        np.ndarray[np.npy_intp, ndim=2] traces,
        int traces_i):
    cdef int t, r, tr
    if n == 2:
        if j == i + 1:
            return traces_i
        t = type2_bt[i, j]
        r = bt2[i, j]
        if t == 1:
            tr = backtrack_mh4t(3, i, r, j, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, traces_i)
            heads[r] = i
            traces[tr][0] = r
            traces[tr][1] = j
            traces[tr][2] = 5
            return tr + 1
        elif t == 2:
            tr = backtrack_mh4t(3, i, r, j, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, traces_i)
            heads[r] = j
            traces[tr][0] = r
            traces[tr][1] = j
            traces[tr][2] = 6
            return tr + 1
    elif n == 3:
        t = type3_bt[i, j, k]
        r = bt3[i, j, k]
        if t == 1:
            tr = backtrack_mh4t(2, i, j, 0, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, traces_i)
            traces[tr][0] = i
            traces[tr][1] = j
            traces[tr][2] = 0
            return backtrack_mh4t(2, j, k, 0, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, tr + 1)
        elif t == 2:
            tr = backtrack_mh4t(4, i, j, r, k, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, traces_i)
            heads[r] = i
            traces[tr][0] = r
            traces[tr][1] = k
            traces[tr][2] = 4
            return tr + 1
        elif t == 3:
            tr =  backtrack_mh4t(4, i, j, r, k, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, traces_i)
            heads[r] = j
            traces[tr][0] = r
            traces[tr][1] = k
            traces[tr][2] = 5
            return tr + 1
        elif t == 4:
            tr = backtrack_mh4t(4, i, j, r, k, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, traces_i)
            heads[r] = k
            traces[tr][0] = r
            traces[tr][1] = k
            traces[tr][2] = 6
            return tr + 1
        elif t == 5:
            tr = backtrack_mh4t(4, i, r, j, k, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, traces_i)
            heads[r] = i
            traces[tr][0] = j
            traces[tr][1] = k
            traces[tr][2] = 1
            return tr + 1
        elif t == 6:
            tr = backtrack_mh4t(4, i, r, j, k, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, traces_i)
            heads[r] = j
            traces[tr][0] = j
            traces[tr][1] = k
            traces[tr][2] = 2
            return tr + 1
        elif t == 7:
            tr = backtrack_mh4t(4, i, r, j, k, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, traces_i)
            heads[r] = k
            traces[tr][0] = j
            traces[tr][1] = k
            traces[tr][2] = 3
            return tr + 1
    elif n == 4:
        t = type4_bt[i, j, k, l]
        if t == 1:
            tr = backtrack_mh4t(3, i, j, k, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, traces_i)
            traces[tr][0] = j
            traces[tr][1] = k
            traces[tr][2] = 0
            return backtrack_mh4t(2, k, l, 0, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, tr + 1)
        elif t == 2:
            tr = backtrack_mh4t(2, i, j, 0, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, traces_i)
            traces[tr][0] = i
            traces[tr][1] = j
            traces[tr][2] = 0
            return backtrack_mh4t(3, j, k, l, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, tr + 1)


@cython.boundscheck(False)
@cython.wraparound(False)
def parse_mh4t_sh(np.ndarray[ndim=2, dtype=np.float64_t] shift_scores, np.ndarray[ndim=4, dtype=np.float64_t] reduce_scores,
        np.ndarray[ndim=2, dtype=np.float64_t] mst_scores):
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
    cdef np.ndarray[np.npy_intp, ndim=2] traces

    nr, nc = np.shape(mst_scores)
    N = nr + 1

    table2 = np.full((nr, N), NEGINF, dtype=np.float)
    for i in range(nr):
        table2[i, i + 1] = 0.
    table3 = np.full((nr, nr, N), NEGINF, dtype=np.float)
    table4 = np.full((nr, nr, nr, N), NEGINF, dtype=np.float)
    guarded_mst_scores = np.full((N, N), NEGINF, dtype=np.float)

    type2_bt = np.zeros((N, N), dtype=int)
    type3_bt = np.zeros((N, N, N), dtype=int)
    type4_bt = np.zeros((N, N, N, N), dtype=int)
    bt2 = np.zeros((N, N), dtype=int)
    bt3 = np.zeros((N, N, N), dtype=int)

    heads = -np.ones(nr, dtype=int)
    traces = -np.ones((2 * (nr - 1), 5), dtype=int)

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
                    tmp = table2[i, j] + table2[j, l] + shift_scores[i, j]
                    if tmp > table3[i, j, l]:
                        table3[i, j, l] = tmp
                        type3_bt[i, j, l] = 1

                    if k != l:
                        # combine rule # 2
                        # type4 = 1
                        tmp = table3[i, j, k] + table2[k, l] + shift_scores[j, k]
                        if tmp > table4[i, j, k, l]:
                            table4[i, j, k, l] = tmp
                            type4_bt[i, j, k, l] = 1

                        # combine rule # 3
                        # type4 = 2
                        tmp = table2[i, j] + table3[j, k, l] + shift_scores[i, j]
                        if tmp > table4[i, j, k, l]:
                            table4[i, j, k, l] = tmp
                            type4_bt[i, j, k, l] = 2

                    if k != l:
                        # link rule # 5
                        # type3 = 2
                        tmp = table4[i, j, k, l] + guarded_mst_scores[i, k] + reduce_scores[j, k, l, 0]
                        if tmp > table3[i, j, l]:
                            table3[i, j, l] = tmp
                            type3_bt[i, j, l] = 2
                            bt3[i, j, l] = k

                        # type3 = 3
                        tmp = table4[i, j, k, l] + guarded_mst_scores[j, k] + reduce_scores[j, k, l, 1]
                        if tmp > table3[i, j, l]:
                            table3[i, j, l] = tmp
                            type3_bt[i, j, l] = 3
                            bt3[i, j, l] = k

                        # type3 = 4
                        tmp = table4[i, j, k, l] + guarded_mst_scores[l, k] + reduce_scores[j, k, l, 2]
                        if tmp > table3[i, j, l]:
                            table3[i, j, l] = tmp
                            type3_bt[i, j, l] = 4
                            bt3[i, j, l] = k

                        # link rule # 6
                        # type3 = 5
                        tmp = table4[i, j, k, l] + guarded_mst_scores[i, j] + reduce_scores[j, k, l, 3]
                        if tmp > table3[i, k, l]:
                            table3[i, k, l] = tmp
                            type3_bt[i, k, l] = 5
                            bt3[i, k, l] = j

                        # type3 = 6
                        tmp = table4[i, j, k, l] + guarded_mst_scores[k, j] + reduce_scores[j, k, l, 4]
                        if tmp > table3[i, k, l]:
                            table3[i, k, l] = tmp
                            type3_bt[i, k, l] = 6
                            bt3[i, k, l] = j

                        # type3 = 7
                        tmp = table4[i, j, k, l] + guarded_mst_scores[l, j] + reduce_scores[j, k, l, 5]
                        if tmp > table3[i, k, l]:
                            table3[i, k, l] = tmp
                            type3_bt[i, k, l] = 7
                            bt3[i, k, l] = j

                    # link rule # 4
                    # type2 = 1
                    tmp = table3[i, j, l] + guarded_mst_scores[i, j] + reduce_scores[i, j, l, 1]
                    if tmp > table2[i, l]:
                        table2[i, l] = tmp
                        type2_bt[i, l] = 1
                        bt2[i, l] = j

                    # type2 = 2
                    tmp = table3[i, j, l] + guarded_mst_scores[l, j] + reduce_scores[i, j, l, 2]
                    if tmp > table2[i, l]:
                        table2[i, l] = tmp
                        type2_bt[i, l] = 2
                        bt2[i, l] = j

    backtrack_mh4t_sh(2, 0, N-1, 0, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, 0)

    return table2[0, N-1], heads, traces

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int backtrack_mh4t_sh(int n, int i, int j, int k, int l,
        np.ndarray[np.npy_intp, ndim=2] type2_bt,
        np.ndarray[np.npy_intp, ndim=3] type3_bt,
        np.ndarray[np.npy_intp, ndim=4] type4_bt,
        np.ndarray[np.npy_intp, ndim=2] bt2,
        np.ndarray[np.npy_intp, ndim=3] bt3,
        np.ndarray[np.npy_intp, ndim=1] heads,
        np.ndarray[np.npy_intp, ndim=2] traces,
        int traces_i):
    cdef int t, r, tr
    if n == 2:
        if j == i + 1:
            return traces_i
        t = type2_bt[i, j]
        r = bt2[i, j]
        if t == 1:
            tr = backtrack_mh4t_sh(3, i, r, j, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, traces_i)
            heads[r] = i
            traces[tr][0] = 3
            traces[tr][1] = i
            traces[tr][2] = r
            traces[tr][3] = j
            traces[tr][4] = 1
            return tr + 1
        elif t == 2:
            tr = backtrack_mh4t_sh(3, i, r, j, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, traces_i)
            heads[r] = j
            traces[tr][0] = 3
            traces[tr][1] = i
            traces[tr][2] = r
            traces[tr][3] = j
            traces[tr][4] = 2
            return tr + 1
    elif n == 3:
        t = type3_bt[i, j, k]
        r = bt3[i, j, k]
        if t == 1:
            tr = backtrack_mh4t_sh(2, i, j, 0, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, traces_i)
            traces[tr][0] = 2
            traces[tr][1] = i
            traces[tr][2] = j
            return backtrack_mh4t_sh(2, j, k, 0, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, tr + 1)
        elif t == 2:
            tr = backtrack_mh4t_sh(4, i, j, r, k, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, traces_i)
            heads[r] = i
            traces[tr][0] = 3
            traces[tr][1] = j
            traces[tr][2] = r
            traces[tr][3] = k
            traces[tr][4] = 0
            return tr + 1
        elif t == 3:
            tr =  backtrack_mh4t_sh(4, i, j, r, k, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, traces_i)
            heads[r] = j
            traces[tr][0] = 3
            traces[tr][1] = j
            traces[tr][2] = r
            traces[tr][3] = k
            traces[tr][4] = 1
            return tr + 1
        elif t == 4:
            tr = backtrack_mh4t_sh(4, i, j, r, k, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, traces_i)
            heads[r] = k
            traces[tr][0] = 3
            traces[tr][1] = j
            traces[tr][2] = r
            traces[tr][3] = k
            traces[tr][4] = 2
            return tr + 1
        elif t == 5:
            tr = backtrack_mh4t_sh(4, i, r, j, k, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, traces_i)
            heads[r] = i
            traces[tr][0] = 3
            traces[tr][1] = r
            traces[tr][2] = j
            traces[tr][3] = k
            traces[tr][4] = 3
            return tr + 1
        elif t == 6:
            tr = backtrack_mh4t_sh(4, i, r, j, k, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, traces_i)
            heads[r] = j
            traces[tr][0] = 3
            traces[tr][1] = r
            traces[tr][2] = j
            traces[tr][3] = k
            traces[tr][4] = 4
            return tr + 1
        elif t == 7:
            tr = backtrack_mh4t_sh(4, i, r, j, k, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, traces_i)
            heads[r] = k
            traces[tr][0] = 3
            traces[tr][1] = r
            traces[tr][2] = j
            traces[tr][3] = k
            traces[tr][4] = 5
            return tr + 1
    elif n == 4:
        t = type4_bt[i, j, k, l]
        if t == 1:
            tr = backtrack_mh4t_sh(3, i, j, k, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, traces_i)
            traces[tr][0] = 2
            traces[tr][1] = j
            traces[tr][2] = k
            return backtrack_mh4t_sh(2, k, l, 0, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, tr + 1)
        elif t == 2:
            tr = backtrack_mh4t_sh(2, i, j, 0, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, traces_i)
            traces[tr][0] = 2
            traces[tr][1] = i
            traces[tr][2] = j
            return backtrack_mh4t_sh(3, j, k, l, 0, type2_bt, type3_bt, type4_bt, bt2, bt3, heads, traces, tr + 1)
