#!/usr/bin/env python
# encoding: utf-8

cimport cython

import numpy as np
cimport numpy as np
np.import_array()
import sys

from cpython cimport bool

cdef np.float64_t NEGINF = -np.inf
cdef np.float64_t INF = np.inf
cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline np.float64_t float64_max(np.float64_t a, np.float64_t b): return a if a >= b else b
cdef inline np.float64_t float64_min(np.float64_t a, np.float64_t b): return a if a <= b else b


def is_1ec(np.ndarray[ndim=1, dtype=np.npy_intp] heads):
    # O(n^3) check
    cdef int e1l, e1r, e2l, e2r, nr, i, j, k
    cdef set points

    nr, = np.shape(heads)

    for i in range(1, nr):
        e1l = int_min(i, heads[i])
        e1r = int_max(i, heads[i])

        points = {k for k in range(nr)}

        for j in range(1, nr):
            e2l = int_min(j, heads[j])
            e2r = int_max(j, heads[j])

            # crossing
            if (e1l < e2l and e1r > e2l and e1r < e2r) or (e1l > e2l and e1l < e2r and e1r > e2r):
                points &= {e2l, e2r}

        if len(points) < 1:
            return False

    return True

cdef enum EC_TYPE:
    TYPE_INT,
    TYPE_TRAP = TYPE_INT + 4,
    TYPE_CHAIN = TYPE_TRAP + 2,
    TYPE_LEFTFARCROSSED,
    TYPE_TRIFARCROSSED,
    TYPE_TRIG = TYPE_TRIFARCROSSED + 2,
    TYPE_TRAPG = TYPE_TRIG + 2,
    TYPE_BOXG = TYPE_TRAPG + 2,
    TYPE_ONEFARCROSSEDG,
    TYPE_PAGE1 = TYPE_ONEFARCROSSEDG + 2,
    TYPE_PAGE2,
    TYPE_LR,
    TYPE_CHAIN_JFROMI = TYPE_LR + 2,
    TYPE_CHAIN_IFROMJ,
    TYPE_CHAIN_JFROMX,
    TYPE_CHAIN_IFROMX,
    TYPE_N,
    TYPE_L = TYPE_N + 8,
    TYPE_R = TYPE_L + 8,
    TYPE_L_XFROMI = TYPE_R + 8,
    TYPE_L_IFROMX,
    TYPE_L_JFROMX,
    TYPE_L_JFROMI,
    TYPE_R_XFROMJ,
    TYPE_R_JFROMX,
    TYPE_R_IFROMX,
    TYPE_R_IFROMJ,
    TYPE_TOTAL

cdef enum EDGE_TYPE:
    TYPE_GS,
    TYPE_G,
    TYPE_S,
    TYPE_E,
    TYPE_CE

cdef inline int idx_type(int t, int b0, int b1=0, int b2=0):
    return t + b0 + b1 * 2 + b2 * 4

cdef inline int vcross_start(int i, int j, int bi, int bj, int t):
    if t == TYPE_LR or ((t == TYPE_L or bi == 0) and (t == TYPE_R or bj == 0)):
        return i + 1
    if bi == 1 and (t == TYPE_R or (t == TYPE_N and bj == 0)):
        return i
    if bj == 1 and (t == TYPE_L or (t == TYPE_N and bi == 0)):
        return i + 1
    if t == TYPE_N and bi == 1 and bj == 1:
        return i
    return -1

cdef inline int vcross_end(int i, int j, int bi, int bj, int t):
    if t == TYPE_LR or ((t == TYPE_L or bi == 0) and (t == TYPE_R or bj == 0)):
        return j - 1
    if bi == 1 and (t == TYPE_R or (t == TYPE_N and bj == 0)):
        return j - 1
    if bj == 1 and (t == TYPE_L or (t == TYPE_N and bi == 0)):
        return j
    if t == TYPE_N and bi == 1 and bj == 1:
        return j
    return -1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.float64_t GS(int g, int h, int m, int s, double [:, :, ::1] scores):
    return scores[g, m, 0] + scores[h, m, 1] + scores[s, m, 2]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.float64_t G(int g, int h, int m, double [:, :, ::1] scores):
    return scores[g, m, 0] + scores[h, m, 1]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.float64_t S(int h, int m, int s, double [:, :, ::1] scores):
    return scores[h, m, 1] + scores[s, m, 2]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.float64_t E(int h, int m, double [:, :, ::1] scores):
    return scores[h, m, 1]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.float64_t CE(int h, int m, double [:, :, ::1] scores):
    return scores[h, m, 3]

@cython.boundscheck(False)
@cython.wraparound(False)
def parse_1ec_o3(np.ndarray[ndim=3, dtype=np.float64_t] mst_scores):
    cdef int nr, nc, _
    cdef int i, j, d, k, _k, l, h, g, x, bl, bm, br, bi, bj, bx, b, idx
    cdef int start, end
    cdef np.float64_t tmp, cand

    cdef double [:, :, ::1] scores = np.ascontiguousarray(mst_scores)

    nr, nc, _ = np.shape(scores)

    cdef double [:, :, :, ::1] Int = np.full((nr, nr, 2, 2), NEGINF, dtype=np.float, order="c")
    cdef double [:, :, ::1] Trap = np.full((nr, nr, 2), NEGINF, dtype=np.float, order="c")
    cdef double [:, ::1] Chain = np.full((nr, nr), NEGINF, dtype=np.float, order="c")
    cdef double [:, ::1] LeftFarCrossed = np.full((nr, nr), NEGINF, dtype=np.float, order="c")
    cdef double [:, :, ::1] TriFarCrossed = np.full((nr, nr, 2), NEGINF, dtype=np.float, order="c")

    cdef double [:, :, :, ::1] TriG = np.full((nr, nr, nr, 2), NEGINF, dtype=np.float, order="c")
    cdef double [:, :, :, ::1] TrapG = np.full((nr, nr, nr, 2), NEGINF, dtype=np.float, order="c")
    cdef double [:, :, ::1] BoxG = np.full((nr, nr, nr), NEGINF, dtype=np.float, order="c")
    cdef double [:, :, :, ::1] OneFarCrossedG = np.full((nr, nr, nr, 2), NEGINF, dtype=np.float, order="c")
    cdef double [:, :, ::1] Page1 = np.full((nr, nr, nr), NEGINF, dtype=np.float, order="c")
    cdef double [:, :, ::1] Page2 = np.full((nr, nr, nr), NEGINF, dtype=np.float, order="c")
    cdef double [:, :, :, ::1] LR = np.full((nr, nr, nr, 2), NEGINF, dtype=np.float, order="c")
    cdef double [:, :, ::1] Chain_JFromI = np.full((nr, nr, nr), NEGINF, dtype=np.float, order="c")
    cdef double [:, :, ::1] Chain_IFromJ = np.full((nr, nr, nr), NEGINF, dtype=np.float, order="c")
    cdef double [:, :, ::1] Chain_JFromX = np.full((nr, nr, nr), NEGINF, dtype=np.float, order="c")
    cdef double [:, :, ::1] Chain_IFromX = np.full((nr, nr, nr), NEGINF, dtype=np.float, order="c")
    cdef double [:, :, :, :, :, ::1] N = np.full((nr, nr, nr, 2, 2, 2), NEGINF, dtype=np.float, order="c")
    cdef double [:, :, :, :, :, ::1] L = np.full((nr, nr, nr, 2, 2, 2), NEGINF, dtype=np.float, order="c")
    cdef double [:, :, :, :, :, ::1] R = np.full((nr, nr, nr, 2, 2, 2), NEGINF, dtype=np.float, order="c")
    cdef double [:, :, ::1] L_XFromI = np.full((nr, nr, nr), NEGINF, dtype=np.float, order="c")
    cdef double [:, :, ::1] L_IFromX = np.full((nr, nr, nr), NEGINF, dtype=np.float, order="c")
    cdef double [:, :, ::1] L_JFromX = np.full((nr, nr, nr), NEGINF, dtype=np.float, order="c")
    cdef double [:, :, ::1] L_JFromI = np.full((nr, nr, nr), NEGINF, dtype=np.float, order="c")
    cdef double [:, :, ::1] R_XFromJ = np.full((nr, nr, nr), NEGINF, dtype=np.float, order="c")
    cdef double [:, :, ::1] R_JFromX = np.full((nr, nr, nr), NEGINF, dtype=np.float, order="c")
    cdef double [:, :, ::1] R_IFromX = np.full((nr, nr, nr), NEGINF, dtype=np.float, order="c")
    cdef double [:, :, ::1] R_IFromJ = np.full((nr, nr, nr), NEGINF, dtype=np.float, order="c")

    cdef int [:, :, :, :, ::1] backtrack = np.full((TYPE_TOTAL, nr, nr, nr, 22), -1, dtype=np.int32, order="c")
    cdef int cand_bt[22]
    cdef int [::1] heads
    cdef int [:, ::1] traces

    for i in range(nr):
        Int[i, i, 0, 1] = 0.
        Int[i, i, 1, 0] = 0.
        Int[i, i, 0, 0] = 0.
    for i in range(nr - 1):
        Int[i, i + 1, 0, 0] = 0.
    for g in range(nr):
        for h in range(g + 1, nr):
            TriG[h, h, g, 0] = 0.
            TriG[h, h, g, 1] = 0.
        for h in range(g):
            TriG[h, h, g, 0] = 0.
            TriG[h, h, g, 1] = 0.

    for d in range(nr):
        for i in range(nr - d):
            j = i + d
            # Chain
            cand = NEGINF
            for idx in range(22): cand_bt[idx] = -1
            for k in range(i + 1, j):
                tmp = CE(i, k, scores) + LR[i, k, j, 0] + Int[k, j, 0, 0]
                if tmp > cand:
                    cand = tmp
                    cand_bt[:8] = TYPE_LR, i, k, j, TYPE_INT, k, j, 0
                    cand_bt[8:12] = -1, -1, -1, -1
                    cand_bt[12:17] = TYPE_CE, -1, i, k, -1
                    cand_bt[17:22] = -1, -1, -1, -1, -1
                for l in range(i + 1, k):
                    tmp = CE(i, k, scores) + CE(j, l, scores) + Int[i, l, 0, 0] + L[l, k, i, 0, 0, 0] + N[k, j, l, 0, 0, 0]
                    if tmp > cand:
                        cand = tmp
                        cand_bt[:12] = TYPE_INT, i, l, 0, TYPE_L, l, k, i, TYPE_N, k, j, l
                        cand_bt[12:22] = TYPE_CE, -1, i, k, -1, TYPE_CE, -1, j, l, -1
                    tmp = CE(i, k, scores) + CE(j, l, scores) + Int[i, l, 0, 0] + Int[l, k, 0, 0] + L[k, j, l, 0, 0, 0]
                    if tmp > cand:
                        cand = tmp
                        cand_bt[:12] = TYPE_INT, i, l, 0, TYPE_INT, l, k, 0, TYPE_L, k, j, l
                        cand_bt[12:22] = TYPE_CE, -1, i, k, -1, TYPE_CE, -1, j, l, -1
                    tmp = CE(i, k, scores) + CE(j, l, scores) + R[i, l, k, 0, 0, 0] + Int[l, k, 0, 0] + L[k, j, l, 0, 0, 0]
                    if tmp > cand:
                        cand = tmp
                        cand_bt[:12] = TYPE_R, i, l, k, TYPE_INT, l, k, 0, TYPE_L, k, j, l
                        cand_bt[12:22] = TYPE_CE, -1, i, k, -1, TYPE_CE, -1, j, l, -1
                for l in range(k + 1, j):
                    tmp = CE(i, k, scores) + R[i, k, l, 0, 0, 0] + Int[k, l, 0, 0] + Page1[l, j, k]
                    if tmp > cand:
                        cand = tmp
                        cand_bt[:12] = TYPE_R, i, k, l, TYPE_INT, k, l, 0, TYPE_PAGE1, l, j, k
                        cand_bt[12:17] = TYPE_CE, -1, i, k, -1
            if cand > Chain[i, j]:
                Chain[i, j] = cand
                _k = TYPE_CHAIN
                for idx in range(22): backtrack[_k, i, j, 0, idx] = cand_bt[idx]

            # TriFarCrossed - L
            cand = NEGINF
            for idx in range(22): cand_bt[idx] = -1
            for k in range(i + 1, j):
                for b in range(1, 4):
                    bl = 1 if b == 1 else 0
                    bm = 1 if b == 2 else 0
                    br = 1 if b == 3 else 0
                    for l in range(k + 1, j + 1):
                        tmp = CE(i, k, scores) + R[i, k, l, 0, 0, bl] + Int[k, l, 0, bm] + L[l, j, k, br, 1, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:12] = idx_type(TYPE_R, 0, 0, bl), i, k, l, idx_type(TYPE_INT, 0, bm), k, l, 0, idx_type(TYPE_L, br, 1, 0), l, j, k
                            cand_bt[12:17] = TYPE_CE, -1, i, k, -1
                        tmp = CE(i, k, scores) + LR[i, k, l, bl] + Int[k, l, 0, bm] + Int[l, j, br, 1]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:12] = idx_type(TYPE_LR, bl), i, k, l, idx_type(TYPE_INT, 0, bm), k, l, 0, idx_type(TYPE_INT, br, 1), l, j, 0
                            cand_bt[12:17] = TYPE_CE, -1, i, k, -1
                    for l in range(i + 1, k):
                        tmp = CE(i, k, scores) + Int[i, l, 0, bl] + L[l, k, i, bm, 0, 0] + N[k, j, l, 0, 1, br]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:12] = idx_type(TYPE_INT, 0, bl), i, l, 0, idx_type(TYPE_L, bm, 0, 0), l, k, i, idx_type(TYPE_N, 0, 1, br), k, j, l
                            cand_bt[12:17] = TYPE_CE, -1, i, k, -1
                        tmp = CE(i, k, scores) + Int[i, l, 0, bl] + Int[l, k, bm, 0] + L[k, j, l, 0, 1, br]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:12] = idx_type(TYPE_INT, 0, bl), i, l, 0, idx_type(TYPE_INT, bm, 0), l, k, 0, idx_type(TYPE_L, 0, 1, br), k, j, l
                            cand_bt[12:17] = TYPE_CE, -1, i, k, -1
                        tmp = CE(i, k, scores) + R[i, l, k, 0, bl, 0] + Int[l, k, bm, 0] + L[k, j, l, 0, 1, br]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:12] = idx_type(TYPE_R, 0, bl, 0), i, l, k, idx_type(TYPE_INT, bm, 0), l, k, 0, idx_type(TYPE_L, 0, 1, br), k, j, l
                            cand_bt[12:17] = TYPE_CE, -1, i, k, -1
            if cand > TriFarCrossed[i, j, 0]:
                TriFarCrossed[i, j, 0] = cand
                _k = TYPE_TRIFARCROSSED
                for idx in range(22): backtrack[_k, i, j, 0, idx] = cand_bt[idx]

            # TriFarCrossed - R
            cand = NEGINF
            for idx in range(22): cand_bt[idx] = -1
            for k in range(i + 1, j):
                for b in range(1, 4):
                    bl = 1 if b == 1 else 0
                    bm = 1 if b == 2 else 0
                    br = 1 if b == 3 else 0
                    for l in range(i, k):
                        tmp = CE(j, k, scores) + L[k, j, l, 0, 0, bl] + Int[l, k, bm, 0] + R[i, l, k, 1, br, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:12] = idx_type(TYPE_L, 0, 0, bl), k, j, l, idx_type(TYPE_INT, bm, 0), l, k, 0, idx_type(TYPE_R, 1, br, 0), i, l, k
                            cand_bt[12:17] = TYPE_CE, -1, j, k, -1
                        tmp = CE(j, k, scores) + LR[k, j, l, bl] + Int[l, k, bm, 0] + Int[i, l, 1, br]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:12] = idx_type(TYPE_LR, bl), k, j, l, idx_type(TYPE_INT, bm, 0), l, k, 0, idx_type(TYPE_INT, 1, br), i, l, 0
                            cand_bt[12:17] = TYPE_CE, -1, j, k, -1
                    for l in range(k + 1, j):
                        tmp = CE(j, k, scores) + Int[l, j, bl, 0] + R[k, l, j, 0, bm, 0] + N[i, k, l, 1, 0, br]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:12] = idx_type(TYPE_INT, bl, 0), l, j, 0, idx_type(TYPE_R, 0, bm, 0), k, l, j, idx_type(TYPE_N, 1, 0, br), i, k, l
                            cand_bt[12:17] = TYPE_CE, -1, j, k, -1
                        tmp = CE(j, k, scores) + Int[l, j, bl, 0] + Int[k, l, 0, bm] + R[i, k, l, 1, 0, br]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:12] = idx_type(TYPE_INT, bl, 0), l, j, 0, idx_type(TYPE_INT, 0, bm), k, l, 0, idx_type(TYPE_R, 1, 0, br), i, k, l
                            cand_bt[12:17] = TYPE_CE, -1, j, k, -1
                        tmp = CE(j, k, scores) + L[l, j, k, bl, 0, 0] + Int[k, l, 0, bm] + R[i, k, l, 1, 0, br]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:12] = idx_type(TYPE_L, bl, 0, 0), l, j, k, idx_type(TYPE_INT, 0, bm), k, l, 0, idx_type(TYPE_R, 1, 0, br), i, k, l
                            cand_bt[12:17] = TYPE_CE, -1, j, k, -1
            if cand > TriFarCrossed[i, j, 1]:
                TriFarCrossed[i, j, 1] = cand
                _k = TYPE_TRIFARCROSSED + 1
                for idx in range(22): backtrack[_k, i, j, 0, idx] = cand_bt[idx]

            # OneFarCrossedG - L
            for g in range(nr):
                if g < i or g > j:
                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    for k in range(i, j):
                        tmp = TriG[i, k, g, 0] + Chain[k, j]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = TYPE_TRIG, i, k, g, TYPE_CHAIN, k, j, 0
                    if cand > OneFarCrossedG[i, j, g, 0]:
                        OneFarCrossedG[i, j, g, 0] = cand
                        _k = TYPE_ONEFARCROSSEDG
                        for idx in range(22): backtrack[_k, i, j, g, idx] = cand_bt[idx]

            # OneFarCrossedG - R
            for g in range(nr):
                if g < i or g > j:
                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    for k in range(i + 1, j + 1):
                        tmp = TriG[k, j, g, 1] + Chain[i, k]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = TYPE_TRIG, k, j, g, TYPE_CHAIN, i, k, 0
                    if cand > OneFarCrossedG[i, j, g, 1]:
                        OneFarCrossedG[i, j, g, 1] = cand
                        _k = TYPE_ONEFARCROSSEDG + 1
                        for idx in range(22): backtrack[_k, i, j, g, idx] = cand_bt[idx]

            # LeftFarCrossed
            cand = NEGINF
            for idx in range(22): cand_bt[idx] = -1
            for k in range(i, j + 1):
                tmp = Chain[i, k] + Int[k, j, 1, 0]
                if tmp > cand:
                    cand = tmp
                    cand_bt[:8] = TYPE_CHAIN, i, k, 0, idx_type(TYPE_INT, 1, 0), k, j, 0
            if cand > LeftFarCrossed[i, j]:
                LeftFarCrossed[i, j] = cand
                _k = TYPE_LEFTFARCROSSED
                for idx in range(22): backtrack[_k, i, j, 0, idx] = cand_bt[idx]

            # TrapG
            for g in range(nr):
                if g < i or g > j:
                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    if i + 1 < nr:
                        tmp = GS(g, i, j, 0, scores) + TriG[i + 1, j, i, 1]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:4] = TYPE_TRIG + 1, i + 1, j, i
                            cand_bt[4:8] = -1, -1, -1, -1
                            cand_bt[12:17] = TYPE_GS, g, i, j, 0
                    tmp = G(g, i, j, scores) + Chain[i, j]
                    if tmp > cand:
                        cand = tmp
                        cand_bt[:4] = TYPE_CHAIN, i, j, 0
                        cand_bt[4:8] = -1, -1, -1, -1
                        cand_bt[12:17] = TYPE_G, g, i, j, -1
                    for k in range(i + 1, j):
                        tmp = GS(g, i, j, k, scores) + TrapG[i, k, g, 0] + BoxG[k, j, i]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = TYPE_TRAPG, i, k, g, TYPE_BOXG, k, j, i
                            cand_bt[12:17] = TYPE_GS, g, i, j, k
                        tmp = G(g, i, j, scores) + TriFarCrossed[i, k, 0] + TriG[k + 1, j, i, 1]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = TYPE_TRIFARCROSSED, i, k, 0, TYPE_TRIG + 1, k + 1, j, i
                            cand_bt[12:17] = TYPE_G, g, i, j, -1
                        tmp = G(g, i, j, scores) + Chain[i, k] + TriG[k, j, i, 1]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = TYPE_CHAIN, i, k, 0, TYPE_TRIG + 1, k, j, i
                            cand_bt[12:17] = TYPE_G, g, i, j, -1
                    if cand > TrapG[i, j, g, 0]:
                        TrapG[i, j, g, 0] = cand
                        _k = TYPE_TRAPG
                        for idx in range(22): backtrack[_k, i, j, g, idx] = cand_bt[idx]

                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    if j > 0:
                        tmp = GS(g, j, i, 0, scores) + TriG[i, j - 1, j, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:4] = TYPE_TRIG, i, j - 1, j
                            cand_bt[4:8] = -1, -1, -1, -1
                            cand_bt[12:17] = TYPE_GS, g, j, i, 0
                    tmp = G(g, j, i, scores) + Chain[i, j]
                    if tmp > cand:
                        cand = tmp
                        cand_bt[:4] = TYPE_CHAIN, i, j, 0
                        cand_bt[4:8] = -1, -1, -1, -1
                        cand_bt[12:17] = TYPE_G, g, j, i, -1
                    for k in range(i + 1, j):
                        tmp = GS(g, j, i, k, scores) + TrapG[k, j, g, 1] + BoxG[i, k, j]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = TYPE_TRAPG + 1, k, j, g, TYPE_BOXG, i, k, j
                            cand_bt[12:17] = TYPE_GS, g, j, i, k
                        tmp = G(g, j, i, scores) + TriFarCrossed[k, j, 1] + TriG[i, k - 1, j, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = TYPE_TRIFARCROSSED + 1, k, j, 0, TYPE_TRIG, i, k - 1, j
                            cand_bt[12:17] = TYPE_G, g, j, i, -1
                        tmp = G(g, j, i, scores) + Chain[k, j] + TriG[i, k, j, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = TYPE_CHAIN, k, j, 0, TYPE_TRIG, i, k, j
                            cand_bt[12:17] = TYPE_G, g, j, i, -1
                    if cand > TrapG[i, j, g, 1]:
                        TrapG[i, j, g, 1] = cand
                        _k = TYPE_TRAPG + 1
                        for idx in range(22): backtrack[_k, i, j, g, idx] = cand_bt[idx]

            # Trap
            cand = NEGINF
            for idx in range(22): cand_bt[idx] = -1
            if i + 1 < nr:
                tmp = S(i, j, 0, scores) + TriG[i + 1, j, i, 1]
                if tmp > cand:
                    cand = tmp
                    cand_bt[:4] = TYPE_TRIG + 1, i + 1, j, i
                    cand_bt[4:8] = -1, -1, -1, -1
                    cand_bt[12:17]= TYPE_S, -1, i, j, 0
            tmp = E(i, j, scores) + Chain[i, j]
            if tmp > cand:
                cand = tmp
                cand_bt[:4] = TYPE_CHAIN, i, j, 0
                cand_bt[4:8] = -1, -1, -1, -1
                cand_bt[12:17]= TYPE_E, -1, i, j, -1
            for k in range(i + 1, j):
                tmp = S(i, j, k, scores) + Trap[i, k, 0] + BoxG[k, j, i]
                if tmp > cand:
                    cand = tmp
                    cand_bt[:8] = TYPE_TRAP, i, k, 0, TYPE_BOXG, k, j, i
                    cand_bt[12:17]= TYPE_S, -1, i, j, k
                tmp = E(i, j, scores) + TriFarCrossed[i, k, 0] + TriG[k + 1, j, i, 1]
                if tmp > cand:
                    cand = tmp
                    cand_bt[:8] = TYPE_TRIFARCROSSED, i, k, 0, TYPE_TRIG + 1, k + 1, j, i
                    cand_bt[12:17]= TYPE_E, -1, i, j, -1
                tmp = E(i, j, scores) + Chain[i, k] + TriG[k, j, i, 1]
                if tmp > cand:
                    cand = tmp
                    cand_bt[:8] = TYPE_CHAIN, i, k, 0, TYPE_TRIG + 1, k, j, i
                    cand_bt[12:17]= TYPE_E, -1, i, j, -1
            if cand > Trap[i, j, 0]:
                Trap[i, j, 0] = cand
                _k = TYPE_TRAP
                for idx in range(22): backtrack[_k, i, j, 0, idx] = cand_bt[idx]

            cand = NEGINF
            for idx in range(22): cand_bt[idx] = -1
            if j > 0:
                tmp = S(j, i, 0, scores) + TriG[i, j - 1, j, 0]
                if tmp > cand:
                    cand = tmp
                    cand_bt[:4] = TYPE_TRIG, i, j - 1, j
                    cand_bt[4:8] = -1, -1, -1, -1
                    cand_bt[12:17]= TYPE_S, -1, j, i, 0
            tmp = E(j, i, scores) + Chain[i, j]
            if tmp > cand:
                cand = tmp
                cand_bt[:4] = TYPE_CHAIN, i, j, 0
                cand_bt[4:8] = -1, -1, -1, -1
                cand_bt[12:17]= TYPE_E, -1, j, i, -1
            for k in range(i + 1, j):
                tmp = S(j, i, k, scores) + Trap[k, j, 1] + BoxG[i, k, j]
                if tmp > cand:
                    cand = tmp
                    cand_bt[:8] = TYPE_TRAP + 1, k, j, 0, TYPE_BOXG, i, k, j
                    cand_bt[12:17]= TYPE_S, -1, j, i, k
                tmp = E(j, i, scores) + TriFarCrossed[k, j, 1] + TriG[i, k - 1, j, 0]
                if tmp > cand:
                    cand = tmp
                    cand_bt[:8] = TYPE_TRIFARCROSSED + 1, k, j, 0, TYPE_TRIG, i, k - 1, j
                    cand_bt[12:17]= TYPE_E, -1, j, i, -1
                tmp = E(j, i, scores) + Chain[k, j] + TriG[i, k, j, 0]
                if tmp > cand:
                    cand = tmp
                    cand_bt[:8] = TYPE_CHAIN, k, j, 0, TYPE_TRIG, i, k, j
                    cand_bt[12:17]= TYPE_E, -1, j, i, -1
            if cand > Trap[i, j, 1]:
                Trap[i, j, 1] = cand
                _k = TYPE_TRAP + 1
                for idx in range(22): backtrack[_k, i, j, 0, idx] = cand_bt[idx]

            # TriG
            for g in range(nr):
                if g < i or g > j:
                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    for k in range(i + 1, j + 1):
                        tmp = TrapG[i, k, g, 0] + TriG[k, j, i, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = TYPE_TRAPG, i, k, g, TYPE_TRIG, k, j, i
                    tmp = TriFarCrossed[i, j, 0]
                    if tmp > cand:
                        cand = tmp
                        cand_bt[:4] = TYPE_TRIFARCROSSED, i, j, 0
                        cand_bt[4:8] = -1, -1, -1, -1
                    if cand > TriG[i, j, g, 0]:
                        TriG[i, j, g, 0] = cand
                        _k = TYPE_TRIG
                        for idx in range(22): backtrack[_k, i, j, g, idx] = cand_bt[idx]

                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    for k in range(i, j):
                        tmp = TriG[i, k, j, 1] + TrapG[k, j, g, 1]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = TYPE_TRIG + 1, i, k, j, TYPE_TRAPG + 1, k, j, g
                    tmp = TriFarCrossed[i, j, 1]
                    if tmp > cand:
                        cand = tmp
                        cand_bt[:4] = TYPE_TRIFARCROSSED + 1, i, j, 0
                        cand_bt[4:8] = -1, -1, -1, -1
                    if cand > TriG[i, j, g, 1]:
                        TriG[i, j, g, 1] = cand
                        _k = TYPE_TRIG + 1
                        for idx in range(22): backtrack[_k, i, j, g, idx] = cand_bt[idx]

            # Tri
            cand = NEGINF
            for idx in range(22): cand_bt[idx] = -1
            for k in range(i + 1, j + 1):
                tmp = Trap[i, k, 0] + TriG[k, j, i, 0]
                if tmp > cand:
                    cand = tmp
                    cand_bt[:8] = TYPE_TRAP, i, k, 0, TYPE_TRIG, k, j, i
            tmp = TriFarCrossed[i, j, 0]
            if tmp > cand:
                cand = tmp
                cand_bt[:4] = TYPE_TRIFARCROSSED, i, j, 0
                cand_bt[4:8] = -1, -1, -1, -1
            if cand > Int[i, j, 0, 1]:
                Int[i, j, 0, 1] = cand
                _k = idx_type(TYPE_INT, 0, 1)
                for idx in range(22): backtrack[_k, i, j, 0, idx] = cand_bt[idx]

            cand = NEGINF
            for idx in range(22): cand_bt[idx] = -1
            for k in range(i, j):
                tmp = TriG[i, k, j, 1] + Trap[k, j, 1]
                if tmp > cand:
                    cand = tmp
                    cand_bt[:8] = TYPE_TRIG + 1, i, k, j, TYPE_TRAP + 1, k, j, 0
            tmp = TriFarCrossed[i, j, 1]
            if tmp > cand:
                cand = tmp
                cand_bt[:4] = TYPE_TRIFARCROSSED + 1, i, j, 0
                cand_bt[4:8] = -1, -1, -1, -1
            if cand > Int[i, j, 1, 0]:
                Int[i, j, 1, 0] = cand
                _k = idx_type(TYPE_INT, 1, 0)
                for idx in range(22): backtrack[_k, i, j, 0, idx] = cand_bt[idx]

            # BoxG
            for g in range(nr):
                if g < i or g > j:
                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    for k in range(i, j):
                        tmp = TriG[i, k, g, 0] + TriG[k + 1, j, g, 1]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = TYPE_TRIG, i, k, g, TYPE_TRIG + 1, k + 1, j, g
                    for k in range(i, j):
                        tmp = TriG[i, k, g, 0] + OneFarCrossedG[k, j, g, 1]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = TYPE_TRIG, i, k, g, TYPE_ONEFARCROSSEDG + 1, k, j, g
                    if cand > BoxG[i, j, g]:
                        BoxG[i, j, g] = cand
                        _k = TYPE_BOXG
                        for idx in range(22): backtrack[_k, i, j, g, idx] = cand_bt[idx]

            # TwoRooted
            cand = NEGINF
            for idx in range(22): cand_bt[idx] = -1
            for k in range(i, j):
                tmp = Int[i, k, 0, 1] + Int[k + 1, j, 1, 0]
                if tmp > cand:
                    cand = tmp
                    cand_bt[:8] = idx_type(TYPE_INT, 0, 1), i, k, 0, idx_type(TYPE_INT, 1, 0), k + 1, j, 0
                tmp = Int[i, k, 0, 1] + LeftFarCrossed[k, j]
                if tmp > cand:
                    cand = tmp
                    cand_bt[:8] = idx_type(TYPE_INT, 0, 1), i, k, 0, TYPE_LEFTFARCROSSED, k, j, 0
            if cand > Int[i, j, 0, 0]:
                Int[i, j, 0, 0] = cand
                _k = idx_type(TYPE_INT, 0, 0)
                for idx in range(22): backtrack[_k, i, j, 0, idx] = cand_bt[idx]

            # N - bi, bj, F
            for x in range(nr):
                if x < i or x > j:
                    for bj in range(2):
                        for bi in range(2):
                            start = vcross_start(i, j, bi, bj, TYPE_N)
                            end = vcross_end(i, j, bi, bj, TYPE_N)
                            cand = NEGINF
                            for idx in range(22): cand_bt[idx] = -1
                            for k in range(start, end + 1):
                                tmp = CE(x, k, scores) + Int[i, k, bi, 0] + Int[k, j, 0, bj]
                                if tmp > cand:
                                    cand = tmp
                                    cand_bt[:8] = idx_type(TYPE_INT, bi, 0), i, k, 0, idx_type(TYPE_INT, 0, bj), k, j, 0
                                    cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                                tmp = CE(x, k, scores) + N[i, k, x, bi, 0, 0] + Int[k, j, 0, bj]
                                if tmp > cand:
                                    cand = tmp
                                    cand_bt[:8] = idx_type(TYPE_N, bi, 0, 0), i, k, x, idx_type(TYPE_INT, 0, bj), k, j, 0
                                    cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                            if cand > N[i, j, x, bi, bj, 0]:
                                N[i, j, x, bi, bj, 0] = cand
                                _k = idx_type(TYPE_N, bi, bj, 0)
                                for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

            # N - bi, bj, T
            for x in range(nr):
                if x < i or x > j:
                    for bi in range(2):
                        for bj in range(2):
                            start = vcross_start(i, j, bi, bj, TYPE_N)
                            end = vcross_end(i, j, bi, bj, TYPE_N)
                            cand = NEGINF
                            for idx in range(22): cand_bt[idx] = -1
                            for k in range(start, end + 1):
                                tmp = CE(k, x, scores) + Int[i, k, bi, 1] + Int[k, j, 0, bj]
                                if tmp > cand:
                                    cand = tmp
                                    cand_bt[:8] = idx_type(TYPE_INT, bi, 1), i, k, 0, idx_type(TYPE_INT, 0, bj), k, j, 0
                                    cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                                tmp = CE(k, x, scores) + Int[i, k, bi, 0] + Int[k, j, 1, bj]
                                if tmp > cand:
                                    cand = tmp
                                    cand_bt[:8] = idx_type(TYPE_INT, bi, 0), i, k, 0, idx_type(TYPE_INT, 1, bj), k, j, 0
                                    cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                                tmp = CE(k, x, scores) + N[i, k, x, bi, 0, 0] + Int[k, j, 1, bj]
                                if tmp > cand:
                                    cand = tmp
                                    cand_bt[:8] = idx_type(TYPE_N, bi, 0, 0), i, k, x, idx_type(TYPE_INT, 1, bj), k, j, 0
                                    cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                                tmp = CE(k, x, scores) + Int[i, k, bi, 1] + N[k, j, x, 0, bj, 0]
                                if tmp > cand:
                                    cand = tmp
                                    cand_bt[:8] = idx_type(TYPE_INT, bi, 1), i, k, 0, idx_type(TYPE_N, 0, bj), k, j, x
                                    cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                            if cand > N[i, j, x, bi, bj, 1]:
                                N[i, j, x, bi, bj, 1] = cand
                                _k = idx_type(TYPE_N, bi, bj, 1)
                                for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

            # L_FFF
            for x in range(nr):
                if x < i or x > j:
                    bi = 0
                    bj = 0
                    start = vcross_start(i, j, bi, bj, TYPE_L)
                    end = vcross_end(i, j, bi, bj, TYPE_L)
                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    for k in range(start, end + 1):
                        tmp = CE(x, k, scores) + Int[i, k, bi, 0] + Int[k, j, 0, bj]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, bi, 0), i, k, 0, idx_type(TYPE_INT, 0, bj), k, j, 0
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(x, k, scores) + Int[i, k, 0, 0] + L[k, j, i, 0, bj, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, 0, 0), i, k, 0, idx_type(TYPE_L, 0, bj, 0), k, j, i
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(x, k, scores) + L[i, k, x, bi, 0, 0] + Int[k, j, 0, bj]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_L, bi, 0, 0), i, k, x, idx_type(TYPE_INT, 0, bj), k, j, 0
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(i, k, scores) + L[i, k, x, 0, 0, 0] + Int[k, j, 0, bj]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_L, 0, 0, 0), i, k, x, idx_type(TYPE_INT, 0, bj), k, j, 0
                            cand_bt[12:17] = TYPE_CE, -1, i, k, -1
                    if cand > L[i, j, x, bi, bj, 0]:
                        L[i, j, x, bi, bj, 0] = cand
                        _k = idx_type(TYPE_L, bi, bj, 0)
                        for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

            # L_XfromI
            for x in range(nr):
                if x < i or x > j:
                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    for k in range(i + 1, j):
                        tmp = CE(k, x, scores) + Int[i, k, 0, 1] + Int[k, j, 0, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, 0, 1), i, k, 0, idx_type(TYPE_INT, 0, 0), k, j, 0
                            cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                        tmp = CE(k, x, scores) + Int[i, k, 0, 1] + L[k, j, i, 0, 0, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, 0, 1), i, k, 0, idx_type(TYPE_L, 0, 0, 0), k, j, i
                            cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                        tmp = CE(k, x, scores) + Int[i, k, 0, 0] + L_IFromX[k, j, i]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, 0, 1), i, k, 0, TYPE_L_IFROMX, k, j, i
                            cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                        tmp = CE(k, x, scores) + L_JFromI[i, k, x] + Int[k, j, 0, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = TYPE_L_JFROMI, i, k, x, idx_type(TYPE_INT, 0, 0), k, j, 0
                            cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                        tmp = CE(x, k, scores) + L_XFromI[i, k, x] + Int[k, j, 0, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = TYPE_L_XFROMI, i, k, x, idx_type(TYPE_INT, 0, 0), k, j, 0
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(i, k, scores) + L[i, k, x, 0, 0, 1] + Int[k, j, 0, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_L, 0, 0, 1), i, k, x, idx_type(TYPE_INT, 0, 0), k, j, 0
                            cand_bt[12:17] = TYPE_CE, -1, i, k, -1
                    if cand > L_XFromI[i, j, x]:
                        L_XFromI[i, j, x] = cand
                        _k = TYPE_L_XFROMI
                        for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

            # L_IfromX
            for x in range(nr):
                if x < i or x > j:
                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    for k in range(i + 1, j):
                        tmp = CE(x, k, scores) + Int[i, k, 1, 0] + Int[k, j, 0, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, 1, 0), i, k, 0, idx_type(TYPE_INT, 0, 0), k, j, 0
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(x, k, scores) + Int[i, k, 1, 0] + L[k, j, i, 0, 0, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, 1, 0), i, k, 0, idx_type(TYPE_L, 0, 0, 0), k, j, i
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(x, k, scores) + Int[i, k, 0, 0] + L_XFromI[k, j, i]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, 0, 0), i, k, 0, TYPE_L_XFROMI, k, j, i
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(x, k, scores) + L[i, k, x, 1, 0, 0] + Int[k, j, 0, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_L, 1, 0, 0), i, k, x, idx_type(TYPE_INT, 0, 0), k, j, 0
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(k, i, scores) + L_JFromX[i, k, x] + Int[k, j, 0, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = TYPE_L_JFROMX, i, k, x, idx_type(TYPE_INT, 0, 0), k, j, 0
                            cand_bt[12:17] = TYPE_CE, -1, k, i, -1
                        tmp = CE(i, k, scores) + L_IFromX[i, k, x] + Int[k, j, 0, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = TYPE_L_IFROMX, i, k, x, idx_type(TYPE_INT, 0, 0), k, j, 0
                            cand_bt[12:17] = TYPE_CE, -1, i, k, -1
                    if cand > L_IFromX[i, j, x]:
                        L_IFromX[i, j, x] = cand
                        _k = TYPE_L_IFROMX
                        for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

            # L_JfromX
            for x in range(nr):
                if x < i or x > j:
                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    for k in range(i + 1, j + 1):
                        tmp = CE(x, k, scores) + Int[i, k, 0, 0] + Int[k, j, 0, 1]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, 0, 0), i, k, 0, idx_type(TYPE_INT, 0, 1), k, j, 0
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(x, k, scores) + Int[i, k, 0, 0] + L_JFromI[k, j, i]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, 0, 0), i, k, 0, TYPE_L_JFROMI, k, j, i
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(x, k, scores) + L[i, k, x, 0, 0, 0] + Int[k, j, 0, 1]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_L, 0, 0, 0), i, k, x, idx_type(TYPE_INT, 0, 1), k, j, 0
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                    if cand > L_JFromX[i, j, x]:
                        L_JFromX[i, j, x] = cand
                        _k = TYPE_L_JFROMX
                        for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

            # L_JfromI
            for x in range(nr):
                if x < i or x > j:
                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    for k in range(i + 1, j):
                        tmp = CE(x, k, scores) + Int[i, k, 0, 0] + L_JFromX[k, j, i]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, 0, 0), i, k, 0, TYPE_L_JFROMX, k, j, i
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                    for k in range(i + 1, j + 1):
                        tmp = CE(i, k, scores) + L[i, k, x, 0, 0, 0] + Int[k, j, 0, 1]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_L, 0, 0, 0), i, k, x, idx_type(TYPE_INT, 0, 1), k, j, 0
                            cand_bt[12:17] = TYPE_CE, -1, i, k, -1
                    if cand > L_JFromI[i, j, x]:
                        L_JFromI[i, j, x] = cand
                        _k = TYPE_L_JFROMI
                        for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

            # L_FTF
            for x in range(nr):
                if x < i or x > j:
                    bi = 0
                    bj = 1
                    start = vcross_start(i, j, bi, bj, TYPE_L)
                    end = vcross_end(i, j, bi, bj, TYPE_L)
                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    for k in range(start, end + 1):
                        tmp = CE(x, k, scores) + Int[i, k, bi, 0] + Int[k, j, 0, bj]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, bi, 0), i, k, 0, idx_type(TYPE_INT, 0, bj), k, j, 0
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(x, k, scores) + Int[i, k, 0, 0] + L[k, j, i, 0, bj, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, 0, 0), i, k, 0, idx_type(TYPE_L, 0, bj, 0), k, j, i
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(x, k, scores) + L[i, k, x, bi, 0, 0] + Int[k, j, 0, bj]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_L, bi, 0, 0), i, k, x, idx_type(TYPE_INT, 0, bj), k, j, 0
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(i, k, scores) + L[i, k, x, 0, 0, 0] + Int[k, j, 0, bj]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_L, 0, 0, 0), i, k, x, idx_type(TYPE_INT, 0, bj), k, j, 0
                            cand_bt[12:17] = TYPE_CE, -1, i, k, -1
                    if cand > L[i, j, x, bi, bj, 0]:
                        L[i, j, x, bi, bj, 0] = cand
                        _k = idx_type(TYPE_L, bi, bj, 0)
                        for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

            # L_TbjF
            for x in range(nr):
                if x < i or x > j:
                    bi = 1
                    for bj in range(2):
                        start = vcross_start(i, j, bi, bj, TYPE_L)
                        end = vcross_end(i, j, bi, bj, TYPE_L)
                        cand = NEGINF
                        for idx in range(22): cand_bt[idx] = -1
                        for k in range(start, end + 1):
                            tmp = CE(x, k, scores) + Int[i, k, bi, 0] + Int[k, j, 0, bj]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = idx_type(TYPE_INT, bi, 0), i, k, 0, idx_type(TYPE_INT, 0, bj), k, j, 0
                                cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                            tmp = CE(x, k, scores) + Int[i, k, 1, 0] + L[k, j, i, 0, bj, 0]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = idx_type(TYPE_INT, 1, 0), i, k, 0, idx_type(TYPE_L, 0, bj, 0), k, j, i
                                cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                            tmp = CE(x, k, scores) + Int[i, k, 0, 0] + L[k, j, i, 0, bj, 1]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = idx_type(TYPE_INT, 0, 0), i, k, 0, idx_type(TYPE_L, 0, bj, 1), k, j, i
                                cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                            tmp = CE(x, k, scores) + L[i, k, x, bi, 0, 0] + Int[k, j, 0, bj]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = idx_type(TYPE_L, bi, 0, 0), i, k, x, idx_type(TYPE_INT, 0, bj), k, j, 0
                                cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                            tmp = CE(i, k, scores) + L_IFromX[i, k, x] + Int[k, j, 0, bj]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = TYPE_L_IFROMX, i, k, x, idx_type(TYPE_INT, 0, bj), k, j, 0
                                cand_bt[12:17] = TYPE_CE, -1, i, k, -1
                            tmp = CE(k, i, scores) + L_JFromX[i, k, x] + Int[k, j, 0, bj]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = TYPE_L_JFROMX, i, k, x, idx_type(TYPE_INT, 0, bj), k, j, 0
                                cand_bt[12:17] = TYPE_CE, -1, k, i, -1
                            tmp = CE(k, i, scores) + L[i, k, x, 0, 0, 0] + Int[k, j, 1, bj]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = idx_type(TYPE_L, 0, 0, 0), i, k, x, idx_type(TYPE_INT, 1, bj), k, j, 0
                                cand_bt[12:17] = TYPE_CE, -1, k, i, -1
                        if cand > L[i, j, x, bi, bj, 0]:
                            L[i, j, x, bi, bj, 0] = cand
                            _k = idx_type(TYPE_L, bi, bj, 0)
                            for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

            # L_FbjT
            for x in range(nr):
                if x < i or x > j:
                    bi = 0
                    for bj in range(2):
                        start = vcross_start(i, j, bi, bj, TYPE_L)
                        end = vcross_end(i, j, bi, bj, TYPE_L)
                        cand = NEGINF
                        for idx in range(22): cand_bt[idx] = -1
                        for k in range(start, end + 1):
                            tmp = CE(k, x, scores) + Int[i, k, bi, 1] + Int[k, j, 0, bj]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = idx_type(TYPE_INT, bi, 1), i, k, 0, idx_type(TYPE_INT, 0, bj), k, j, 0
                                cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                            tmp = CE(k, x, scores) + Int[i, k, bi, 0] + Int[k, j, 1, bj]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = idx_type(TYPE_INT, bi, 0), i, k, 0, idx_type(TYPE_INT, 1, bj), k, j, 0
                                cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                            tmp = CE(k, x, scores) + Int[i, k, bi, 0] + L[k, j, i, 1, bj, 0]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = idx_type(TYPE_INT, bi, 0), i, k, 0, idx_type(TYPE_L, 1, bj, 0), k, j, i
                                cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                            tmp = CE(k, x, scores) + Int[i, k, bi, 1] + L[k, j, i, 0, bj, 0]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = idx_type(TYPE_INT, bi, 1), i, k, 0, idx_type(TYPE_L, 0, bj, 0), k, j, i
                                cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                            tmp = CE(k, x, scores) + L_JFromI[i, k, x] + Int[k, j, 0, bj]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = TYPE_L_JFROMI, i, k, x, idx_type(TYPE_INT, 0, bj), k, j, 0
                                cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                            tmp = CE(k, x, scores) + L[i, k, x, 0, 0, 0] + Int[k, j, 1, bj]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = idx_type(TYPE_L, 0, 0, 0), i, k, x, idx_type(TYPE_INT, 1, bj), k, j, 0
                                cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                            tmp = CE(x, k, scores) + L_XFromI[i, k, x] + Int[k, j, 0, bj]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = TYPE_L_XFROMI, i, k, x, idx_type(TYPE_INT, 0, bj), k, j, 0
                                cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                            tmp = CE(i, k, scores) + L[i, k, x, 0, 0, 1] + Int[k, j, 0, bj]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = idx_type(TYPE_L, 0, 0, 1), i, k, x, idx_type(TYPE_INT, 0, bj), k, j, 0
                                cand_bt[12:17] = TYPE_CE, -1, i, k, -1
                        if cand > L[i, j, x, bi, bj, 1]:
                            L[i, j, x, bi, bj, 1] = cand
                            _k = idx_type(TYPE_L, bi, bj, 1)
                            for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

            # R_FFF
            for x in range(nr):
                if x < i or x > j:
                    bj = 0
                    bi = 0
                    start = vcross_start(i, j, bi, bj, TYPE_R)
                    end = vcross_end(i, j, bi, bj, TYPE_R)
                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    for k in range(start, end + 1):
                        tmp = CE(x, k, scores) + Int[k, j, 0, bj] + Int[i, k, bi, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, 0, bj), k, j, 0, idx_type(TYPE_INT, bi, 0), i, k, 0
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(x, k, scores) + Int[k, j, 0, 0] + R[i, k, j, bi, 0, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, 0, 0), k, j, 0, idx_type(TYPE_R, bi, 0, 0), i, k, j
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(x, k, scores) + R[k, j, x, 0, bj, 0] + Int[i, k, bi, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_R, 0, bj, 0), k, j, x, idx_type(TYPE_INT, bi, 0), i, k, 0
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(j, k, scores) + R[k, j, x, 0, 0, 0] + Int[i, k, bi, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_R, 0, 0, 0), k, j, x, idx_type(TYPE_INT, bi, 0), i, k, 0
                            cand_bt[12:17] = TYPE_CE, -1, j, k, -1
                    if cand > R[i, j, x, bi, bj, 0]:
                        R[i, j, x, bi, bj, 0] = cand
                        _k = idx_type(TYPE_R, bi, bj, 0)
                        for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

            # R_XfromJ
            for x in range(nr):
                if x < i or x > j:
                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    for k in range(i + 1, j):
                        tmp = CE(k, x, scores) + Int[k, j, 1, 0] + Int[i, k, 0, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, 1, 0), k, j, 0, idx_type(TYPE_INT, 0, 0), i, k, 0
                            cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                        tmp = CE(k, x, scores) + Int[k, j, 1, 0] + R[i, k, j, 0, 0, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, 1, 0), k, j, 0, idx_type(TYPE_R, 0, 0, 0), i, k, j
                            cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                        tmp = CE(k, x, scores) + Int[k, j, 0, 0] + R_JFromX[i, k, j]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, 0, 0), k, j, 0, TYPE_R_JFROMX, i, k, j
                            cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                        tmp = CE(k, x, scores) + R_IFromJ[k, j, x] + Int[i, k, 0, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = TYPE_R_IFROMJ, k, j, x, idx_type(TYPE_INT, 0, 0), i, k, 0
                            cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                        tmp = CE(x, k, scores) + R_XFromJ[k, j, x] + Int[i, k, 0, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = TYPE_R_XFROMJ, k, j, x, idx_type(TYPE_INT, 0, 0), i, k, 0
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(j, k, scores) + R[k, j, x, 0, 0, 1] + Int[i, k, 0, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_R, 0, 0, 1), k, j, x, idx_type(TYPE_INT, 0, 0), i, k, 0
                            cand_bt[12:17] = TYPE_CE, -1, j, k, -1
                    if cand > R_XFromJ[i, j, x]:
                        R_XFromJ[i, j, x] = cand
                        _k = TYPE_R_XFROMJ
                        for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

            # R_JfromX
            for x in range(nr):
                if x < i or x > j:
                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    for k in range(i + 1, j):
                        tmp = CE(x, k, scores) + Int[k, j, 0, 1] + Int[i, k, 0, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, 0, 1), k, j, 0, idx_type(TYPE_INT, 0, 0), i, k, 0
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(x, k, scores) + Int[k, j, 0, 1] + R[i, k, j, 0, 0, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, 0, 1), k, j, 0, idx_type(TYPE_R, 0, 0, 0), i, k, j
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(x, k, scores) + Int[k, j, 0, 0] + R_XFromJ[i, k, j]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, 0, 0), k, j, 0, TYPE_R_XFROMJ, i, k, j
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(x, k, scores) + R[k, j, x, 0, 1, 0] + Int[i, k, 0, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_R, 0, 1, 0), k, j, x, idx_type(TYPE_INT, 0, 0), i, k, 0
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(k, j, scores) + R_IFromX[k, j, x] + Int[i, k, 0, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = TYPE_R_IFROMX, k, j, x, idx_type(TYPE_INT, 0, 0), i, k, 0
                            cand_bt[12:17] = TYPE_CE, -1, k, j, -1
                        tmp = CE(j, k, scores) + R_JFromX[k, j, x] + Int[i, k, 0, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = TYPE_R_JFROMX, k, j, x, idx_type(TYPE_INT, 0, 0), i, k, 0
                            cand_bt[12:17] = TYPE_CE, -1, j, k, -1
                    if cand > R_JFromX[i, j, x]:
                        R_JFromX[i, j, x] = cand
                        _k = TYPE_R_JFROMX
                        for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

            # R_IfromX
            for x in range(nr):
                if x < i or x > j:
                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    for k in range(i, j):
                        tmp = CE(x, k, scores) + Int[k, j, 0, 0] + Int[i, k, 1, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, 0, 0), k, j, 0, idx_type(TYPE_INT, 1, 0), i, k, 0
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(x, k, scores) + Int[k, j, 0, 0] + R_IFromJ[i, k, j]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, 0, 0), k, j, 0, TYPE_R_IFROMJ, i, k, j
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(x, k, scores) + R[k, j, x, 0, 0, 0] + Int[i, k, 1, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_R, 0, 0, 0), k, j, x, idx_type(TYPE_INT, 1, 0), i, k, 0
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                    if cand > R_IFromX[i, j, x]:
                        R_IFromX[i, j, x] = cand
                        _k = TYPE_R_IFROMX
                        for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

            # R_IfromJ
            for x in range(nr):
                if x < i or x > j:
                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    for k in range(i + 1, j):
                        tmp = CE(x, k, scores) + Int[k, j, 0, 0] + R_IFromX[i, k, j]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, 0, 0), k, j, 0, TYPE_R_IFROMX, i, k, j
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                    for k in range(i, j):
                        tmp = CE(j, k, scores) + R[k, j, x, 0, 0, 0] + Int[i, k, 1, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_R, 0, 0, 0), k, j, x, idx_type(TYPE_INT, 1, 0), i, k, 0
                            cand_bt[12:17] = TYPE_CE, -1, j, k, -1
                    if cand > R_IFromJ[i, j, x]:
                        R_IFromJ[i, j, x] = cand
                        _k = TYPE_R_IFROMJ
                        for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

            # R_TFF
            for x in range(nr):
                if x < i or x > j:
                    bj = 0
                    bi = 1
                    start = vcross_start(i, j, bi, bj, TYPE_R)
                    end = vcross_end(i, j, bi, bj, TYPE_R)
                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    for k in range(start, end + 1):
                        tmp = CE(x, k, scores) + Int[k, j, 0, bj] + Int[i, k, bi, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, 0, bj), k, j, 0, idx_type(TYPE_INT, bi, 0), i, k, 0
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(x, k, scores) + Int[k, j, 0, 0] + R[i, k, j, bi, 0, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_INT, 0, 0), k, j, 0, idx_type(TYPE_R, bi, 0, 0), i, k, j
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(x, k, scores) + R[k, j, x, 0, bj, 0] + Int[i, k, bi, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_R, 0, bj, 0), k, j, x, idx_type(TYPE_INT, bi, 0), i, k, 0
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                        tmp = CE(j, k, scores) + R[k, j, x, 0, 0, 0] + Int[i, k, bi, 0]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = idx_type(TYPE_R, 0, 0, 0), k, j, x, idx_type(TYPE_INT, bi, 0), i, k, 0
                            cand_bt[12:17] = TYPE_CE, -1, j, k, -1
                    if cand > R[i, j, x, bi, bj, 0]:
                        R[i, j, x, bi, bj, 0] = cand
                        _k = idx_type(TYPE_R, bi, bj, 0)
                        for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

            # R biTF
            for x in range(nr):
                if x < i or x > j:
                    bj = 1
                    for bi in range(2):
                        start = vcross_start(i, j, bi, bj, TYPE_R)
                        end = vcross_end(i, j, bi, bj, TYPE_R)
                        cand = NEGINF
                        for idx in range(22): cand_bt[idx] = -1
                        for k in range(start, end + 1):
                            tmp = CE(x, k, scores) + Int[k, j, 0, bj] + Int[i, k, bi, 0]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = idx_type(TYPE_INT, 0, bj), k, j, 0, idx_type(TYPE_INT, bi, 0), i, k, 0
                                cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                            tmp = CE(x, k, scores) + Int[k, j, 0, 1] + R[i, k, j, bi, 0, 0]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = idx_type(TYPE_INT, 0, 1), k, j, 0, idx_type(TYPE_R, bi, 0, 0), i, k, j
                                cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                            tmp = CE(x, k, scores) + Int[k, j, 0, 0] + R[i, k, j, bi, 0, 1]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = idx_type(TYPE_INT, 0, 0), k, j, 0, idx_type(TYPE_R, bi, 0, 1), i, k, j
                                cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                            tmp = CE(x, k, scores) + R[k, j, x, 0, bj, 0] + Int[i, k, bi, 0]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = idx_type(TYPE_R, 0, bj, 0), k, j, x, idx_type(TYPE_INT, bi, 0), i, k, 0
                                cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                            tmp = CE(j, k, scores) + R_JFromX[k, j, x] + Int[i, k, bi, 0]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = TYPE_R_JFROMX, k, j, x, idx_type(TYPE_INT, bi, 0), i, k, 0
                                cand_bt[12:17] = TYPE_CE, -1, j, k, -1
                            tmp = CE(k, j, scores) + R_IFromX[k, j, x] + Int[i, k, bi, 0]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = TYPE_R_IFROMX, k, j, x, idx_type(TYPE_INT, bi, 0), i, k, 0
                                cand_bt[12:17] = TYPE_CE, -1, k, j, -1
                            tmp = CE(k, j, scores) + R[k, j, x, 0, 0, 0] + Int[i, k, bi, 1]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = idx_type(TYPE_R, 0, 0, 0), k, j, x, idx_type(TYPE_INT, bi, 1), i, k, 0
                                cand_bt[12:17] = TYPE_CE, -1, k, j, -1
                        if cand > R[i, j, x, bi, bj, 0]:
                            R[i, j, x, bi, bj, 0] = cand
                            _k = idx_type(TYPE_R, bi, bj, 0)
                            for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

            # R biFT
            for x in range(nr):
                if x < i or x > j:
                    bj = 0
                    for bi in range(2):
                        start = vcross_start(i, j, bi, bj, TYPE_R)
                        end = vcross_end(i, j, bi, bj, TYPE_R)
                        cand = NEGINF
                        for idx in range(22): cand_bt[idx] = -1
                        for k in range(start, end + 1):
                            tmp = CE(k, x, scores) + Int[k, j, 1, bj] + Int[i, k, bi, 0]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = idx_type(TYPE_INT, 1, bj), k, j, 0, idx_type(TYPE_INT, bi, 0), i, k, 0
                                cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                            tmp = CE(k, x, scores) + Int[k, j, 0, bj] + Int[i, k, bi, 1]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = idx_type(TYPE_INT, 0, bj), k, j, 0, idx_type(TYPE_INT, bi, 1), i, k, 0
                                cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                            tmp = CE(k, x, scores) + Int[k, j, 0, bj] + R[i, k, j, bi, 1, 0]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = idx_type(TYPE_INT, 0, bj), k, j, 0, idx_type(TYPE_R, bi, 1, 0), i, k, j
                                cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                            tmp = CE(k, x, scores) + Int[k, j, 1, bj] + R[i, k, j, bi, 0, 0]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = idx_type(TYPE_INT, 1, bj), k, j, 0, idx_type(TYPE_R, bi, 0, 0), i, k, j
                                cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                            tmp = CE(k, x, scores) + R_IFromJ[k, j, x] + Int[i, k, bi, 0]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = TYPE_R_IFROMJ, k, j, x, idx_type(TYPE_INT, bi, 0), i, k, 0
                                cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                            tmp = CE(k, x, scores) + R[k, j, x, 0, 0, 0] + Int[i, k, bi, 1]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = idx_type(TYPE_R, 0, 0, 0), k, j, x, idx_type(TYPE_INT, bi, 1), i, k, 0
                                cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                            tmp = CE(x, k, scores) + R_XFromJ[k, j, x] + Int[i, k, bi, 0]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = TYPE_R_XFROMJ, k, j, x, idx_type(TYPE_INT, bi, 0), i, k, 0
                                cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                            tmp = CE(j, k, scores) + R[k, j, x, 0, 0, 1] + Int[i, k, bi, 0]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:8] = idx_type(TYPE_R, 0, 0, 1), k, j, x, idx_type(TYPE_INT, bi, 0), i, k, 0
                                cand_bt[12:17] = TYPE_CE, -1, j, k, -1
                        if cand > R[i, j, x, bi, bj, 1]:
                            R[i, j, x, bi, bj, 1] = cand
                            _k = idx_type(TYPE_R, bi, bj, 1)
                            for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

            # Chain_JFromI
            for x in range(nr):
                if x < i or x > j:
                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    tmp = CE(i, j, scores) + L[i, j, x, 0, 0, 0]
                    if tmp > cand:
                        cand = tmp
                        cand_bt[:4] = TYPE_L, i, j, x
                        cand_bt[4:8] = -1, -1, -1, -1
                        cand_bt[12:17] = TYPE_CE, -1, i, j, -1
                    for k in range(i + 1, j):
                        tmp = CE(x, k, scores) + Int[i, k, 0, 0] + Chain_JFromX[k, j, i]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:4] = TYPE_INT, i, k, 0
                            cand_bt[4:8] = TYPE_CHAIN_JFROMX, k, j, i
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                    if cand > Chain_JFromI[i, j, x]:
                        Chain_JFromI[i, j, x] = cand
                        _k = TYPE_CHAIN_JFROMI
                        for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

            # Chain_IFromJ
            for x in range(nr):
                if x < i or x > j:
                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    tmp = CE(j, i, scores) + R[i, j, x, 0, 0, 0]
                    if tmp > cand:
                        cand = tmp
                        cand_bt[:4] = TYPE_R, i, j, x
                        cand_bt[4:8] = -1, -1, -1, -1
                        cand_bt[12:17] = TYPE_CE, -1, j, i, -1
                    for k in range(i + 1, j):
                        tmp = CE(x, k, scores) + Int[k, j, 0, 0] + Chain_IFromX[i, k, j]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = TYPE_INT, k, j, 0, TYPE_CHAIN_IFROMX, i, k, j
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                    if cand > Chain_IFromJ[i, j, x]:
                        Chain_IFromJ[i, j, x] = cand
                        _k = TYPE_CHAIN_IFROMJ
                        for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

            # Chain_JFromX
            for x in range(nr):
                if x < i or x > j:
                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    for k in range(i + 1, j):
                        tmp = CE(x, k, scores) + Int[i, k, 0, 0] + Chain_JFromI[k, j, i]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = TYPE_INT, i, k, 0, TYPE_CHAIN_JFROMI, k, j, i
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                    if cand > Chain_JFromX[i, j, x]:
                        Chain_JFromX[i, j, x] = cand
                        _k = TYPE_CHAIN_JFROMX
                        for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

            # Chain_IFromX
            for x in range(nr):
                if x < i or x > j:
                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    for k in range(i + 1, j):
                        tmp = CE(x, k, scores) + Int[k, j, 0, 0] + Chain_IFromJ[i, k, j]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:8] = TYPE_INT, k, j, 0, TYPE_CHAIN_IFROMJ, i, k, j
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                    if cand > Chain_IFromX[i, j, x]:
                        Chain_IFromX[i, j, x] = cand
                        _k = TYPE_CHAIN_IFROMX
                        for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

            # LR
            for x in range(nr):
                if x < i or x > j:
                    for bx in range(2):
                        cand = NEGINF
                        for idx in range(22): cand_bt[idx] = -1
                        tmp = L[i, j, x, 0, 0, bx]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:4] = idx_type(TYPE_L, 0, 0, bx), i, j, x
                            cand_bt[4:8] = -1, -1, -1, -1
                        tmp = R[i, j, x, 0, 0, bx]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:4] = idx_type(TYPE_R, 0, 0, bx), i, j, x
                            cand_bt[4:8] = -1, -1, -1, -1
                        for k in range(i + 1, j):
                            tmp = Chain_JFromI[i, k, x] + R[k, j, x, 0, 0, bx]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:4] = TYPE_CHAIN_JFROMI, i, k, x
                                cand_bt[4:8] = idx_type(TYPE_R, 0, 0, bx), k, j, x
                            if bx == 1:
                                tmp = Chain_JFromX[i, k, x] + R_XFromJ[k, j, x]
                                if tmp > cand:
                                    cand = tmp
                                    cand_bt[:4] = TYPE_CHAIN_JFROMX, i, k, x
                                    cand_bt[4:8] = TYPE_R_XFROMJ, k, j, x
                            else:
                                tmp = Chain_JFromX[i, k, x] + R[k, j, x, 0, 0, 0]
                                if tmp > cand:
                                    cand = tmp
                                    cand_bt[:4] = TYPE_CHAIN_JFROMX, i, k, x
                                    cand_bt[4:8] = TYPE_R, k, j, x
                            tmp = L[i, k, x, 0, 0, bx] + Chain_IFromJ[k, j, x]
                            if tmp > cand:
                                cand = tmp
                                cand_bt[:4] = idx_type(TYPE_L, 0, 0, bx), i, k, x
                                cand_bt[4:8] = TYPE_CHAIN_IFROMJ, k, j, x
                            if bx == 1:
                                tmp = L_XFromI[i, k, x] + Chain_IFromX[k, j, x]
                                if tmp > cand:
                                    cand = tmp
                                    cand_bt[:4] = TYPE_L_XFROMI, i, k, x
                                    cand_bt[4:8] = TYPE_CHAIN_IFROMX, k, j, x
                            else:
                                tmp = L[i, k, x, 0, 0, 0] + Chain_IFromX[k, j, x]
                                if tmp > cand:
                                    cand = tmp
                                    cand_bt[:4] = TYPE_L, i, k, x
                                    cand_bt[4:8] = TYPE_CHAIN_IFROMX, k, j, x
                        if cand > LR[i, j, x, bx]:
                            LR[i, j, x, bx] = cand
                            _k = TYPE_LR + bx
                            for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

            # Page
            for x in range(nr):
                if x < i or x > j:
                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    tmp = CE(j, i, scores) + L[i, j, x, 0, 0, 0]
                    if tmp > cand:
                        cand = tmp
                        cand_bt[:4] = TYPE_L, i, j, x
                        cand_bt[4:8] = -1, -1, -1, -1
                        cand_bt[12:17] = TYPE_CE, -1, j, i, -1
                    for k in range(i + 1, j):
                        tmp = CE(x, k, scores) + Int[i, k, 0, 0] + Page2[k, j, i]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:4] = TYPE_INT, i, k, 0
                            cand_bt[4:8] = TYPE_PAGE2, k, j, i
                            cand_bt[12:17] = TYPE_CE, -1, x, k, -1
                    if cand > Page1[i, j, x]:
                        Page1[i, j, x] = cand
                        _k = TYPE_PAGE1
                        for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

            for x in range(nr):
                if x < i or x > j:
                    cand = NEGINF
                    for idx in range(22): cand_bt[idx] = -1
                    for k in range(i + 1, j):
                        tmp = CE(k, x, scores) + Int[i, k, 0, 0] + Page1[k, j, i]
                        if tmp > cand:
                            cand = tmp
                            cand_bt[:4] = TYPE_INT, i, k, 0
                            cand_bt[4:8] = TYPE_PAGE1, k, j, i
                            cand_bt[12:17] = TYPE_CE, -1, k, x, -1
                    if cand > Page2[i, j, x]:
                        Page2[i, j, x] = cand
                        _k = TYPE_PAGE2
                        for idx in range(22): backtrack[_k, i, j, x, idx] = cand_bt[idx]

    heads = -np.ones(nr - 1, dtype=np.int32, order="c")
    traces = -np.ones((nr * 3, 3), dtype=np.int32, order="c")
    backtrack_1ec_o3(backtrack, TYPE_TRIG, 1, nr - 1, 0, heads, traces, 0)

    return np.asarray(heads), np.asarray(traces)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int backtrack_1ec_o3(int[:, :, :, :, ::1] backtrack,
        int t, int i, int j, int x, int[::1] heads, int[:, ::1] traces, int next_i):
    cdef int t1, i1, j1, x1
    cdef int t2, i2, j2, x2
    cdef int t3, i3, j3, x3
    cdef int e1, g1, h1, m1, s1
    cdef int e2, g2, h2, m2, s2

    if (t == idx_type(TYPE_INT, 0, 1) or t == idx_type(TYPE_INT, 1, 0)) and i == j:
        return next_i
    if (t == idx_type(TYPE_INT, 0, 0)) and (i == j or j == i + 1):
        return next_i
    if (t == TYPE_TRIG or t == TYPE_TRIG + 1) and i == j:
        return next_i

    t1, i1, j1, x1 = backtrack[t, i, j, x, 0], backtrack[t, i, j, x, 1], backtrack[t, i, j, x, 2], backtrack[t, i, j, x, 3]
    t2, i2, j2, x2 = backtrack[t, i, j, x, 4], backtrack[t, i, j, x, 5], backtrack[t, i, j, x, 6], backtrack[t, i, j, x, 7]
    t3, i3, j3, x3 = backtrack[t, i, j, x, 8], backtrack[t, i, j, x, 9], backtrack[t, i, j, x, 10], backtrack[t, i, j, x, 11]
    e1, g1, h1, m1, s1 = backtrack[t, i, j, x, 12], backtrack[t, i, j, x, 13], backtrack[t, i, j, x, 14], backtrack[t, i, j, x, 15], backtrack[t, i, j, x, 16]
    e2, g2, h2, m2, s2 = backtrack[t, i, j, x, 17], backtrack[t, i, j, x, 18], backtrack[t, i, j, x, 19], backtrack[t, i, j, x, 20], backtrack[t, i, j, x, 21]

    if e1 >= 0:
        heads[m1 - 1] = h1 - 1
        if e1 == TYPE_GS or e1 == TYPE_G:
            traces[next_i, 0] = g1
            traces[next_i, 1] = m1
            traces[next_i, 2] = 0
            next_i += 1
        if e1 == TYPE_GS or e1 == TYPE_S:
            traces[next_i, 0] = s1
            traces[next_i, 1] = m1
            traces[next_i, 2] = 2
            next_i += 1
        if e1 == TYPE_CE:
            traces[next_i, 0] = h1
            traces[next_i, 1] = m1
            traces[next_i, 2] = 3
            next_i += 1
        else:
            traces[next_i, 0] = h1
            traces[next_i, 1] = m1
            traces[next_i, 2] = 1
            next_i += 1
    if e2 >= 0:
        heads[m2 - 1] = h2 - 1
        if e2 == TYPE_GS or e2 == TYPE_G:
            traces[next_i, 0] = g2
            traces[next_i, 1] = m2
            traces[next_i, 2] = 0
            next_i += 1
        if e2 == TYPE_GS or e2 == TYPE_S:
            traces[next_i, 0] = s2
            traces[next_i, 1] = m2
            traces[next_i, 2] = 2
            next_i += 1
        if e2 == TYPE_CE:
            traces[next_i, 0] = h2
            traces[next_i, 1] = m2
            traces[next_i, 2] = 3
            next_i += 1
        else:
            traces[next_i, 0] = h2
            traces[next_i, 1] = m2
            traces[next_i, 2] = 1
            next_i += 1
    if t1 >= 0:
        next_i = backtrack_1ec_o3(backtrack, t1, i1, j1, x1, heads, traces, next_i)
    if t2 >= 0:
        next_i = backtrack_1ec_o3(backtrack, t2, i2, j2, x2, heads, traces, next_i)
    if t3 >= 0:
        next_i = backtrack_1ec_o3(backtrack, t3, i3, j3, x3, heads, traces, next_i)

    return next_i
