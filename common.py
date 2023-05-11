#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：TN_learn 
@File    ：common.py
@Author  ：ZiHao Li
@Date    ：2023/2/28 22:35 
"""
from itertools import combinations, product
import numpy as np


def sub_lists(l):
    subs = []
    for i in range(0, len(l) + 1):
        temp = [list(x) for x in combinations(l, i)]
        if len(temp) > 0:
            subs.extend(temp)
    return subs


def gen_all_output(n):
    return list(list(x) for x in product([0, 1], repeat=n))


def is_span_rank_ver(super_op, super_op_basis):
    if not super_op_basis:
        return False
    vectorized_basis = [basis.tensor.reshape(1, -1)[0] for basis in super_op_basis]
    tmp = np.zeros((1, 1 << len(super_op_basis[0].shape)))
    for basis in vectorized_basis:
        tmp = np.append(tmp, [basis], axis=0)
    tmp = np.delete(tmp, 0, axis=0)
    basis_rank = np.linalg.matrix_rank(tmp)
    tmp = np.append(tmp, [super_op.tensor.reshape(1, -1)[0]], axis=0)
    new_rank = np.linalg.matrix_rank(tmp)
    if new_rank == basis_rank:
        return True
    else:
        return False

