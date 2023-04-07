#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：SeqQCSim 
@File    ：basis.py.py
@Author  ：ZiHao Li
@Date    ：2023/4/7 14:30 
"""
import numpy as np


# SuperOp基（内含正交基维护）
class SuperOpBasis:
    def __init__(self, basis):
        self.basis = []
        self.orthogonalbasis = []

    def get_vectorized_basis(self):
        return [b.tensor.reshape(1, -1)[0] for b in self.basis]

    def append(self, a):
        self.basis.append(a)
        a = a.tensor.reshape(1, -1)[0]
        for b in self.orthogonalbasis:
            a -= np.dot(a, b) * b
        a = a / np.linalg.norm(a)
        self.orthogonalbasis.append(a)

    def is_independent(self, a):
        a = a.tensor.reshape(1, -1)[0]
        res = 0
        for b in self.orthogonalbasis:
            tmp = np.dot(a, b)
            res += np.real(tmp) * np.real(tmp) + np.imag(tmp) * np.imag(tmp)
        if abs(res - np.dot(a, a).real) < 1e-8:
            return False
        else:
            return True
