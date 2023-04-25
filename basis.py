#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：SeqQCSim 
@File    ：basis.py.py
@Author  ：ZiHao Li
@Date    ：2023/4/7 14:30 
"""
import numpy as np
import copy
import scipy as sp
import tensornetwork as tn


# SuperOp基（内含正交基维护）
class SuperOpBasis:
    def __init__(self):
        self.basis = []
        self.orthogonalbasis = []

    def append(self, a):
        self.basis.append(a)
        va = copy.deepcopy(a)
        va = va.tensor.reshape(1, -1)[0]
        tmp = copy.deepcopy(va)
        # print("tmp_init:", tmp)
        for b in self.orthogonalbasis:
            # print("va:", va)
            # print("b:", b)
            # print("np.vdot(b, va):", np.vdot(b, va))
            # print("np.vdot(b, va) * b:", np.vdot(b, va) * b)
            tmp -= np.vdot(b, va) * b
        #     print("tmp_process:", tmp)
        # print("tmp_final:", tmp)
        va = tmp / np.linalg.norm(tmp)
        self.orthogonalbasis.append(va)
        print("orthogonal?:", self.self_check())

    def is_independent(self, a):
        va = copy.deepcopy(a)
        va = va.tensor.reshape(1, -1)[0]
        res = 0
        for b in self.orthogonalbasis:
            tmp = np.vdot(b, va)
            res += np.linalg.norm(tmp) ** 2
        # print("res:", res)
        # print("|a|**2:", np.linalg.norm(va) ** 2)
        if abs(res - np.linalg.norm(va) ** 2) < 1e-8:
            return False
        else:
            return True

    def size(self):
        return len(self.basis)

    def get_basis(self):
        return self.basis

    def get_orthogonalbasis(self):
        return self.orthogonalbasis

    def clear(self):
        self.basis = []
        self.orthogonalbasis = []

    def print(self):
        print("basis:")
        for x in self.basis:
            print(x.tensor)
        print("orthogonalbasis:")
        for x in self.orthogonalbasis:
            print(x)

    def self_check(self):
        for i in range(len(self.orthogonalbasis)):
            for j in range(i + 1, len(self.orthogonalbasis)):
                if np.vdot(self.orthogonalbasis[i], self.orthogonalbasis[j]) > 1e-8:
                    print("orthogonalbasis %d and %d not orthogonal" % (i + 1, j + 1))
                    print("INFO:\n", self.orthogonalbasis[i], self.orthogonalbasis[j])
                    print(np.linalg.norm(self.orthogonalbasis[i]), np.linalg.norm(self.orthogonalbasis[j]))
                    print(np.vdot(self.orthogonalbasis[i], self.orthogonalbasis[j]))
                    return False
        return True

