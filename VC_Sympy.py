#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：TN_learn 
@File    ：VC_Sympy.py
@Author  ：ZiHao Li
@Date    ：2023/3/1 12:55 
"""
from sympy.physics.quantum.gate import *
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qubit import Qubit

c = H(0) * T(0) * H(0) * CGate(0, Z(1)) * T(0) * T(0) * T(0) * T(0) * T(0) * T(0) * T(0) * H(0) * T(0) * CGate(0,
                                                                                                               Z(1)) * H(
    0) * Z(1) * T(0) * H(0)
result = qapply(c * Qubit('00'))  # note that qubits are indexed from right to left
print(result)
