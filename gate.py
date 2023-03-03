#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：TN_learn 
@File    ：gate.py
@Author  ：ZiHao Li
@Date    ：2023/2/28 20:14 
"""
import numpy as np

# These are just numpy arrays of the operators.
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
T = np.array([[1, 0], [0, np.exp(np.pi * 1.0j / 4)]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
TD = np.conjugate(T)
CNOT = np.zeros((2, 2, 2, 2), dtype=complex)
CNOT[0][0][0][0] = 1
CNOT[0][1][0][1] = 1
CNOT[1][0][1][1] = 1
CNOT[1][1][1][0] = 1
CZ = np.zeros((2, 2, 2, 2), dtype=complex)
CZ[0][0][0][0] = 1
CZ[0][1][0][1] = 1
CZ[1][0][1][0] = 1
CZ[1][1][1][1] = -1
