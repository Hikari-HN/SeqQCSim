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
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
T = np.array([[1, 0], [0, np.exp(np.pi * 1.0j / 4)]], dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
TD = np.conjugate(T).T
V = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex128) / 2
VD = np.conjugate(V).T
CNOT = np.zeros((2, 2, 2, 2), dtype=np.complex128)
CNOT[0][0][0][0] = 1
CNOT[0][1][0][1] = 1
CNOT[1][0][1][1] = 1
CNOT[1][1][1][0] = 1
CZ = np.zeros((2, 2, 2, 2), dtype=np.complex128)
CZ[0][0][0][0] = 1
CZ[0][1][0][1] = 1
CZ[1][0][1][0] = 1
CZ[1][1][1][1] = -1
SWAP = np.zeros((2, 2, 2, 2), dtype=np.complex128)
SWAP[0][0][0][0] = 1
SWAP[0][1][1][0] = 1
SWAP[1][0][0][1] = 1
SWAP[1][1][1][1] = 1
TWALK = np.zeros((2, 2, 2, 2, 2, 2), dtype=np.complex128)
TWALK[0][0][0][0][1][1] = 1
TWALK[0][0][1][0][0][0] = 1
TWALK[0][1][0][0][0][1] = 1
TWALK[0][1][1][0][1][0] = 1
TWALK[1][0][0][1][0][1] = 1
TWALK[1][0][1][1][1][0] = 1
TWALK[1][1][0][1][1][1] = 1
TWALK[1][1][1][1][0][0] = 1
Toffoli = np.zeros((2, 2, 2, 2, 2, 2), dtype=np.complex128)
Toffoli[0][0][0][0][0][0] = 1
Toffoli[0][0][1][0][0][1] = 1
Toffoli[0][1][0][0][1][0] = 1
Toffoli[0][1][1][0][1][1] = 1
Toffoli[1][0][0][1][0][0] = 1
Toffoli[1][0][1][1][0][1] = 1
Toffoli[1][1][0][1][1][1] = 1
Toffoli[1][1][1][1][1][0] = 1
YWALK = np.array([[1, 1j], [1j, 1]], dtype=np.complex128) / np.sqrt(2)
CT = np.eye(4, dtype=np.complex128)
CT[3][3] = np.exp(np.pi * 1.0j / 4)
CT = CT.reshape((2, 2, 2, 2))


def RX(theta):
    """Returns the rotation around the X axis by theta."""
    return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                     [-1j * np.sin(theta / 2), np.cos(theta / 2)]], dtype=np.complex128)


def RandWALK(n):
    """Returns the random walk operator for n qubits."""
    N = 1 << n
    mat = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        if i < N // 2:
            mat[(i + 1) % (N // 2)][i] = 1
        else:
            mat[N // 2 + (i - 1) % (N // 2)][i] = 1
    return mat.reshape([2] * (2 * n))


def QFT(n):
    """Returns the QFT operator for n qubits."""
    N = 1 << n
    mat = np.zeros((N, N), dtype=np.complex128)
    for k in range(N):
        for j in range(N):
            mat[k][j] = np.exp(2 * np.pi * 1j * j * k / N) / np.sqrt(N)
    return mat.reshape([2] * (2 * n))


def Cgate(flag_list, gate):
    """
    :param flag_list: a list of flags for control qubits
    :param gate: a gate
    :return: a controlled gate
    """
    num_control_qubits = len(flag_list)
    num_target_qubits = len(gate.shape) // 2
    num_qubits = num_control_qubits + num_target_qubits
    N = 1 << num_qubits
    mat = np.eye(N, dtype=np.complex128)
    flag_num = sum([1 << (num_control_qubits - 1 - i) for i in range(num_control_qubits) if flag_list[i]])
    gate = gate.reshape(1 << num_target_qubits, 1 << num_target_qubits)
    for j in range(N):
        if (j >> num_target_qubits) == flag_num:
            for k in range(N - (1 << num_target_qubits), N):
                mat[j][k] = gate[j % (1 << num_target_qubits)][k % (1 << num_target_qubits)]
    return mat.reshape([2] * (2 * num_qubits))
