#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：TN_learn 
@File    ：operation.py
@Author  ：ZiHao Li
@Date    ：2023/2/28 22:35 
"""
import tensornetwork as tn
import numpy as np
import random

from common import *


def apply_gate(qubit_edges, gate, operating_qubits):
    op = tn.Node(gate)
    for i, bit in enumerate(operating_qubits):
        tn.connect(qubit_edges[bit], op[i])
        qubit_edges[bit] = op[i + len(operating_qubits)]


def measure_by_output(state, operating_qubits, output):
    n = len(state.get_all_edges())
    p, index_list = get_p(state, operating_qubits, output)
    state.tensor = state.tensor.reshape(1 << n)
    for x in range(1 << n):
        if x not in index_list:
            state.tensor[x] = 0
        else:
            state.tensor[x] = state.tensor[x] / np.sqrt(p)
    state.tensor = state.tensor.reshape([2] * n)
    return state


def get_p(state, operating_qubits, output):
    n = len(state.get_all_edges())
    state.tensor = state.tensor.reshape(1 << n)
    sure_value = sum([1 << (n - 1 - bit) for i, bit in enumerate(operating_qubits) if output[i]])
    other_qubits = [i for i in range(n) if i not in operating_qubits]
    other_values = [sum([1 << (n - 1 - x) for x in sub_l]) for sub_l in sub_lists(other_qubits)]
    index_list = [x + sure_value for x in other_values]
    p = sum([abs(state.tensor[x]) ** 2 for x in index_list])
    state.tensor = state.tensor.reshape([2] * n)
    return p, index_list


def p_dict_by_measure(state, operating_qubits):
    p_dict = {}
    for output in gen_all_output(len(operating_qubits)):
        p_dict[str(output)] = get_p(state, operating_qubits, output)[0]
    return p_dict


def gen_random_qubit():
    theta = random.uniform(0, np.pi / 2)
    theta1 = random.uniform(0, 2 * np.pi)
    theta2 = random.uniform(0, 2 * np.pi)
    arg1 = np.sin(theta) * (np.cos(theta1) + np.sin(theta1) * 1.0j)
    arg2 = np.cos(theta) * (np.cos(theta2) + np.sin(theta2) * 1.0j)
    return np.array([arg1, arg2])


def do_random_measure(state, operating_qubits):
    p_dict = p_dict_by_measure(state, operating_qubits)
    output = eval(random.choices(list(p_dict.keys()), weights=list(p_dict.values()), k=1)[0])
    return measure_by_output(state, operating_qubits, output)


def get_new_qubit(state, target_qubit):
    n = len(state.get_all_edges())
    state.tensor = state.tensor.reshape(1 << n)
    new_qubit = np.zeros(2, dtype=complex)
    for i in range(2):
        for j in range(1 << n):
            if (j >> (n - 1 - target_qubit)) % 2 == i:
                new_qubit[i] += state.tensor[j]
    state.tensor = state.tensor.reshape([2] * n)
    return new_qubit
