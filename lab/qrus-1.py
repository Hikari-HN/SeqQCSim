#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：SeqQCSim 
@File    ：qrus-1.py
@Author  ：ZiHao Li
@Date    ：2023/3/12 13:01 
"""
import tensornetwork as tn
import numpy as np
from operation import *
from gate import *
from myqueue import *
from common import is_span

B = [tn.Node(np.array([1, 0], dtype=complex))]  # the collection of all possible input states
O = list(range(1 << len(B[0].get_all_edges())))  # the collection of all possible outputs
gate_info_list = [[H, [0]], [T, [0]], [H, [0]], [CZ, [0, 1]], [TD, [0]], [H, [0]], [T, [0]], [CZ, [0, 1]], [H, [0]],
                  [Z, [1]], [T, [0]], [H, [0]]]
unitary = get_unitary_matrix(2, gate_info_list)

rho_1 = tn.Node(np.array([1, 0], dtype=complex))
rho_2 = tn.Node(np.array([1, 1], dtype=complex) / np.sqrt(2))
stored_density_1 = tn.Node(get_density_matrix(rho_1))
stored_density_2 = tn.Node(get_density_matrix(rho_2))

super_op_basis = []
Q = MyQueue()
Q.push(([], []))
while not Q.is_empty():
    input_state_list, output_list = Q.pop().item
    super_operator_1 = get_total_super_operator(output_list, input_state_list, stored_density_1, unitary)
    super_operator_2 = get_total_super_operator(output_list, input_state_list, stored_density_2, unitary)
    super_operator = super_operator_1 - super_operator_2
    if super_operator:

        # print(super_operator.tensor)
        # print([x.tensor for x in super_op_basis])

        if not is_span(super_operator, super_op_basis):
            super_op_basis.append(super_operator)
            for input_state in B:
                for output in O:
                    Q.push((input_state_list + [input_state], output_list + [output]))
if check_trace_all_zero(super_op_basis):
    print("Yes!")
else:
    print("No!")
