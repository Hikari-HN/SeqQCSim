#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：SeqQCSim 
@File    ：qwalk-sym-1.py
@Author  ：ZiHao Li
@Date    ：2023/3/17 14:41 
"""

import tensornetwork as tn
import numpy as np
from operation import *
from gate import *
from algorithm import eq_check

B = [tn.Node(np.array([1, 0], dtype=complex)),  # |0>
     tn.Node(np.array([0, 1], dtype=complex)),  # |1>
     tn.Node(np.array([1, 1], dtype=complex) / np.sqrt(2)),  # |+>
     tn.Node(np.array([1, 1j], dtype=complex) / np.sqrt(2)),  # |theta>
     ]  # the collection of all possible input states
O = list(range(1 << len(B[0].get_all_edges())))  # the collection of all possible outputs
gate_info_list = [[H, [1]], [TWALK, [1, 2, 3]], [Toffoli, [2, 3, 0]]]
unitary = get_unitary_matrix(4, gate_info_list)

tmp_1 = np.zeros(8, dtype=complex)
tmp_1[0] = 1
rho_1 = tn.Node(tmp_1.reshape([2, 2, 2]))
tmp_2 = np.zeros(8, dtype=complex)
tmp_2[6] = 1
rho_2 = tn.Node(tmp_2.reshape([2, 2, 2]))
stored_density_1 = tn.Node(get_density_matrix(rho_1))
stored_density_2 = tn.Node(get_density_matrix(rho_2))

eq_check(B, O, unitary, unitary, stored_density_1, stored_density_2)
