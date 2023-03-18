#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：SeqQCSim 
@File    ：qft-small-1.py
@Author  ：ZiHao Li
@Date    ：2023/3/17 23:32 
"""

import tensornetwork as tn
import numpy as np
from operation import *
from gate import *
from algorithm import eq_check

B = [tn.Node(np.array([[1, 0], [0, 0]], dtype=complex)),  # |00>
     tn.Node(np.array([[0, 1], [0, 0]], dtype=complex))  # |01>
     ]  # the collection of all possible input states
O = list(range(1 << len(B[0].get_all_edges())))  # the collection of all possible outputs
gate_info_list = [[Cgate([0], QFT(4)), [1, 2, 3, 4, 5]], [CT, [1, 2]], [CNOT, [4, 0]]]
unitary = get_unitary_matrix(6, gate_info_list)

tmp_1 = np.zeros(16, dtype=complex)
tmp_1[0] = 1
rho_1 = tn.Node(tmp_1.reshape([2, 2, 2, 2]))
tmp_2 = np.zeros(16, dtype=complex)
tmp_2[8] = 1
rho_2 = tn.Node(tmp_2.reshape([2, 2, 2, 2]))
stored_density_1 = tn.Node(get_density_matrix(rho_1))
stored_density_2 = tn.Node(get_density_matrix(rho_2))

eq_check(B, O, unitary, unitary, stored_density_1, stored_density_2)
