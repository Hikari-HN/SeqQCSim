#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：SeqQCSim 
@File    ：qwalk-large.py
@Author  ：ZiHao Li
@Date    ：2023/4/6 14:16 
"""

import tensornetwork as tn
import numpy as np
from operation import *
from gate import *
from algorithm import *

n = 6
B = [get_computational_basis_by_index(1, 0),  # |0>
     get_computational_basis_by_index(1, 1),  # |1>
     tn.Node(np.array([1, 1], dtype=np.complex128) / np.sqrt(2)),  # |+>
     tn.Node(np.array([1, 1j], dtype=np.complex128) / np.sqrt(2))  # |theta>
     ]  # the collection of all possible input states
O = list(range(1 << len(B[0].get_all_edges())))  # the collection of all possible outputs
gate_info_list = [[H, [1]], [RandWALK(n + 1), list(range(1, n + 2))],
                  [Cgate([1 for _ in range(n)], X), list(range(2, n + 2)) + [0]]]
unitary = get_unitary_matrix(n + 2, gate_info_list)

rho_1 = get_computational_basis_by_index(n + 1, 0)
rho_2 = get_computational_basis_by_index(n + 1, (1 << (n + 1)) - 2)
stored_density_1 = tn.Node(get_density_matrix(rho_1))
stored_density_2 = tn.Node(get_density_matrix(rho_2))

eq_check_ver2(B, O, unitary, unitary, stored_density_1, stored_density_2)
