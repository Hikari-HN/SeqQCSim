#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：SeqQCSim 
@File    ：qrus-2.py
@Author  ：ZiHao Li
@Date    ：2023/3/15 23:48 
"""

import tensornetwork as tn
import numpy as np
from operation import *
from gate import *
from algorithm import *
import time

B = [get_computational_basis_by_index(1, 0)]  # the collection of all possible input states
O = list(range(1 << len(B[0].get_all_edges())))  # the collection of all possible outputs
gate_info_list = [[H, [0]], [T, [0]], [H, [0]], [CZ, [0, 1]], [TD, [0]], [H, [0]], [T, [0]], [CZ, [0, 1]], [H, [0]],
                  [Z, [1]], [T, [0]], [H, [0]]]
unitary = get_unitary_matrix(2, gate_info_list)

rho_1 = tn.Node(np.array([1, 1j], dtype=np.complex128) / np.sqrt(2))
rho_2 = tn.Node(np.array([1, 2], dtype=np.complex128) / np.sqrt(5))
stored_density_1 = tn.Node(get_density_matrix(rho_1))
stored_density_2 = tn.Node(get_density_matrix(rho_2))
start = time.time()
eq_check_ver2(B, O, unitary, unitary, stored_density_1, stored_density_2)
end = time.time()
print("time:", end - start)
