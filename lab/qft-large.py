#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：SeqQCSim 
@File    ：qft-large.py
@Author  ：ZiHao Li
@Date    ：2023/4/27 15:00 
"""

import tensornetwork as tn
import numpy as np
from operation import *
from gate import *
from algorithm import *
import time

n = 7
B = [get_computational_basis_by_index(2, 0),  # |00>
     get_computational_basis_by_index(2, 1)  # |01>
     ]  # the collection of all possible input states
O = list(range(1 << len(B[0].get_all_edges())))  # the collection of all possible outputs
gate_info_list = [[Cgate([0], QFT(n)), range(1, n + 2)], [CT, [1, 2]], [CNOT, [4, 0]]]
unitary = get_unitary_matrix(n + 2, gate_info_list)

rho_1 = get_computational_basis_by_index(n, 0)
rho_2 = get_computational_basis_by_index(n, 1 << (n - 1))
stored_density_1 = tn.Node(get_density_matrix(rho_1))
stored_density_2 = tn.Node(get_density_matrix(rho_2))
start = time.time()
eq_check_ver2(B, O, unitary, unitary, stored_density_1, stored_density_2)
end = time.time()
print("time: ", end - start)