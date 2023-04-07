#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：SeqQCSim 
@File    ：qhalf-adder.py
@Author  ：ZiHao Li
@Date    ：2023/3/18 19:34 
"""

import tensornetwork as tn
import numpy as np
from operation import *
from gate import *
from algorithm import *

B = [get_computational_basis_by_index(2, 0),  # |00>
     get_computational_basis_by_index(2, 1),  # |01>
     get_computational_basis_by_index(2, 2),  # |10>
     get_computational_basis_by_index(2, 3)  # |11>
     ]  # the collection of all possible input states
O = list(range(1 << len(B[0].get_all_edges())))  # the collection of all possible outputs
gate_info_list_1 = [[Cgate([1], VD), [2, 0]], [CNOT, [1, 2]], [Cgate([1], V), [2, 0]], [Cgate([1], V), [1, 0]],
                    [CNOT, [1, 0]]]
unitary_1 = get_unitary_matrix(3, gate_info_list_1)
gate_info_list_2 = [[Toffoli, [1, 2, 0]], [CNOT, [1, 2]]]
unitary_2 = get_unitary_matrix(3, gate_info_list_2)

rho_1 = get_computational_basis_by_index(1, 0)  # |0>
rho_2 = get_computational_basis_by_index(1, 0)  # |0>
stored_density_1 = tn.Node(get_density_matrix(rho_1))
stored_density_2 = tn.Node(get_density_matrix(rho_2))

eq_check_ver1(B, O, unitary_1, unitary_2, stored_density_1, stored_density_2)
