#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：SeqQCSim 
@File    ：qctrl-2.py
@Author  ：ZiHao Li
@Date    ：2023/3/18 20:04 
"""

import tensornetwork as tn
import numpy as np
from operation import *
from gate import *
from algorithm import *

B = [get_computational_basis_by_index(3, 0),  # |000>
     get_computational_basis_by_index(3, 1),  # |001>
     get_computational_basis_by_index(3, 2),  # |010>
     get_computational_basis_by_index(3, 3)  # |011>
     ]  # the collection of all possible input states
O = list(range(1 << len(B[0].get_all_edges())))  # the collection of all possible outputs
gate_info_list = [[Cgate([1, 1], Toffoli), [1, 2, 3, 4, 5]], [Cgate([0, 0], H), [1, 2, 3]],
                  [Cgate([0, 1], H), [1, 2, 4]], [Cgate([1, 0], H), [1, 2, 5]], [CNOT, [3, 0]]]
unitary = get_unitary_matrix(6, gate_info_list)

rho_1 = tn.Node(get_computational_basis_by_index(3, 6))
rho_2 = tn.Node(get_computational_basis_by_index(3, 5))
stored_density_1 = tn.Node(get_density_matrix(rho_1))
stored_density_2 = tn.Node(get_density_matrix(rho_2))

eq_check_ver1(B, O, unitary, unitary, stored_density_1, stored_density_2)
