#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：SeqQCSim 
@File    ：qrus-valid-1.py
@Author  ：ZiHao Li
@Date    ：2023/5/31 9:35 
"""

from circuit import *

qc = SeqQCircuit()
stored_pure_state = tn.Node(get_computational_basis_by_index(1, 0))
qc.set_init_stored_pure_state(stored_pure_state)
input_pure_state_list = [tn.Node(get_computational_basis_by_index(1, 0)) for _ in range(2)]
qc.set_input_pure_state_queue(input_pure_state_list)
qc.set_expected_output_queue([1, 0])
gate_info_list = [[H, [0]], [T, [0]], [CNOT, [0, 1]], [H, [0]], [CNOT, [0, 1]], [T, [0]], [H, [0]]]
qc.set_gate_info_list(gate_info_list)
qc.initialize()
qc.print_stored_density()
qc.run_one_step()
qc.print_stored_density()
qc.run_one_step()
qc.print_stored_density()
