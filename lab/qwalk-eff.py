#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：SeqQCSim 
@File    ：qwalk-eff.py.py
@Author  ：ZiHao Li
@Date    ：2023/5/17 1:06 
"""

from circuit import *
import random
import time

num_qubits = 13
length = 10
random.seed(time.time())

qc = SeqQCircuit()
stored_pure_state = get_computational_basis_by_index(num_qubits - 1, 0)
qc.set_init_stored_pure_state(stored_pure_state)
input_pure_state_basis = [get_computational_basis_by_index(1, 0), get_computational_basis_by_index(1, 1),
                          tn.Node(np.array([1, 1], dtype=np.complex128) / np.sqrt(2)),
                          tn.Node(np.array([1, 1j], dtype=np.complex128) / np.sqrt(2))]
input_pure_state_list = [input_pure_state_basis[0] for _ in range(length)]
qc.set_input_pure_state_queue(input_pure_state_list)
qc.set_expected_output_queue([0 for _ in range(length)])
gate_info_list = [[H, [1]], [RandWALK(num_qubits - 1), list(range(1, num_qubits))],
                  [Cgate([1 for _ in range(num_qubits - 2)], X), list(range(2, num_qubits)) + [0]]]
qc.set_gate_info_list(gate_info_list)
qc.initialize()
start = time.time()
qc.run_til_stop()
end = time.time()
print("time:", end - start)
