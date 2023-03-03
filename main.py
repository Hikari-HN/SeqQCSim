#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：TN_learn 
@File    ：main.py
@Author  ：ZiHao Li
@Date    ：2023/2/28 20:01 
"""
from gate import *
from operation import *

# NodeCollection allows us to store all the nodes created under this context.
for i in range(100):
    all_nodes = []
    with tn.NodeCollection(all_nodes):
        state_nodes = [
            tn.Node(np.array([1.0 + 0.0j, 0.0 + 0.0j])),
            tn.Node(gen_random_qubit()) if i == 0 else tn.Node(get_new_qubit(result, 1))
        ]
        print(state_nodes[1].tensor)
        qubits = [node[0] for node in state_nodes]
        apply_gate(qubits, H, [0])
        apply_gate(qubits, T, [0])
        apply_gate(qubits, H, [0])
        apply_gate(qubits, CZ, [0, 1])
        apply_gate(qubits, TD, [0])
        apply_gate(qubits, H, [0])
        apply_gate(qubits, T, [0])
        apply_gate(qubits, CZ, [0, 1])
        apply_gate(qubits, H, [0])
        apply_gate(qubits, Z, [1])
        apply_gate(qubits, T, [0])
        apply_gate(qubits, H, [0])
    result = tn.contractors.optimal(all_nodes, output_edge_order=qubits)
    result = do_random_measure(result, [0])
