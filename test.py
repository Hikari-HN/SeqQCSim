# from gate import *
# from operation import *
#
# all_nodes = []
# # NodeCollection allows us to store all the nodes created under this context.
# with tn.NodeCollection(all_nodes):
#     state_nodes = [tn.Node(np.array([1.0 + 0.0j, 0.0 + 0.0j])) for _ in range(3)]
#     qubits = [node[0] for node in state_nodes]
#     apply_gate(qubits, H, [0])
#     apply_gate(qubits, CNOT, [0, 1])
#     apply_gate(qubits, CNOT, [0, 2])
#
# result = tn.contractors.optimal(all_nodes, output_edge_order=qubits)
#
# print(result.tensor)
# result = do_random_measure(result, [0, 1])
# print(result.tensor)
# import numpy as np
# print(-np.sqrt(2)*1.0j*np.exp(1.0j*np.pi/4)/4 + np.sqrt(2)*np.exp(1.0j*np.pi/4)/2)