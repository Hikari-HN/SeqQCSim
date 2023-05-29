#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：SeqQCSim 
@File    ：mealy.py
@Author  ：ZiHao Li
@Date    ：2023/5/14 5:11 
"""

from operation import *
from gate import *
from myqueue import *
from basis import *


# 量子米利机
class QuantumMealy:
    def __init__(self, dim_in, dim_s, unitary):
        self.dim_in = dim_in
        self.dim_s = dim_s
        self.unitary = unitary
        self.measure = [tn.Node(get_measurement_matrix_by_output(int(np.log2(dim_in)) + int(np.log2(dim_s)),
                                                                 list(range(int(np.log2(dim_in)))), output)) for output
                        in range(dim_in)]
        self.init_density = None
        self.stored_density = None

    def set_init_density(self, density):
        self.init_density = density

    def initialize(self):
        if self.init_density is None:
            raise Exception("No initial state!")
        self.stored_density = self.init_density

    def set_stored_density(self, density):
        self.stored_density = density

    def apply(self, input_density, expected_output):
        if input_density is None:
            return [self.stored_density, 1]
        if expected_output is None:
            raise Exception("No output for this input!")
        num_input_qubits = int(np.log2(self.dim_in))
        num_qubits = num_input_qubits + int(np.log2(self.dim_s))
        total_density = tn.Node(np.kron(input_density.tensor.reshape(self.dim_in, self.dim_in),
                                        self.stored_density.tensor.reshape(self.dim_s, self.dim_s)).reshape(
            [2] * 2 * num_qubits))
        unitary_dagger = tn.Node(np.conj(matrix_transpose(self.unitary.tensor)))
        measure = self.measure[expected_output]
        measure_dagger = tn.Node(np.conj(matrix_transpose(measure.tensor)))
        [measure[i + num_qubits] ^ self.unitary[i] for i in range(num_qubits)]
        [self.unitary[i + num_qubits] ^ total_density[i] for i in range(num_qubits)]
        [total_density[i + num_qubits] ^ unitary_dagger[i] for i in range(num_qubits)]
        [unitary_dagger[i + num_qubits] ^ measure_dagger[i] for i in range(num_qubits)]
        new_density = measure @ self.unitary @ total_density @ unitary_dagger @ measure_dagger
        tmp = new_density.copy()
        [new_density[i] ^ new_density[i + num_qubits] for i in range(num_input_qubits)]
        super_op = new_density @ new_density
        [tmp[i] ^ tmp[i + num_qubits] for i in range(num_qubits)]
        prob = tmp @ tmp
        return [super_op, prob]

    def get_super_op(self, input_density, expected_output):
        return self.apply(input_density, expected_output)[0]

    def get_prob(self, input_density, expected_output):
        return self.apply(input_density, expected_output)[1]

    def get_stored_density(self, input_density, expected_output):
        super_op, prob = self.apply(input_density, expected_output)
        if np.abs(prob.tensor) <= 1e-8:
            return None
        else:
            return super_op / prob

    def eq_check_with_mealy(self, qmealy, basis):
        assert (self.dim_in == qmealy.dim_in and self.dim_s == qmealy.dim_s)
        self.initialize()
        qmealy.initialize()
        O = list(range(self.dim_in))
        super_op_basis = SuperOpBasis()
        Q = MyQueue()
        Q.push(([], [], self.stored_density, qmealy.stored_density))
        while not Q.is_empty():
            input_state_list, output_list, rho1, rho2 = Q.pop().item
            if rho1:
                if rho2:
                    super_op = rho1 - rho2
                else:
                    super_op = rho1
            else:
                if rho2:
                    super_op = tn.Node(-rho2.tensor)
                else:
                    super_op = None
            if super_op:
                if super_op_basis.is_independent(super_op):
                    if not is_zero_trace(super_op):
                        return [False, input_state_list, output_list, rho1, rho2]
                    super_op_basis.append(super_op)
                    for input_state in basis:
                        for output in O:
                            input_density = tn.Node(get_density_matrix(input_state))
                            if rho1:
                                self.set_stored_density(rho1)
                                rho11 = self.get_stored_density(input_density, output)
                            else:
                                rho11 = None
                            if rho2:
                                qmealy.set_stored_density(rho2)
                                rho22 = qmealy.get_stored_density(input_density, output)
                            else:
                                rho22 = None
                            Q.push((input_state_list + [input_state], output_list + [output], rho11, rho22))
        return [True, [], [], self.init_density, qmealy.init_density]


# basis = [get_computational_basis_by_index(1, 0)]
# gate_info_list = [[H, [0]], [T, [0]], [H, [0]], [CZ, [0, 1]], [TD, [0]], [H, [0]], [T, [0]], [CZ, [0, 1]], [H, [0]],
#                   [Z, [1]], [T, [0]], [H, [0]]]
# unitary = tn.Node(get_unitary_matrix(2, gate_info_list))
# qmealy_1 = QuantumMealy(2, 2, unitary)
# init_density_1 = tn.Node(get_density_matrix(get_computational_basis_by_index(1, 0)))
# qmealy_1.set_init_density(init_density_1)
# qmwaly_2 = QuantumMealy(2, 2, unitary)
# init_density_2 = tn.Node(get_density_matrix(tn.Node(np.array([1, 1], dtype=np.complex128) / np.sqrt(2))))
# qmwaly_2.set_init_density(init_density_2)
# print(qmealy_1.eq_check_with_mealy(qmwaly_2, basis))

# basis = [get_computational_basis_by_index(1, 0),  # |0>
#          get_computational_basis_by_index(1, 1),  # |1>
#          tn.Node(np.array([1, 1], dtype=np.complex128) / np.sqrt(2)),  # |+>
#          tn.Node(np.array([1, 1j], dtype=np.complex128) / np.sqrt(2))  # |theta>
#          ]
# gate_info_list = [[H, [1]], [TWALK, [1, 2, 3]], [Toffoli, [2, 3, 0]]]
# unitary = tn.Node(get_unitary_matrix(4, gate_info_list))
# qmealy_1 = QuantumMealy(2, 8, unitary)
# init_density_1 = tn.Node(get_density_matrix(get_computational_basis_by_index(3, 0)))
# qmealy_1.set_init_density(init_density_1)
# qmwaly_2 = QuantumMealy(2, 8, unitary)
# init_density_2 = tn.Node(get_density_matrix(get_computational_basis_by_index(3, 2)))
# qmwaly_2.set_init_density(init_density_2)
# flag, input_state_list, output_list, _, _ = qmealy_1.eq_check_with_mealy(qmwaly_2, basis)
# print(flag)
# if not flag:
#     print("Counter Example:")
#     print("input_state_list:", [x.tensor for x in input_state_list])
#     print("output_list:", output_list)

# basis = [get_computational_basis_by_index(1, 0)]
# gate_info_list = [[H, [0]], [CNOT, [0, 1]]]
# unitary = tn.Node(get_unitary_matrix(2, gate_info_list))
# qmealy_1 = QuantumMealy(2, 2, unitary)
# init_density_1 = tn.Node(get_density_matrix(get_computational_basis_by_index(1, 0)))
# qmealy_1.set_init_density(init_density_1)
# qmwaly_2 = QuantumMealy(2, 2, unitary)
# init_density_2 = tn.Node(get_density_matrix(get_computational_basis_by_index(1, 1)))
# qmwaly_2.set_init_density(init_density_2)
# print(qmealy_1.eq_check_with_mealy(qmwaly_2, basis))

# input_state_list: [array([0.70710678+0.j, 0.70710678+0.j]), array([1.+0.j, 0.+0.j]), array([1.+0.j, 0.+0.j])]
# output_list: [0, 0, 0]
B = [get_computational_basis_by_index(2, 0),  # |00>
     get_computational_basis_by_index(2, 1)  # |01>
     ]  # the collection of all possible input states
O = list(range(1 << len(B[0].get_all_edges())))  # the collection of all possible outputs
gate_info_list = [[Cgate([0], QFT(4)), [1, 2, 3, 4, 5]], [CT, [1, 2]], [CNOT, [4, 0]]]
unitary = tn.Node(get_unitary_matrix(6, gate_info_list))
rho_1 = get_computational_basis_by_index(4, 0)
rho_2 = get_computational_basis_by_index(4, 8)
stored_density_1 = tn.Node(get_density_matrix(rho_1))
stored_density_2 = tn.Node(get_density_matrix(rho_2))
qmealy_1 = QuantumMealy(4, 16, unitary)
qmealy_1.set_stored_density(stored_density_1)
D = [tn.Node(get_density_matrix(x)) for x in B]
q1 = qmealy_1.get_prob(D[2], O[0]).tensor
qmealy_1.set_stored_density(qmealy_1.get_stored_density(D[2], O[0]))
q2 = qmealy_1.get_prob(D[0], O[0]).tensor
qmealy_1.set_stored_density(qmealy_1.get_stored_density(D[0], O[0]))
q3 = qmealy_1.get_prob(D[0], O[0]).tensor
qmealy_1.set_stored_density(qmealy_1.get_stored_density(D[0], O[0]))
print(q1 * q2 * q3, q1, q2, q3)

qmwaly_2 = QuantumMealy(4, 16, unitary)
qmwaly_2.set_stored_density(stored_density_2)
