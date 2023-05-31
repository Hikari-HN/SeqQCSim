#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：SeqQCSim 
@File    ：circuit.py
@Author  ：ZiHao Li
@Date    ：2023/5/10 23:41 
"""

from operation import *
from gate import *
from myqueue import *


# 组合量子电路
class CombQCircuit:
    def __init__(self):
        self.gate_info_list = []
        self.unitary = None
        self.init_pure_state = None
        self.init_density = None
        self.work_mode = True  # True: pure state; False: density matrix

    def add_gate_info(self, gate_info):
        self.gate_info_list.append(gate_info)

    def set_gate_info_list(self, gate_info_list):
        self.gate_info_list = gate_info_list

    def set_init_pure_state(self, pure_state):
        self.init_pure_state = pure_state

    def set_init_density(self, density):
        self.init_density = density

    def check_work_mode(self):
        if self.init_pure_state is not None:
            self.work_mode = True
        elif self.init_density is not None:
            self.work_mode = False
        else:
            raise Exception("No initial state!")

    def get_unitary(self):
        num_qubits = len(self.init_pure_state.get_all_edges()) if self.work_mode else len(
            self.init_density.get_all_edges()) // 2
        self.unitary = tn.Node(get_unitary_matrix(num_qubits, self.gate_info_list))

    def get_final_result(self):
        self.check_work_mode()
        if self.unitary is None:
            self.get_unitary()
        if self.work_mode:
            num_qubits = len(self.init_pure_state.get_all_edges())
            [self.unitary[i + num_qubits] ^ self.init_pure_state[i] for i in range(num_qubits)]
            return self.unitary @ self.init_pure_state
        else:
            num_qubits = len(self.init_density.get_all_edges()) // 2
            unitary_dagger = tn.Node(np.conj(matrix_transpose(self.unitary.tensor)))
            [self.unitary[i + num_qubits] ^ self.init_density[i] for i in range(num_qubits)]
            [self.init_density[i + num_qubits] ^ unitary_dagger[i] for i in range(num_qubits)]
            return self.unitary @ self.init_density @ unitary_dagger


# qc = CombQCircuit()
# pure_state = tn.Node(get_computational_basis_by_index(2, 0))
# density = tn.Node(get_density_matrix(pure_state))
# qc.set_init_density(density)
# print(density.tensor)
# qc.add_gate_info([H, [0]])
# # qc.add_gate_info([SWAP, [0, 1]])
# qc.add_gate_info([CNOT, [0, 1]])
# print(qc.get_final_result().tensor)
# # print(p_dict_by_measure(qc.get_final_result(), [0, 1]))


# 时序量子电路
class SeqQCircuit:
    def __init__(self):
        self.comb_qc = CombQCircuit()
        self.input_pure_state_queue = MyQueue()
        self.input_density_queue = MyQueue()
        self.expected_output_queue = MyQueue()
        self.num_input_qubits = 0
        self.num_stored_qubits = 0
        self.num_qubits = 0
        self.init_stored_pure_state = None
        self.init_stored_density = None
        self.stored_density = None
        self.if_initialized = False

    def set_init_stored_pure_state(self, pure_state):
        self.init_stored_pure_state = pure_state
        self.init_stored_density = tn.Node(get_density_matrix(pure_state))
        self.num_stored_qubits = len(pure_state.get_all_edges())

    def set_init_stored_density(self, density):
        self.init_stored_density = density
        self.num_stored_qubits = len(density.get_all_edges()) // 2

    def set_input_pure_state_queue(self, pure_state_list):
        for pure_state in pure_state_list:
            self.input_pure_state_queue.push(pure_state)
        for pure_state in pure_state_list:
            self.input_density_queue.push(tn.Node(get_density_matrix(pure_state)))
        self.num_input_qubits = len(pure_state_list[0].get_all_edges())

    def set_input_density_queue(self, density_list):
        for density in density_list:
            self.input_density_queue.push(density)
        self.num_input_qubits = len(density_list[0].get_all_edges()) // 2

    def set_expected_output_queue(self, expected_output_list):
        for expected_output in expected_output_list:
            self.expected_output_queue.push(expected_output)

    def add_gate_info(self, gate_info):
        self.comb_qc.add_gate_info(gate_info)

    def set_gate_info_list(self, gate_info_list):
        self.comb_qc.set_gate_info_list(gate_info_list)

    def initialize(self):
        if self.num_stored_qubits == 0:
            raise Exception("No init stored density!")
        self.stored_density = self.init_stored_density
        self.num_qubits = self.num_input_qubits + self.num_stored_qubits
        self.if_initialized = True

    def print_stored_density(self):
        if self.stored_density is not None:
            print(self.stored_density.tensor)
        else:
            print("Maybe not initialized! Or the expected output queue is impossible!")

    def run_one_step(self):
        if not self.if_initialized:
            self.initialize()
        input_density = self.input_density_queue.pop().item
        if input_density is None:
            raise Exception("No input density!")
        expected_output = self.expected_output_queue.pop().item
        if expected_output is None:
            raise Exception("No expected output!")
        total_density = tn.Node(
            np.kron(input_density.tensor.reshape(1 << self.num_input_qubits, 1 << self.num_input_qubits),
                    self.stored_density.tensor.reshape(1 << self.num_stored_qubits,
                                                       1 << self.num_stored_qubits)).reshape([2] * 2 * self.num_qubits))
        self.comb_qc.set_init_density(total_density)
        transformed_density = self.comb_qc.get_final_result()
        # print(transformed_density.tensor)
        measurement = tn.Node(
            get_measurement_matrix_by_output(self.num_qubits, list(range(self.num_input_qubits)), expected_output))
        measurement_dagger = tn.Node(np.conj(matrix_transpose(measurement.tensor)))
        [measurement[i + self.num_qubits] ^ transformed_density[i] for i in range(self.num_qubits)]
        [transformed_density[i + self.num_qubits] ^ measurement_dagger[i] for i in range(self.num_qubits)]
        measured_density = measurement @ transformed_density @ measurement_dagger
        # print(measured_density.tensor)
        prob = measured_density.copy()
        [prob[i] ^ prob[i + self.num_qubits] for i in range(self.num_qubits)]
        prob = prob @ prob
        [measured_density[i] ^ measured_density[i + self.num_qubits] for i in range(self.num_input_qubits)]
        partial_trace = measured_density @ measured_density
        # print(partial_trace.tensor)
        print("prob:", prob.tensor)
        if np.abs(prob.tensor) <= 1e-8:  # set error threshold
            self.stored_density = None
        else:
            self.stored_density = partial_trace / prob

    def run_til_stop(self):
        while not self.input_density_queue.is_empty():
            self.run_one_step()
            if self.stored_density is None:
                break

# qc = SeqQCircuit()
# stored_pure_state = tn.Node(np.array([1, 1], dtype=np.complex128) / np.sqrt(2))
# qc.set_init_stored_pure_state(stored_pure_state)
# input_pure_state_list = [tn.Node(get_computational_basis_by_index(1, 0)) for _ in range(2)]
# qc.set_input_pure_state_queue(input_pure_state_list)
# qc.set_expected_output_queue([1,0])
# gate_info_list = [[H, [0]], [T, [0]], [H, [0]], [CZ, [0, 1]], [TD, [0]], [H, [0]], [T, [0]], [CZ, [0, 1]], [H, [0]],
#                   [Z, [1]], [T, [0]], [H, [0]]]
# qc.set_gate_info_list(gate_info_list)
# qc.initialize()
# qc.print_stored_density()
# qc.run_one_step()
# qc.print_stored_density()
# qc.run_one_step()
# qc.print_stored_density()
