#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：TN_learn 
@File    ：operation.py
@Author  ：ZiHao Li
@Date    ：2023/2/28 22:35 
"""
import tensornetwork as tn
import numpy as np
import random

from common import *


def apply_gate(qubit_edges, gate, operating_qubits):
    op = tn.Node(gate)
    for i, bit in enumerate(operating_qubits):
        tn.connect(qubit_edges[bit], op[i])
        qubit_edges[bit] = op[i + len(operating_qubits)]


def measure_by_output(state, operating_qubits, output):
    n = len(state.get_all_edges())
    p, index_list = get_p(state, operating_qubits, output)
    state.tensor = state.tensor.reshape(1 << n)
    for x in range(1 << n):
        if x not in index_list:
            state.tensor[x] = 0
        else:
            state.tensor[x] = state.tensor[x] / np.sqrt(p)
    state.tensor = state.tensor.reshape([2] * n)
    return state


def get_p(state, operating_qubits, output):
    n = len(state.get_all_edges())
    state.tensor = state.tensor.reshape(1 << n)
    sure_value = sum([1 << (n - 1 - bit) for i, bit in enumerate(operating_qubits) if output[i]])
    other_qubits = [i for i in range(n) if i not in operating_qubits]
    other_values = [sum([1 << (n - 1 - x) for x in sub_l]) for sub_l in sub_lists(other_qubits)]
    index_list = [x + sure_value for x in other_values]
    p = sum([abs(state.tensor[x]) ** 2 for x in index_list])
    state.tensor = state.tensor.reshape([2] * n)
    return p, index_list


def p_dict_by_measure(state, operating_qubits):
    p_dict = {}
    for output in gen_all_output(len(operating_qubits)):
        p_dict[str(output)] = get_p(state, operating_qubits, output)[0]
    return p_dict


def gen_random_qubit():
    theta = random.uniform(0, np.pi / 2)
    theta1 = random.uniform(0, 2 * np.pi)
    theta2 = random.uniform(0, 2 * np.pi)
    arg1 = np.sin(theta) * (np.cos(theta1) + np.sin(theta1) * 1.0j)
    arg2 = np.cos(theta) * (np.cos(theta2) + np.sin(theta2) * 1.0j)
    return np.array([arg1, arg2])


def do_random_measure(state, operating_qubits):
    p_dict = p_dict_by_measure(state, operating_qubits)
    output = eval(random.choices(list(p_dict.keys()), weights=list(p_dict.values()), k=1)[0])
    return measure_by_output(state, operating_qubits, output)


def get_new_qubit(state, target_qubit):
    n = len(state.get_all_edges())
    state.tensor = state.tensor.reshape(1 << n)
    new_qubit = np.zeros(2, dtype=np.complex128)
    for i in range(2):
        for j in range(1 << n):
            if (j >> (n - 1 - target_qubit)) % 2 == i:
                new_qubit[i] += state.tensor[j]
    state.tensor = state.tensor.reshape([2] * n)
    return new_qubit


def init_unitary_matrix(num_qubits):
    return np.eye(1 << num_qubits, dtype=np.complex128).reshape([2] * 2 * num_qubits)


def get_unitary_matrix(num_qubits, gate_info_list):
    all_gate_nodes = []
    with tn.NodeCollection(all_gate_nodes):
        unitary_matrix = tn.Node(init_unitary_matrix(num_qubits))
        input_edges, output_edges = [unitary_matrix.get_all_edges()[i: i + num_qubits] for i in
                                     range(0, 2 * num_qubits, num_qubits)]
        gate_info_list = gate_info_list[::-1]
        for gate_info in gate_info_list:
            apply_gate(output_edges, gate_info[0], gate_info[1])
    return tn.contractors.optimal(all_gate_nodes, output_edge_order=input_edges + output_edges).tensor


def get_density_matrix(state):
    return np.outer(state.tensor, tn.conj(state.tensor)).reshape([2] * 2 * len(state.get_all_edges()))


def get_measurement_operator(num_qubits, output):
    if type(output) == list:
        if len(output) != num_qubits:
            raise ValueError("output length must be equal to num_qubits")
        output = sum([1 << (num_qubits - 1 - i) for i in range(len(output)) if output[i]])
    if output >= 1 << num_qubits:
        raise ValueError("output must be less than 2 ** num_qubits")
    basis = np.zeros(1 << num_qubits, dtype=np.complex128)
    basis[output] = 1
    return np.outer(basis, np.conj(basis)).reshape([2] * 2 * num_qubits)


def get_measurement_matrix_by_output(num_qubits, target_qubits, output):
    all_gate_nodes = []
    with tn.NodeCollection(all_gate_nodes):
        measurement_matrix = tn.Node(init_unitary_matrix(num_qubits))
        input_edges, output_edges = [measurement_matrix.get_all_edges()[i: i + num_qubits] for i in
                                     range(0, 2 * num_qubits, num_qubits)]
        apply_gate(output_edges, get_measurement_operator(len(target_qubits), output), target_qubits)
    return tn.contractors.optimal(all_gate_nodes, output_edge_order=input_edges + output_edges).tensor


def get_super_operator_and_prob(output, input_density, stored_density, unitary):
    num_stored_qubits = len(stored_density.get_all_edges()) // 2
    num_input_qubits = len(input_density.get_all_edges()) // 2
    num_qubits = num_input_qubits + num_stored_qubits

    # print("input_density\n", input_density.tensor.reshape(1 << num_input_qubits, 1 << num_input_qubits))
    # print("stored_density\n", stored_density.tensor.reshape(1 << num_stored_qubits, 1 << num_stored_qubits))

    total_density_node = tn.Node(np.kron(input_density.tensor.reshape(1 << num_input_qubits, 1 << num_input_qubits),
                                         stored_density.tensor.reshape(1 << num_stored_qubits,
                                                                       1 << num_stored_qubits)).reshape(
        [2] * 2 * num_qubits))

    # print("total_density_node\n", total_density_node.tensor.reshape(1 << num_qubits, 1 << num_qubits))

    unitary_node = tn.Node(unitary)
    measurement = get_measurement_matrix_by_output(num_qubits, list(range(num_input_qubits)), output)
    measurement_node = tn.Node(measurement)
    unitary_dagger_node = tn.Node(np.conj(matrix_transpose(unitary)))
    measurement_dagger_node = tn.Node(np.conj(matrix_transpose(measurement)))

    # print("measurement_node\n", measurement_node.tensor.reshape(1 << num_qubits, 1 << num_qubits))
    # print("unitary_node\n", unitary_node.tensor.reshape(1 << num_qubits, 1 << num_qubits))

    [measurement_node[i + num_qubits] ^ unitary_node[i] for i in range(num_qubits)]
    unitary_node = measurement_node @ unitary_node

    # print("measurement * unitary\n", unitary_node.tensor.reshape(1 << num_qubits, 1 << num_qubits))
    # print("unitary_dagger_node\n", unitary_dagger_node.tensor.reshape(1 << num_qubits, 1 << num_qubits))
    # print("measurement_dagger_node\n", measurement_dagger_node.tensor.reshape(1 << num_qubits, 1 << num_qubits))

    [unitary_dagger_node[i + num_qubits] ^ measurement_dagger_node[i] for i in range(num_qubits)]
    unitary_dagger_node = unitary_dagger_node @ measurement_dagger_node

    # print("unitary_dagger * measurement_dagger\n", unitary_dagger_node.tensor.reshape(1 << num_qubits, 1 << num_qubits))
    # print("total_density_node\n", total_density_node.tensor.reshape(1 << num_qubits, 1 << num_qubits))

    [unitary_node[i + num_qubits] ^ total_density_node[i] for i in range(num_qubits)]
    total_density_node = unitary_node @ total_density_node

    # print("left * total_density\n", total_density_node.tensor.reshape(1 << num_qubits, 1 << num_qubits))

    [total_density_node[i + num_qubits] ^ unitary_dagger_node[i] for i in range(num_qubits)]
    new_density_node = total_density_node @ unitary_dagger_node

    # print("left * total_density * right\n", new_density_node.tensor.reshape(1 << num_qubits, 1 << num_qubits))

    tmp = new_density_node.copy()
    [new_density_node[i] ^ new_density_node[i + num_qubits] for i in range(num_input_qubits)]
    new_density_node = new_density_node @ new_density_node

    # print("partial_trace\n", new_density_node.tensor.reshape(1 << num_stored_qubits, 1 << num_stored_qubits))

    [tmp[i] ^ tmp[i + num_qubits] for i in range(num_qubits)]
    prob = tmp @ tmp

    # print("prob\n", prob.tensor)

    return new_density_node, prob


def get_total_super_operator(output_list, input_state_list, stored_density, unitary):
    super_operator = stored_density
    prob = tn.Node(1)
    for output, input_state in zip(output_list, input_state_list):
        input_density = tn.Node(get_density_matrix(input_state))
        super_operator, prob = get_super_operator_and_prob(output, input_density, stored_density, unitary)
        if np.abs(prob.tensor) <= 1e-8:  # set error threshold
            return None
        stored_density = super_operator / prob
    return super_operator


def is_zero_trace(super_op):
    tmp = super_op.copy()
    num_qubits = len(tmp.get_all_edges()) // 2
    var = [tmp[i] ^ tmp[i + num_qubits] for i in range(num_qubits)]
    trace = tmp @ tmp
    if np.abs(trace.tensor) > 1e-8:  # set error threshold
        return False
    return True


def check_trace_all_zero(super_op_basis):
    for super_op in super_op_basis:
        if not is_zero_trace(super_op):
            return False
    return True


def matrix_transpose(tensor):
    shape = tensor.shape
    n = 1 << (len(shape) // 2)
    return np.transpose(tensor.reshape(n, n)).reshape(shape)


def get_computational_basis_by_index(num_qubits, index):
    tmp = np.zeros(1 << num_qubits, dtype=np.complex128)
    tmp[index] = 1
    basis = tn.Node(tmp.reshape([2] * num_qubits))
    return basis
