#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：SeqQCSim 
@File    ：algorithm.py
@Author  ：ZiHao Li
@Date    ：2023/3/17 14:13 
"""

import tensornetwork as tn
import numpy as np
from operation import *
from myqueue import *
from common import is_span


def eq_check(B, O, unitary_1, unitary_2, stored_density_1, stored_density_2):
    super_op_basis = []
    Q = MyQueue()
    Q.push(([], []))
    while not Q.is_empty():
        input_state_list, output_list = Q.pop().item
        super_operator_1 = get_total_super_operator(output_list, input_state_list, stored_density_1, unitary_1)
        super_operator_2 = get_total_super_operator(output_list, input_state_list, stored_density_2, unitary_2)
        if super_operator_1:
            if super_operator_2:
                super_operator = super_operator_1 - super_operator_2
            else:
                super_operator = super_operator_1
        else:
            if super_operator_2:
                super_operator = tn.Node(-super_operator_2.tensor)
            else:
                super_operator = None
        if super_operator:
            if not is_span(super_operator, super_op_basis):
                if not is_zero_trace(super_operator):
                    print("No!")
                    print("input_state_list:", [x.tensor for x in input_state_list])
                    print("output_list:", output_list)
                    return
                super_op_basis.append(super_operator)
                for input_state in B:
                    for output in O:
                        Q.push((input_state_list + [input_state], output_list + [output]))
    print("Yes!")
    # if check_trace_all_zero(super_op_basis):
    #     print("Yes!")
    # else:
    #     print("No!")
