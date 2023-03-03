#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：TN_learn 
@File    ：common.py
@Author  ：ZiHao Li
@Date    ：2023/2/28 22:35 
"""
from itertools import combinations, product


def sub_lists(l):
    subs = []
    for i in range(0, len(l) + 1):
        temp = [list(x) for x in combinations(l, i)]
        if len(temp) > 0:
            subs.extend(temp)
    return subs


def gen_all_output(n):
    return list(list(x) for x in product([0, 1], repeat=n))
