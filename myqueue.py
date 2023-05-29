#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：SeqQCSim 
@File    ：queue.py
@Author  ：ZiHao Li
@Date    ：2023/3/12 21:03 
"""


# 链式队列
class MyNode:
    def __init__(self, item):
        self.item = item
        self.next = None


class MyQueue:
    def __init__(self):
        self.head = None
        self.tail = None

    def push(self, item):
        node = MyNode(item)
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            self.tail = node

    def pop(self):
        if self.head is None:
            return None
        else:
            node = self.head
            self.head = self.head.next
            return node

    def is_empty(self):
        return self.head is None

    def size(self):
        if self.head is None:
            return 0
        else:
            node = self.head
            count = 1
            while node.next is not None:
                node = node.next
                count += 1
            return count

    def print(self):
        if self.head is None:
            print("Empty queue!")
        else:
            node = self.head
            while node is not None:
                print(node.item, end=" ")
                node = node.next
            print()