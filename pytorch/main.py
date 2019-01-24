#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : main.py
# Create date : 2019-01-10 16:39
# Modified date : 2019-01-24 14:05
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import torch

import matplotlib.pyplot as plt

def create_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def get_file_full_name(path, name):
    create_path(path)
    if path[-1] == "/":
        full_name = path +  name
    else:
        full_name = path + "/" +  name
    return full_name

def create_file(path, name, open_type='w'):
    file_name = get_file_full_name(path, name)
    return open(file_name, open_type)

def _plot_record(record, full_path):
    _plot_cpu_gpu_time(record, full_path)
    _plot_acceleration(record, full_path)

def _get_full_path(repeats, size_begin, size_end):
    if not os.path.exists("./output"):
        os.makedirs("./output")
    path_str = "./output/%s_%s_%s" % (repeats, size_begin, size_end)
    return path_str

def _plot_cpu_gpu_time(record, full_path):
    float32_numpy_lt = []
    float64_numpy_lt = []
    float32_cpu_lt = []
    float64_cpu_lt = []
    float32_gpu_lt = []
    float64_gpu_lt = []
    steps = []
    for key in record:
        steps.append([key])
    steps.sort()


    for i in range(len(steps)):
        step_dic = record[steps[i][0]]
        float32_numpy_value = step_dic["float32_numpy"]
        float32_numpy_lt.append(float32_numpy_value)
        float64_numpy_value = step_dic["float64_numpy"]
        float64_numpy_lt.append(float64_numpy_value)

        float32_cpu_value = step_dic["float32_torch_cpu"]
        float32_cpu_lt.append(float32_cpu_value)
        float64_cpu_value = step_dic["float64_torch_cpu"]
        float64_cpu_lt.append(float64_cpu_value)

        float32_gpu_value = step_dic["float32_torch_gpu"]
        float32_gpu_lt.append(float32_gpu_value)
        float64_gpu_value = step_dic["float64_torch_gpu"]
        float64_gpu_lt.append(float64_gpu_value)

    float32_numpy_lt = np.array(float32_numpy_lt)
    float64_numpy_lt = np.array(float64_numpy_lt)

    float32_cpu_lt = np.array(float32_cpu_lt)
    float64_cpu_lt = np.array(float64_cpu_lt)
    float32_gpu_lt = np.array(float32_gpu_lt)
    float64_gpu_lt = np.array(float64_gpu_lt)

    steps = np.array(steps)
    steps = steps*steps

    float32_gpu_line, = plt.plot(steps, float32_gpu_lt)
    float64_gpu_line, = plt.plot(steps, float64_gpu_lt)
    float32_cpu_line, = plt.plot(steps, float32_cpu_lt)
    float64_cpu_line, = plt.plot(steps, float64_cpu_lt)

    float32_numpy_line, = plt.plot(steps, float32_numpy_lt)
    float64_numpy_line, = plt.plot(steps, float64_numpy_lt)
    # pylint: disable=bad-continuation

    line_lt = [
    float32_gpu_line,
    float64_gpu_line,
    float32_cpu_line,
    float64_cpu_line,
    float32_numpy_line,
    float64_numpy_line,
    ]

    labels_lt = (
    "float32 torch gpu",
    "float64 torch gpu",
    "float32 torch cpu",
    "float64 torch cpu",
    "float32 numpy",
    "float64 numpy",
    )
    # pylint: enable=bad-continuation
    plt.legend(handles=line_lt, labels=labels_lt, loc='best')
    full_path_name = "%s/cpu_gpu.jpg" % (full_path)
#    plt.show()
    plt.savefig(full_path_name)
    plt.close()

def _plot_acceleration(record, full_path):
    float64_acceleration_lt = []
    float32_acceleration_lt = []
    float64_np_torch_cpu_acceleration_lt = []
    float32_np_torch_cpu_acceleration_lt = []
    float64_np_torch_gpu_acceleration_lt = []
    float32_np_torch_gpu_acceleration_lt = []

    steps = []
    for key in record:
        steps.append([key])
    steps.sort()

    for i in range(len(steps)):
        step_dic = record[steps[i][0]]
        float64_acceleration_lt.append(step_dic["float64_torch_acceleration"])
        float32_acceleration_lt.append(step_dic["float32_torch_acceleration"])

        float64_np_torch_cpu_acceleration_lt.append(step_dic["float64_np_torch_cpu_acceleration"])
        float32_np_torch_cpu_acceleration_lt.append(step_dic["float32_np_torch_cpu_acceleration"])

        float64_np_torch_gpu_acceleration_lt.append(step_dic["float64_np_torch_gpu_acceleration"])
        float32_np_torch_gpu_acceleration_lt.append(step_dic["float32_np_torch_gpu_acceleration"])

    float64_acceleration_lt = np.array(float64_acceleration_lt)
    float32_acceleration_lt = np.array(float32_acceleration_lt)

    float64_np_torch_cpu_acceleration_lt = np.array(float64_np_torch_cpu_acceleration_lt)
    float32_np_torch_cpu_acceleration_lt = np.array(float32_np_torch_cpu_acceleration_lt)

    float64_np_torch_gpu_acceleration_lt = np.array(float64_np_torch_gpu_acceleration_lt)
    float32_np_torch_gpu_acceleration_lt = np.array(float32_np_torch_gpu_acceleration_lt)

    steps = np.array(steps)
    steps = steps*steps

    l1, = plt.plot(steps, float32_acceleration_lt)
    l2, = plt.plot(steps, float64_acceleration_lt)

    l3, = plt.plot(steps, float32_np_torch_cpu_acceleration_lt)
    l4, = plt.plot(steps, float64_np_torch_cpu_acceleration_lt)

    l5, = plt.plot(steps, float32_np_torch_gpu_acceleration_lt)
    l6, = plt.plot(steps, float64_np_torch_gpu_acceleration_lt)
    # pylint: disable=bad-continuation

    line_lt = [
        l1,
        l2,
        l3,
        l4,
        l5,
        l6,
    ]


    labels_lt = (
    'float32 torch acceleration',
    'float64 torch acceleration',
    'float64 np torch cpu acceleration',
    'float32 np torch cpu acceleration',
    'float64 np torch gpu acceleration',
    'float32 np torch gpu acceleration',
    )
    # pylint: enable=bad-continuation

    plt.legend(handles=line_lt, labels=labels_lt, loc='best')
    full_path_name = "%s/acceleration.jpg" % (full_path)
#    plt.show()
    plt.savefig(full_path_name)
    plt.close()

def _write_status(file_obj, i, time_lt):
    float32_acceleration = time_lt[1] / time_lt[3]
    float64_acceleration = time_lt[0] / time_lt[2]

    float64_cpu_str = "i:%s float64 cpu:%s" % (i, time_lt[0])
    float32_cpu_str = "i:%s float32 cpu:%s" % (i, time_lt[1])
    float64_gpu_str = "i:%s float64 gpu:%s" % (i, time_lt[2])
    float32_gpu_str = "i:%s float32 gpu:%s" % (i, time_lt[3])
    float64_numpy_str = "i:%s float64 numpy:%s" % (i, time_lt[4])
    float32_numpy_str = "i:%s float32 numpy:%s" % (i, time_lt[5])

    float32_torch_acceleration_str = "float32 torch acceleration:%s" % float32_acceleration
    float64_torch_acceleration_str = "float64 torch acceleration:%s" % float64_acceleration

    file_obj.write("%s\n" % float64_cpu_str)
    file_obj.write("%s\n" % float32_cpu_str)
    file_obj.write("%s\n" % float64_gpu_str)
    file_obj.write("%s\n" % float32_gpu_str)
    file_obj.write("%s\n" % float64_numpy_str)
    file_obj.write("%s\n" % float32_numpy_str)
    file_obj.write("%s\n" % float32_torch_acceleration_str)
    file_obj.write("%s\n" % float64_torch_acceleration_str)

    print(float64_cpu_str)
    print(float32_cpu_str)
    print(float64_gpu_str)
    print(float32_gpu_str)
    print(float64_numpy_str)
    print(float32_numpy_str)
    print(float32_torch_acceleration_str)
    print(float64_torch_acceleration_str)

def _record_status(record, i, time_lt):
    dic = {}
    dic["float64_torch_cpu"] = time_lt[0]
    dic["float32_torch_cpu"] = time_lt[1]
    dic["float64_torch_gpu"] = time_lt[2]
    dic["float32_torch_gpu"] = time_lt[3]
    dic["float64_numpy"] = time_lt[4]
    dic["float32_numpy"] = time_lt[5]

    dic["float64_torch_acceleration"] = time_lt[0] / time_lt[2]
    dic["float32_torch_acceleration"] = time_lt[1] / time_lt[3]

    dic["float64_np_torch_cpu_acceleration"] = time_lt[4] / time_lt[0]
    dic["float32_np_torch_cpu_acceleration"] = time_lt[5] / time_lt[1]

    dic["float64_np_torch_gpu_acceleration"] = time_lt[4] / time_lt[2]
    dic["float32_np_torch_gpu_acceleration"] = time_lt[5] / time_lt[3]

    record[i] = dic

def _get_numpy_take_time(x, y, repeats, data_type):
    x = np.array(x, dtype=data_type)
    y = np.array(y, dtype=data_type)

    t0 = time.time()
    for i in range(repeats):
        z = np.matmul(x, y)
    t1 = time.time()
    v = z.sum()

    all_time = t1 - t0
    avg_time = all_time / repeats
    return avg_time, v

def _get_take_time(x, y, repeats, data_type, dev="cpu"):
    x = torch.from_numpy(x)
    x = x.type(data_type)

    y = torch.from_numpy(y)
    y = y.type(data_type)

    if dev == "gpu":
        device = torch.device("cuda")
        x = x.to(device)
        y = y.to(device)

    t0 = time.time()
    for i in range(repeats):
        z = torch.matmul(x, y)
    t1 = time.time()

    v = z.sum()
    all_time = t1 - t0
    avg_time = all_time / repeats
    return avg_time, v.item()

def test_cpu_gpu(repeats, size_begin, size_end, step=1):
    record = {}
    full_path = _get_full_path(repeats, size_begin, size_end)
    file_obj = create_file(full_path, "output")
    for s in range(size_begin, size_end, step):
        time_lt = []

        x = np.random.randn(s, s)
        y = np.random.randn(s, s)

        float64_cpu_time, v1 = _get_take_time(x, y, repeats, torch.double, "cpu")
        float32_cpu_time, v2 = _get_take_time(x, y, repeats, torch.float, "cpu")
        time_lt.append(float64_cpu_time)
        time_lt.append(float32_cpu_time)

        float64_gpu_time, v3 = _get_take_time(x, y, repeats, torch.double, "gpu")
        float32_gpu_time, v4 = _get_take_time(x, y, repeats, torch.float, "gpu")
        time_lt.append(float64_gpu_time)
        time_lt.append(float32_gpu_time)

        float64_numpy_time, v5 = _get_numpy_take_time(x, y, repeats, np.float64)
        float32_numpy_time, v6 = _get_numpy_take_time(x, y, repeats, np.float32)
        time_lt.append(float64_numpy_time)
        time_lt.append(float32_numpy_time)
        print(v1, v2, v3, v4, v5, v6)
        file_obj.write("%s %s %s %s %s %s" % (v1, v2, v3, v4, v5, v6))

        _write_status(file_obj, s, time_lt)
        _record_status(record, s, time_lt)

    file_obj.close()
    _plot_record(record, full_path)

def test_matmul(repeats, max_size, step):
    for i in range(int(max_size / step)):
        size_begin = 1 + i*step
        size_end = (i + 1)*step
        test_cpu_gpu(repeats, size_begin, size_end)

    size_begin = 1
    size_end = max_size
    test_cpu_gpu(repeats, size_begin, size_end)

def test():
    repeats = 1000
    max_size = 500
    step = 100
    test_matmul(repeats, max_size, step)

    repeats = 5
    size_begin = 500
    size_end = 3000
    step = 5
    test_cpu_gpu(repeats, size_begin, size_end, step)

    repeats = 1
    size_begin = 1
    size_end = 10000
    step = 50
    test_cpu_gpu(repeats, size_begin, size_end, step)

    repeats = 1
    size_begin = 10000
    size_end = 20000
    step = 200
    test_cpu_gpu(repeats, size_begin, size_end, step)

test()
