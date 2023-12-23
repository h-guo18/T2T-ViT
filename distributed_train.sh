#!/bin/bash
# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
NUM_PROC=$1
shift
torchrun  --nproc_per_node=$NUM_PROC --rdzv-backend=c10d --rdzv-endpoint=localhost:29600 main.py "$@" 

