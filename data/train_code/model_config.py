# -*- coding: utf-8 -*-
# @Author : DX3906
# @Email : lingjintao.su@gmail.com
# @Time: 2022/1/10 21:36
# @File : model_config

import multiprocessing

# Parameter Setting
NUM_RX = 4

NUM_TX = 32

NUM_DELAY = 32

NUM_REAL_1 = 500

NUM_REAL_2 = 4000

NUM_FAKE_1 = NUM_REAL_1

NUM_FAKE_2 = NUM_REAL_2

LATENT_DIM = 128

NUM_CORES = multiprocessing.cpu_count()
