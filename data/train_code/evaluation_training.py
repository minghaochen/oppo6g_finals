# -*- coding: utf-8 -*-
# @Author : DX3906
# @Email : lingjintao.su@gmail.com
# @Time: 2022/1/10 21:35
# @File : evaluation_training
import numpy as np
from model_config import *


def K_nearest(h_true_smp, h_fake_smp, rx_num, tx_num, delay_num, flag):
    h_true = np.reshape(h_true_smp, [h_true_smp.shape[0], rx_num * tx_num * delay_num])
    h_fake = np.reshape(h_fake_smp, [h_fake_smp.shape[0], rx_num * tx_num * delay_num])
    h_true_norm = np.linalg.norm(h_true, axis=1)
    h_fake_norm = np.linalg.norm(h_fake, axis=1)
    h_true_norm = h_true_norm[:, np.newaxis]
    h_fake_norm = h_fake_norm[:, np.newaxis]
    h_true_norm_matrix = np.tile(h_true_norm, (1, rx_num * tx_num * delay_num))
    h_fake_norm_matrix = np.tile(h_fake_norm, (1, rx_num * tx_num * delay_num))
    h_true = h_true / h_true_norm_matrix
    h_fake = h_fake / h_fake_norm_matrix

    r_s = abs(np.dot(h_fake, h_true.conj().T))
    r = r_s * r_s

    r_max = np.max(r, axis=1)
    r_idx = np.argmax(r, axis=1)
    K_sim_abs_mean = np.mean(r_max)

    counts_idx, counts_num = np.unique(r_idx, return_counts=True)
    K_multi = np.zeros((1, h_true_smp.shape[0]))
    K_multi[:, counts_idx] = counts_num
    K_multi_std = float(np.sqrt(np.var(K_multi, axis=1) * h_fake_smp.shape[0] / (h_fake_smp.shape[0] - 1)))

    # print('Norming: ' + str(t2 - t1) + 's')
    # print('Matrix multiplication: ' + str(t3 - t2) + 's')
    # print('UE Loop: ' + str(t4 - t3) + 's')
    return K_sim_abs_mean, K_multi_std, K_multi_std / K_sim_abs_mean
