# -*- coding: utf-8 -*-
# @Author : DX3906
# @Email : lingjintao.su@gmail.com
# @Time: 2022/1/17 15:09
# @File : dataset
import numpy
import numpy as np
import h5py

import torch
from torch.utils import data


class ChannelDataset(data.Dataset):
    def __init__(self, file_path, row_name, num_sample, num_rx, num_tx, num_delay):
        super().__init__()
        self.file_path = file_path
        self.num_sample = num_sample
        self.num_rx = num_rx
        self.num_tx = num_tx
        self.num_delay = num_delay
        self.row_name = row_name

        self.load_data()
        # self.data_augmentation()

    def load_data(self):
        data_train = h5py.File(self.file_path, 'r')
        data_train = np.transpose(data_train[self.row_name][:])
        self.data_test = data_train[::int(data_train.shape[0] / self.num_sample), :, :, :]
        self.data_test = self.data_test['real'] + self.data_test['imag'] * 1j
        data_train = data_train[:, :, :, :, np.newaxis]
        data_train = np.concatenate([data_train['real'], data_train['imag']], 4)
        data_train = data_train.astype(np.float32)
        if self.file_path[6]=='1':
            Augdata = []
            for i in range(100):
                print(i)
                temp = data_train[5 * i:5 * (i + 1), :, :, :].copy()
                for j in range(5):
                    for k in range(j, 5):
                        # for r in range(11):
                        for r in range(5, 6):
                            comb = r / 10 * temp[j, :, :, :] + (1 - r / 10) * temp[k, :, :, :]
                            comb = comb.astype(np.float32)
                            Augdata.append(comb)
            Augdata = np.array(Augdata, dtype=np.float32)
            # Augdata = data_train
        else:
            # Augdata = []
            # for i in range(200):
            #     print(i)
            #     temp = data_train[20 * i:20 * (i + 1), :, :, :].copy()
            #     for j in range(20):
            #         for k in range(j, 20):
            #             # for r in range(11):
            #             for r in range(5, 6):
            #                 comb = r / 10 * temp[j, :, :, :] + (1 - r / 10) * temp[k, :, :, :]
            #                 comb = comb.astype(np.float32)
            #                 Augdata.append(comb)
            # Augdata = np.array(Augdata, dtype=np.float32)
            Augdata = data_train
        self.train_channel = np.concatenate([Augdata,
                                     data_train * -1,
                                     np.concatenate([data_train[:, :, :, :, 1:2] * -1, data_train[:, :, :, :, 0:1]], axis=-1),
                                     np.concatenate([data_train[:, :, :, :, 1:2], data_train[:, :, :, :, 0:1] * -1],
                                                    axis=-1)], axis=0)

    def get_all_data(self):
        return self.data_test

    def get_all_normed_data(self):
        return self.train_channel

    def __len__(self):
        return len(self.train_channel)

    def __getitem__(self, index):
        return self.train_channel[index]


def get_data(file_path, row_name, num_sample, batch_size):
    train_dataset = ChannelDataset(file_path=file_path, row_name=row_name, num_sample=num_sample, num_rx=4, num_tx=32,
                                   num_delay=32)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               drop_last=False)
    return train_loader, train_dataset.get_all_data(), train_dataset.get_all_normed_data()


if __name__ == "__main__":
    loader, _, _ = get_data('data/H2_32T4R.mat', 'H2_32T4R', 4000, 32)
    # for sample in loader:
    #     print(sample.shape)
    #     break
