from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.utils.data as data
import numpy as np
import os
import h5py


def load_h5_kitti(h5_filename):
    f = h5py.File(h5_filename)
    data = f['point_cloud'][:]
    label = f['gt_normals'][:]
    return data, label

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return data, label

class KittiNormalEst(data.Dataset):
    def __init__(self, stage='train'):
        super(KittiNormalEst, self).__init__()
        print('loading ', stage, ' data...')
        h5_city = '../KITTI_raw_data/city4096.h5'
        h5_resi = '../KITTI_raw_data/resi4096.h5'
        h5_road = '../KITTI_raw_data/road4096.h5'
        h5_campus = '../KITTI_raw_data/campus4096.h5'
        data_city, label_city = load_h5_kitti(h5_city)
        data_resi, label_resi = load_h5_kitti(h5_resi)
        data_road, label_road = load_h5_kitti(h5_road)
        data_campus, label_campus = load_h5_kitti(h5_campus)

        if stage == 'train':
            pts_data = np.concatenate([data_city[:int(0.6 * data_city.shape[0]), ...], \
                                        data_resi[:int(0.6 * data_resi.shape[0]), ...], \
                                        data_road[:int(0.6 * data_road.shape[0]), ...], \
                                        data_campus[:int(0.6 * data_campus.shape[0]), ...]], axis=0)
            label = np.concatenate([label_city[:int(0.6 * label_city.shape[0]), ...], \
                                        label_resi[:int(0.6 * label_resi.shape[0]), ...], \
                                        label_road[:int(0.6 * label_road.shape[0]), ...], \
                                        label_campus[:int(0.6 * label_campus.shape[0]), ...]], axis=0)

            idx = np.arange(pts_data.shape[0])
            np.random.shuffle(idx)
            pts_data = pts_data[idx, ...]
            label = label[idx, ...]

        elif stage == 'val':
            pts_data = np.concatenate([data_city[int(0.6 * data_city.shape[0]):int(0.8 * data_city.shape[0]), ...], \
                                        data_resi[int(0.6 * data_resi.shape[0]):int(0.8 * data_resi.shape[0]), ...], \
                                        data_road[int(0.6 * data_road.shape[0]):int(0.8 * data_road.shape[0]), ...], \
                                        data_campus[int(0.6 * data_campus.shape[0]):int(0.8 * data_campus.shape[0]), ...]], axis=0)
            label = np.concatenate([label_city[int(0.6 * label_city.shape[0]):int(0.8 * label_city.shape[0]), ...], \
                                        label_resi[int(0.6 * label_resi.shape[0]):int(0.8 * label_resi.shape[0]), ...], \
                                        label_road[int(0.6 * label_road.shape[0]):int(0.8 * label_road.shape[0]), ...], \
                                        label_campus[int(0.6 * label_campus.shape[0]):int(0.8 * label_campus.shape[0]), ...]], axis=0)
        else:
            pts_data = np.concatenate([data_city[int(0.8 * data_city.shape[0]):, ...], \
                                        data_resi[int(0.8 * data_resi.shape[0]):, ...], \
                                        data_road[int(0.8 * data_road.shape[0]):, ...], \
                                        data_campus[int(0.8 * data_campus.shape[0]):, ...]], axis=0)
            label = np.concatenate([label_city[int(0.8 * label_city.shape[0]):, ...], \
                                        label_resi[int(0.8 * label_resi.shape[0]):, ...], \
                                        label_road[int(0.8 * label_road.shape[0]):, ...], \
                                        label_campus[int(0.8 * label_campus.shape[0]):, ...]], axis=0)
        self.points = pts_data
        self.labels = label


        print('data shape: ', self.points.shape)
        print('label shape: ', self.labels.shape)
 
        np.savez('data.npz',data=self.points[:150,...])
        print('150 data saved')

    def __getitem__(self, idx):
        # pt_idxs = np.arange(0, self.num_points)
        # np.random.shuffle(pt_idxs)
        current_points = torch.from_numpy(self.points[idx, ...].copy()).type(
            torch.FloatTensor
        )
        current_labels = torch.from_numpy(self.labels[idx, ...].copy()).type(
            torch.FloatTensor
        )
        return current_points, current_labels

    def __len__(self):
        return int(self.points.shape[0])

class GeneratedDataset(data.Dataset):
    def __init__(self, h5_filename):
        self.points, self.labels = load_h5(h5_filename)

        print('data shape: ', self.points.shape)
        print('label shape: ', self.labels.shape)

    def __getitem__(self, index):
        point_set = self.points[index]
        normals = self.labels[index]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        point_set = torch.from_numpy(point_set).type(torch.FloatTensor)
        normals = torch.from_numpy(normals).type(torch.FloatTensor)

        return point_set, normals

    def __len__(self):
        return self.points.shape[0]


if __name__ == "__main__":
    dset = GeneratedDataset('../data/train_plane_no_noise.h5')
    print(dset[0])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
    for i, data in enumerate(dloader, 0):
        inputs, labels = data
        if i == len(dloader) - 1:
            print(inputs.size())
