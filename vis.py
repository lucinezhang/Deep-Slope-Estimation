from numpy import linalg as LA
import matplotlib.pyplot as plt
import h5py
from open3d import *
import time
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='res_pruned.npz', help='predicted normal and data')
opt = parser.parse_args()



res = np.load(opt.file)
points = res['points']
normals = res['pred']
normals = normals / np.linalg.norm(normals, axis=2, keepdims=True)
points = np.concatenate(points, axis=0)
normals = np.concatenate(normals, axis=0)
print(points.shape, normals.shape)
res_p = PointCloud()
res_p.points = Vector3dVector(points)
res_p.paint_uniform_color([0.5, 0.5, 0.5])
estimate_normals(res_p, search_param = KDTreeSearchParamHybrid(radius = 1.2, max_nn = 25))
for i in range(normals.shape[0]):
    res_p.normals[i] = normals[i]
    res_p.colors[i] = normals[i]


draw_geometries([res_p])

# import h5py
# import matplotlib.pyplot as plt
# def load_h5_kitti(h5_filename):
#     f = h5py.File(h5_filename)
#     data = f['point_cloud'][:]
#     label = f['gt_normals'][:]
#     return data, label

# data_test, label_test = load_h5_kitti('test.h5')
# print(data_test.shape, label_test.shape)

# points = np.concatenate(data_test, axis=0)
# normals = np.concatenate(label_test, axis=0)
# res_p = PointCloud()
# res_p.points = Vector3dVector(points)
# estimate_normals(res_p, search_param = KDTreeSearchParamHybrid(radius = 1.2, max_nn = 25))
# for i in range(normals.shape[0]):
#     res_p.normals[i] = normals[i]
# draw_geometries([res_p])

# # print(np.array(res_p.normals))
# grid_pt = []
# for i in range(data_test.shape[0]):
#     tmp = np.unique(data_test[i], axis=0)
# #     print(tmp.shape)
#     grid_pt.append(tmp.shape[0])
# plt.hist(grid_pt, bins='auto')
# plt.title("Histogram with 'auto' bins")
# plt.show()