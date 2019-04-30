import numpy as np
import os
from open3d import *
import h5py

# plane no noise
# f = h5py.File('data/test_plane_no_noise.h5')
# points = f['data'][:]
# gt = f['label'][:]
# pred = np.load('results/plane_no_noise_pred.npy')

# plane with noise 0.1
# f = h5py.File('data/test_plane_noise0.1.h5')
# points = f['data'][:]
# gt = f['label'][:]
# pred = np.load('results/plane_noise0.1_pred.npy')

# plane with noise 0.3
# f = h5py.File('data/test_plane_noise0.3.h5')
# points = f['data'][:]
# gt = f['label'][:]
# pred = np.load('results/plane_noise0.3_pred.npy')

# curv no noise
f = h5py.File('data/test_curv_no_noise.h5')
points = f['data'][:]
gt = f['label'][:]
pred = np.load('results/curv_no_noise_pred.npy')

pred = pred / np.linalg.norm(pred, axis=2, keepdims=True)

cos_angle = np.abs(np.sum(np.multiply(gt, pred), axis=2))
cos_angle = np.clip(cos_angle, 0, 1.0)
angle_dif = np.arccos(cos_angle) / 3.14 * 180

loss = np.sqrt(np.mean(np.square(angle_dif)))

print("RMS angle error:", loss)


pcd = PointCloud()
pcd.points = Vector3dVector(points[100])
estimate_normals(pcd, search_param = KDTreeSearchParamHybrid(radius = 0.1, max_nn = 5000))
POINT_NUM = points.shape[1]
for i in range(POINT_NUM):
	pcd.normals[i] = pred[100,i,:]

draw_geometries([pcd])