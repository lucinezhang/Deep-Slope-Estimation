import numpy as np
import sys
import h5py

def load_h5(h5_filename):
	f = h5py.File(h5_filename)
	data = f['data'][:]
	label = f['label'][:]
	return (data, label)

# print help message
if len(sys.argv) < 3:
	print("Usage: python generate_plane.py point_num point_cloud_num gaussian_noise_sigma")
	exit()

POINT_NUM = int(sys.argv[1])
PCLOUD_NUM = int(sys.argv[2])
NOISE_SIGMA = float(sys.argv[3])
h5_filename = 'test_plane.h5'

points = np.empty((PCLOUD_NUM, POINT_NUM, 3), dtype=np.float)
ground_truth_normals = np.empty((PCLOUD_NUM, POINT_NUM, 3), dtype=np.float)

for pc_num in range(PCLOUD_NUM):
	# generate parameters of the 3D plane ax+by+cz+d=0
	f = (np.random.rand(4) - 0.5) * 10

	# generate x and y coordinates randomly
	points[pc_num, :, :2] = (np.random.rand(POINT_NUM, 2) - 0.5) * 64

	# compute the z coordinates
	points[pc_num, :, 2] = (-f[0] * points[pc_num, :, 0] - f[1] * points[pc_num, :, 1] - f[3]) / f[2]

	# add gaussian noise
	if NOISE_SIGMA:
		print("Added noise!")
		noise = np.random.randn(POINT_NUM, 3) * NOISE_SIGMA
		points[pc_num,...] = points[pc_num,...] + noise

	# compute ground truth normal
	normal_gt = f[:3] / np.linalg.norm(f[:3])
	ground_truth_normals[pc_num,...] = np.tile(normal_gt, POINT_NUM).reshape((POINT_NUM, 3))


# save
f = h5py.File(h5_filename, 'w')
f.create_dataset('data', data=points)
f.create_dataset('label', data=ground_truth_normals)

data, label = load_h5(h5_filename)
print(data.shape, label.shape)
