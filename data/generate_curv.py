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
	print("Usage: python generate_curv.py point_num point_cloud_num gaussian_noise_sigma")
	exit()

POINT_NUM = int(sys.argv[1])
PCLOUD_NUM = int(sys.argv[2])
NOISE_SIGMA = float(sys.argv[3])
h5_filename = 'test_curv.h5'

points = np.empty((PCLOUD_NUM, POINT_NUM, 3), dtype=np.float)
ground_truth_normals = np.empty((PCLOUD_NUM, POINT_NUM, 3), dtype=np.float)

for pc_num in range(PCLOUD_NUM):
	# generate parameters of the 3D surface
	f = (np.random.rand(11) - 0.5) * 10

	# generate x and y coordinates randomly
	points[pc_num, :, :2] = (np.random.rand(POINT_NUM, 2) - 0.5) * 64

	x = points[pc_num, :, 0]
	y = points[pc_num, :, 1]

	# compute the z coordinates ax^3 + bx^2y + c .... + z + constant = 0
	points[pc_num, :, 2] = -(f[0]*x*x*x + f[1]*x*x*y + f[2]*x*y*y + f[3]*y*y*y + f[4]*x*x + f[5]*x*y + f[6]*y*y + f[7]*x + f[8]*y + f[10]) / (f[9]*1000)
	z = points[pc_num, :, 2]

	# add gaussian noise
	if NOISE_SIGMA:
		print("Added noise!")
		noise = np.random.randn(POINT_NUM, 3) * NOISE_SIGMA
		points[pc_num,...] = points[pc_num,...] + noise

	# compute and assign ground truth normals to each point
	ground_truth_normals[pc_num, :, 0] = 3*f[0]*x*x + 2*f[1]*x*y + f[2]*y*y + 2*f[4]*x + f[5]*y + f[7]
	ground_truth_normals[pc_num, :, 1] = f[1]*x*x + 2*f[2]*x*y + 3*f[3]*y*y + f[5]*x + 2*f[6]*y + f[8]
	ground_truth_normals[pc_num, :, 2] = f[9]*1000

	ground_truth_normals[pc_num, :, :] = ground_truth_normals[pc_num, :, :] / np.linalg.norm(ground_truth_normals[pc_num, :, :], axis=1, keepdims=True)


# save
f = h5py.File(h5_filename, 'w')
f.create_dataset('data', data=points)
f.create_dataset('label', data=ground_truth_normals)

data, label = load_h5(h5_filename)
print(data.shape, label.shape)

