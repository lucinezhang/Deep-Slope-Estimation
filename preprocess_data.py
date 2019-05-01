import numpy as np

def compute_grid(points, xl, yl, zl):
	x_min  = np.min(points[:, 0])
	x_max  = np.max(points[:, 0])
	y_min  = np.min(points[:, 1])
	y_max  = np.max(points[:, 1])
	z_min  = np.min(points[:, 2])
	z_max  = np.max(points[:, 2])
	x_num = int((x_max - x_min) // xl)
	y_num = int((y_max - y_min) // yl)
	z_num = int((z_max - z_min) // zl)
	return x_min, y_min, z_min, x_num, y_num, z_num

def data_grid(points, normals):
	''' 
	points: n * 3 
	'''
	xl, yl, zl = 2.4, 2.4, 2.4
	grid_points_num = 25
	global_x_min, global_y_min, global_z_min, x_num, y_num, z_num = compute_grid(points, xl, yl, zl)

	# create the dictionary
	# grid_dic_points = {(x_start, y_start, z_start): []}
	grid_dic_points = {}
	grid_dic_normals = {}
	for x in range(x_num+1):
		for y in range(y_num+1):
			for z in range(z_num+1):
				grid_dic_points[(x, y, z)] = []
				grid_dic_normals[(x, y, z)] = []

	for i in range(points.shape[0]):
		x = points[i, 0]
		y = points[i, 1]
		z = points[i, 2]

		x_idx = int((x - global_x_min) // xl)
		y_idx = int((y - global_y_min) // yl)
		z_idx = int((z - global_z_min) // zl)

		grid_dic_points[(x_idx, y_idx, z_idx)].append(points[i, :])
		grid_dic_normals[(x_idx, y_idx, z_idx)].append(normals[i, :])

	filter_grid_points = []
	filter_grid_normals = []
	for k, v in grid_dic_points.items():
		if len(v) > 15 and len(v) < grid_points_num:
			num = len(v)
			grid_points = np.empty((grid_points_num, 3), dtype=np.float)
			grid_points[:num, :] = np.vstack(v)
			grid_points[num:, :] = np.tile(v[-1], (grid_points_num - num, 1))
			filter_grid_points.append(grid_points)

			normals = grid_dic_normals[k]
			grid_normals = np.empty((grid_points_num, 3), dtype=np.float)
			grid_normals[:num, :] = np.vstack(normals)
			grid_normals[num:, :] = np.tile(normals[-1], (grid_points_num - num, 1))
			filter_grid_normals.append(grid_normals)
		elif len(v) >= grid_points_num:
			num = len(v)
			idx = np.random.choice(num, grid_points_num, replace=False)
			grid_points = np.vstack(v)[idx, :]
			filter_grid_points.append(grid_points)

			normals = grid_dic_normals[k]
			grid_normals = np.vstack(normals)[idx, :]
			filter_grid_normals.append(grid_normals)

	filter_grid_points = np.array(filter_grid_points)
	filter_grid_normals = np.array(filter_grid_normals)

	return filter_grid_points, filter_grid_normals








