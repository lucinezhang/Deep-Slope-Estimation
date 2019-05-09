import numpy as np
import pykitti
import h5py
from open3d import *
import time


#9_26 1 2 5 9 11 48 59 106 9_29 0071   10.6GB  2753frames
#9_26 20 22 35 46 79                   4.9GB  1273frames
#9_26 15 28 32 52                      4.7GB  1218frames
#9_28 16 37 38 43 45                   2.3GB  602frames
#total:                                22.5GB 5846frames -> 87690samples


# Change this to the directory where you store KITTI data
basedir = '.'

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
    xl, yl, zl = 3, 3, 3
    grid_points_num = 30
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




def load_dataset(date, drive, calibrated=False, frame_range=None):
    """
    Loads the dataset with `date` and `drive`.

    Parameters
    ----------
    date        : Dataset creation date.
    drive       : Dataset drive.
    calibrated  : Flag indicating if we need to parse calibration data. Defaults to `False`.
    frame_range : Range of frames. Defaults to `None`.

    Returns
    -------
    Loaded dataset of type `raw`.
    """
    dataset = pykitti.raw(basedir, date, drive)

    # Load the data
    if calibrated:
        dataset.load_calib()  # Calibration data are accessible as named tuples

    np.set_printoptions(precision=4, suppress=True)
    print('\nDrive: ' + str(dataset.drive))
#    print('\nFrame range: ' + str(dataset.frames))

    if calibrated:
        print('\nIMU-to-Velodyne transformation:\n' + str(dataset.calib.T_velo_imu))
        print('\nGray stereo pair baseline [m]: ' + str(dataset.calib.b_gray))
        print('\nRGB stereo pair baseline [m]: ' + str(dataset.calib.b_rgb))

    return dataset



city = {'2011_09_26':['0001','0002','0005','0009','0011','0048','0059','0106'], '2011_09_29':['0071']}
resi = {'2011_09_26':['0020','0022','0035','0046','0079']}
road = {'2011_09_26':['0015','0028','0032','0052']}
campus = {'2011_09_28':['0016','0037','0038','0043','0045']}

city = {'2011_09_26':['0001']}
resi = {'2011_09_26':['0035']}
road = {'2011_09_26':['0028']}
campus = {'2011_09_28':['0016']}

cat = [city]

# f_city = h5py.File("city100.h5", 'w')
# f_resi = h5py.File("resi100.h5", 'w')
# f_road = h5py.File("road100.h5", 'w')
# f_campus = h5py.File("campus100.h5", 'w')
f_test = h5py.File("test.h5", 'w')
files = [f_test]

pc = []
normals = []



for clss_nb, clss in enumerate(cat):
    pc = []
    normals = []
    for date in clss.keys():
        for drive in clss[date]:
            dataset = load_dataset(date, drive)
            dataset_velo = list(dataset.velo)
            print(len(dataset_velo))
            for nb, frame in enumerate(dataset_velo):
                p = PointCloud()
                p.points = Vector3dVector(frame[:,:3])
                downpcd = voxel_down_sample(p, voxel_size = 0.3)
                estimate_normals(downpcd, search_param = KDTreeSearchParamHybrid(radius = 1.2, max_nn = 25))
                points_downsampled = np.array(downpcd.points)
                normals_downsampled = np.array(downpcd.normals)
                patch, patch_normal = data_grid(points_downsampled, normals_downsampled)
                # patch = (patch - np.expand_dims(np.mean(patch, axis=1), axis=1))/1.5
                pc.append(patch)
                normals.append(patch_normal)
                break

    pc = np.vstack(pc)
    normals = np.vstack(normals)
    # print(normals)

    print(str(clss_nb)+ ' pc shape: ', pc.shape)
    print(str(clss_nb)+ ' gt_normals shape: ', normals.shape)
    point_cloud = files[clss_nb].create_dataset("point_cloud", data = pc)
    gt_normals = files[clss_nb].create_dataset("gt_normals", data = normals)

#f.close()
for clss_nb, clss in enumerate(cat):
    files[clss_nb].close()


