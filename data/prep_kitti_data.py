import numpy as np
import h5py
from open3d import *
import time
from preprocess_data import *


#9_26 1 2 5 9 11 48 59 106 9_29 0071   10.6GB  2753frames
#9_26 20 22 35 46 79                   4.9GB  1273frames
#9_26 15 28 32 52                      4.7GB  1218frames
#9_28 16 37 38 43 45                   2.3GB  602frames
#total:                                22.5GB 5846frames -> 87690samples  


# Change this to the directory where you store KITTI data
basedir = '.'


city = {'2011_09_26':['0001','0002','0005','0009','0011','0048','0059','0106'], '2011_09_29':['0071']}
resi = {'2011_09_26':['0020','0022','0035','0046','0079']}
road = {'2011_09_26':['0015','0028','0032','0052']}
campus = {'2011_09_28':['0016','0037','0038','0043','0045']}


cat = [city, resi, road, campus]

f_city = h5py.File("city30.h5", 'w')
f_resi = h5py.File("resi30.h5", 'w')
f_road = h5py.File("road30.h5", 'w')
f_campus = h5py.File("campus30.h5", 'w')
files = [f_city, f_resi, f_road, f_campus]


for clss_nb, clss in enumerate(cat):
    pc = []
    normals = []
    for date in clss.keys():
        for drive in clss[date]:
            dataset = load_dataset(date, drive)
            dataset_velo = list(dataset.velo)
            print(len(dataset_velo))
            for nb, frame in enumerate(dataset_velo):
                if nb%50==0 :
                    print(nb)
                p = PointCloud()
                p.points = Vector3dVector(frame[:,:3])
                downpcd = voxel_down_sample(p, voxel_size = 0.3)
                estimate_normals(downpcd, search_param = KDTreeSearchParamHybrid(radius = 1.2, max_nn = 25))
                points_downsampled = np.array(downpcd.points)
                normals_downsampled = np.array(downpcd.normals)
                patch, patch_normal = data_grid(points_downsampled, normals_downsampled)
                patch = (patch - np.expand_dims(np.mean(patch, axis=1), axis=1))/1.5
                
                pc.append(patch)
                normals.append(patch_normal)

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



