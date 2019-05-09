from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import GeneratedDataset, KittiNormalEst
from pointnet.model_new import PointNetDenseCls, feature_transform_regularizer, get_loss
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
import h5py



parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=500, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--model', type=str, default='kitti_output/model.pth', help='model path')
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--thres', type=float, default=1.0, help='threshold for weight pruning')

opt = parser.parse_args()
print(opt)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_h5_kitti(h5_filename):
    f = h5py.File(h5_filename)
    data = f['point_cloud'][:]
    label = f['gt_normals'][:]
    return data, label

h5_test = 'test.h5'
data_test, label_test = load_h5_kitti(h5_test)
data_test_normal = (data_test - np.expand_dims(np.mean(data_test, axis=1), axis=1))/1.5


current_points = torch.from_numpy(data_test_normal.copy()).type(torch.FloatTensor)
current_labels = torch.from_numpy(label_test.copy()).type(torch.FloatTensor)



if opt.feature_transform:
    model_name = "model_feature_transform"
else:
    model_name = "model"

blue = lambda x: '\033[94m' + x + '\033[0m'

classifier = PointNetDenseCls(k=3, feature_transform=opt.feature_transform)

classifier.load_state_dict(torch.load(opt.model))
#classifier.load_state_dict(torch.load('model30_3189.pth'))

classifier.cuda()

points = current_points.transpose(2, 1)

#print(classifier)


points, target = points.cuda(), current_labels.cuda()
classifier = classifier.eval()
with torch.no_grad():
    start = time.time()
    pred, trans, trans_feat = classifier(points)
    end = time.time()
loss, rms_error = get_loss(pred, target, trans)
pred = torch.bmm(pred, trans.transpose(2,1))

#print the original results
print('time: {:.4f}, error: {:.4f}, pred shape: {}'.format(end-start, rms_error.item(), pred.shape))
np.savez('res.npz', points = data_test, pred=pred.detach().cpu().numpy())

#prune the weights
print('param number: ', count_parameters(classifier))
print('weight pruning...')
all_mask = []
params=classifier.state_dict()
for k,v in params.items():
#    print(k)
    if 'conv' in k or 'fc' in k:
        mask = np.abs(v.cpu().detach().numpy()) >= np.std(v.cpu().detach().numpy())*float(opt.thres)
        all_mask.append(mask)
        params[k] *= torch.Tensor(mask.astype(np.float32)).cuda()
classifier.load_state_dict(params)


#compute the results after weight pruning
points, target = points.cuda(), current_labels.cuda()
classifier = classifier.eval()
with torch.no_grad():
    start = time.time()
    pred, trans, trans_feat = classifier(points)
    end = time.time()
loss, rms_error = get_loss(pred, target, trans)
pred = torch.bmm(pred, trans.transpose(2,1))

print('time: {:.4f}, error: {:.4f}, pred shape: {}'.format(end-start, rms_error.item(), pred.shape))

pruned = 0
for each in all_mask:
    pruned += (1-each).sum()
print('pruned: ', pruned)
np.savez('res_pruned.npz', points = data_test, pred=pred.detach().cpu().numpy())



