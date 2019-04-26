from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import GeneratedDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer, get_loss
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=24, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='plane_no_noise', help='output folder')
parser.add_argument('--model', type=str, default='plane_no_noise/model_feature_transform.pth', help='model path')
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)


test_dataset = GeneratedDataset('/scratch/luxinz/test_plane_no_noise.h5')
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

print(len(test_dataset))

try:
    os.makedirs(opt.outf)
except OSError:
    pass

writer = SummaryWriter(opt.outf)

blue = lambda x: '\033[94m' + x + '\033[0m'

classifier = PointNetDenseCls(k=3, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

classifier.cuda()
classifier = classifier.eval()

num_batch = len(test_dataset) / opt.batchSize

preds = []
for i, data in enumerate(test_loader, 0):

    points, target = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()

    pred, trans, trans_feat = classifier(points)
    
    loss, rms_error = get_loss(pred, target, trans)

    if trans is not None:
        pred = torch.bmm(pred, trans.transpose(2, 1))

    preds.append(pred.detach().cpu().numpy())

    print('[%d/%d] loss: %f rms_error: %f' % (i, num_batch, loss.item(), rms_error.item()))

preds = np.vstack(preds)
print(preds.shape)
np.save(opt.outf+'_pred', preds)

