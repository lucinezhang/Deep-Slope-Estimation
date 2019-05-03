from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import GeneratedDataset
from pointnet.model_new import PointNetDenseCls, feature_transform_regularizer, get_loss
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=24, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--outname', type=str, default='curv_no_noise', help='output name')
parser.add_argument('--model', type=str, default='curv_no_noise/model.pth', help='model path')
parser.add_argument('--input_transform', action='store_true', help="use input transform")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)


test_dataset = GeneratedDataset('/scratch/luxinz/test_'+opt.outname+'.h5')
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

print(len(test_dataset))


blue = lambda x: '\033[94m' + x + '\033[0m'

classifier = PointNetDenseCls(k=3, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

classifier.cuda()
classifier = classifier.eval()

num_batch = len(test_dataset) / opt.batchSize

preds = []
targets = []
for i, data in enumerate(test_loader, 0):

    points, target = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()

    pred, trans, trans_feat = classifier(points)
    
    loss, rms_error = get_loss(pred, target, trans)

    if trans is not None:
        pred = torch.bmm(pred, trans.transpose(2, 1))

    preds.append(pred.detach().cpu().numpy())
    targets.append(target.detach().cpu().numpy())

    print('[%d/%d] loss: %f rms_error: %f' % (i, num_batch, loss.item(), rms_error.item()))

preds = np.vstack(preds)
targets = np.vstack(targets)
print(preds.shape)

preds = preds / np.linalg.norm(preds, axis=2, keepdims=True)

cos_angle = np.abs(np.sum(np.multiply(targets, preds), axis=2))
cos_angle = np.clip(cos_angle, 0, 1.0)
angle_dif = np.arccos(cos_angle) / 3.14 * 180

loss = np.sqrt(np.mean(np.square(angle_dif)))

print("RMS angle error:", loss)

np.save(opt.outname+'_pred', preds)

