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
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

train_dataset = GeneratedDataset('/scratch/luxinz/train_plane_no_noise.h5')
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

val_dataset = GeneratedDataset('/scratch/luxinz/val_plane_no_noise.h5')
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(train_dataset), len(val_dataset))

try:
    os.makedirs(opt.outf)
except OSError:
    pass

writer = SummaryWriter(opt.outf)

blue = lambda x: '\033[94m' + x + '\033[0m'

classifier = PointNetDenseCls(k=3, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8000, gamma=0.5)
classifier.cuda()

num_batch = len(train_dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(train_loader, 0):
        step = epoch * num_batch + i

        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        
        loss, rms_error = get_loss(pred, target, trans)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()

        print('[%d: %d/%d] train loss: %f rms_error: %f' % (epoch, i, num_batch, loss.item(), rms_error.item()))
        writer.add_scalar('train/loss', loss.item(), step)
        writer.add_scalar('train/rms_error', rms_error.item(), step)

        if i % 10 == 0:
            j, data = next(enumerate(val_loader, 0))
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()

            classifier = classifier.eval()
            pred, trans, _ = classifier(points)
            
            loss, rms_error = get_loss(pred, target, trans)

            print('[%d: %d/%d] %s loss: %f rms_error: %f' % (epoch, i, num_batch, blue('test'), loss.item(), rms_error.item()))
            writer.add_scalar('val/loss', loss.item(), step)
            writer.add_scalar('val/rms_error', rms_error.item(), step)

if opt.feature_transform:
    model_name = "model_feature_transform"
else:
    model_name = "model"
torch.save(classifier.state_dict(), '%s/%s.pth' % (opt.outf, model_name))

