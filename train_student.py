from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import GeneratedDataset, KittiNormalEst
from pointnet.model_new import *
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='kitti_student', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--eval_interval', type=int, default=10, help="interval of evaluation on val set")

opt = parser.parse_args()
print(opt)


# train_dataset = GeneratedDataset('/scratch/luxinz/train_'+opt.outf+'.h5')
train_dataset = KittiNormalEst(stage='train')
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

# val_dataset = GeneratedDataset('/scratch/luxinz/val_'+opt.outf+'.h5')
val_dataset = KittiNormalEst(stage='val')
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

classifier = StudentNetDenseCls(k=3)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
lr_lambda = lambda batch: max(0.5 ** ((batch * opt.batchSize) // 8000), 0.00001)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
classifier.cuda()

num_batch = len(train_dataset) / opt.batchSize
best_error = 10000

for epoch in range(opt.nepoch):
    for i, data in enumerate(train_loader, 0):
        scheduler.step()
        step = epoch * num_batch + i
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)

        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        optimizer.zero_grad()
        classifier = classifier.train()
        _, _, pred = classifier(points)
        
        loss, rms_error = get_loss(pred, target)
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print('[%d: %d/%d] train loss: %f rms_error: %f' % (epoch, i, num_batch, loss.item(), rms_error.item()))
        writer.add_scalar('train/loss', loss.item(), step)
        writer.add_scalar('train/rms_error', rms_error.item(), step)

        if i % opt.eval_interval == 0:
            j, data = next(enumerate(val_loader, 0))
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()

            classifier = classifier.eval()
            start = time.time()
            _, _, pred = classifier(points)
            end = time.time()
            loss, rms_error = get_loss(pred, target)

            print('[%d: %d/%d] %s loss: %f rms_error: %f, inference time: %f' % (epoch, i, num_batch, blue('test'), loss.item(), rms_error.item(), end-start))
            writer.add_scalar('val/loss', loss.item(), step)
            writer.add_scalar('val/rms_error', rms_error.item(), step)

            if rms_error.item() < best_error:
                best_error = rms_error.item()
                torch.save(classifier.state_dict(), '%s/model.pth' % opt.outf)

