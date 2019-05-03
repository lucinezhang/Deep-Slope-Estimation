from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import *
from pointnet.model_new import *
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=24, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch_1', type=int, default=10, help='number of epochs to train for the hint loss')
parser.add_argument('--nepoch_2', type=int, default=50, help='number of epochs to train for the final loss')
parser.add_argument('--outf', type=str, default='curv_no_noise_kd', help='output folder')
parser.add_argument('--teacher_model', type=str, default='curv_no_noise/model.pth', help='model path')
parser.add_argument('--input_transform', action='store_true', help="use input transform")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--eval_interval', type=int, default=10, help="interval of evaluation on val set")

opt = parser.parse_args()
print(opt)

# opt.manualSeed = random.randint(1, 10000)  # fix seed
# print("Random Seed: ", opt.manualSeed)
# random.seed(opt.manualSeed)
# torch.manual_seed(opt.manualSeed)

train_dataset = GeneratedDataset('/scratch/luxinz/train_curv_no_noise.h5')
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

val_dataset = GeneratedDataset('/scratch/luxinz/val_curv_no_noise.h5')
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

if opt.feature_transform:
    model_name = "model_feature_transform"
else:
    model_name = "model"

blue = lambda x: '\033[94m' + x + '\033[0m'

teacher = PointNetDenseCls(k=3, global_feat=True, input_transform=opt.input_transform, feature_transform=opt.feature_transform)
student = StudentNetDenseCls(k=3)

if opt.teacher_model != '':
    teacher.load_state_dict(torch.load(opt.teacher_model))
else:
    print("No trained teacher model loaded!!")

# stage 1

optimizer = optim.Adam([
    {'params': student.feat.parameters()}, 
    {'params': student.fc.parameters()}], lr=0.001, betas=(0.9, 0.999))
lr_lambda = lambda batch: max(0.5 ** ((batch * opt.batchSize) // 8000), 0.00001)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
teacher.cuda()
student.cuda()

num_batch = len(train_dataset) / opt.batchSize

for epoch in range(opt.nepoch_1):
    for i, data in enumerate(train_loader, 0):
        scheduler.step()

        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        optimizer.zero_grad()
        teacher = teacher.eval()
        student = student.train()
        pc_feat_teacher, _, _, _ = teacher(points)
        pc_feat_student, pc_feat_teacher, _ = student(points, pc_feat_teacher)
        
        loss = get_hint_loss(pc_feat_teacher, pc_feat_student)
        loss.backward()
        optimizer.step()

        print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.item()))

# stage 2

optimizer = optim.Adam(student.parameters(), lr=0.001, betas=(0.9, 0.999))
lr_lambda = lambda batch: max(0.5 ** ((batch * opt.batchSize) // 8000), 0.00001)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

num_batch = len(train_dataset) / opt.batchSize
best_error = 10000

for epoch in range(opt.nepoch_2):
    for i, data in enumerate(train_loader, 0):
        scheduler.step()
        step = epoch * num_batch + i
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)

        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        optimizer.zero_grad()
        student = student.train()
        _, _, pred = student(points)
        
        loss, rms_error = get_loss(pred, target)
        loss.backward()
        optimizer.step()

        print('[%d: %d/%d] train loss: %f rms_error: %f' % (epoch, i, num_batch, loss.item(), rms_error.item()))
        writer.add_scalar('train/loss', loss.item(), step)
        writer.add_scalar('train/rms_error', rms_error.item(), step)

        if i % opt.eval_interval == 0:
            j, data = next(enumerate(val_loader, 0))
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()

            student = student.eval()
            _, _, pred = student(points)

            loss, rms_error = get_loss(pred, target)

            print('[%d: %d/%d] %s loss: %f rms_error: %f' % (epoch, i, num_batch, blue('test'), loss.item(), rms_error.item()))
            writer.add_scalar('val/loss', loss.item(), step)
            writer.add_scalar('val/rms_error', rms_error.item(), step)

            if rms_error.item() < best_error:
                best_error = rms_error.item()
                torch.save(student.state_dict(), '%s/%s.pth' % (opt.outf, model_name))


