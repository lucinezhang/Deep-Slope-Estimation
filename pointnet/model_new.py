from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0]).astype(np.float32))).view(1,4).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden

        # convert quaternion to rotation matrix
        x = batch_quat_to_rotmat(x)

        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = False, input_transform = False, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.conv1a = nn.Conv1d(3, 64, 1)
        self.conv1b = nn.Conv1d(64, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 128)
        self.bn1a = nn.BatchNorm1d(64)
        self.bn1b = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)
        self.global_feat = global_feat
        self.input_transform = input_transform
        self.feature_transform = feature_transform
        if self.input_transform:
            self.stn = STN3d()
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        if self.input_transform:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        else:
            trans = None
        
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        points_feat = F.relu(self.bn4(self.conv4(x)))

        pc_feat = torch.max(points_feat, 2, keepdim=True)[0]
        pc_feat = pc_feat.view(-1, 1024)
        pc_feat = F.relu(self.bn5(self.fc1(pc_feat)))
        pc_feat = F.relu(self.bn6(self.fc2(pc_feat)))
        pc_feat_repeat = pc_feat.view(-1, 128, 1).repeat(1, 1, n_pts)


        if self.global_feat:
            return pc_feat, torch.cat([points_feat, pc_feat_repeat], 1), trans, trans_feat
        else:
            return torch.cat([points_feat, pc_feat_repeat], 1), trans, trans_feat

class PointNetDenseCls(nn.Module):
    def __init__(self, k = 3, global_feat=False, input_transform=False, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.global_feat = global_feat
        self.input_transform = input_transform
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=global_feat, input_transform=input_transform, feature_transform=feature_transform)
        self.conv1 = nn.Conv1d(1152, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, 128, 1)
        self.conv5 = nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]

        if self.global_feat:
            pc_feat, x, trans, trans_feat = self.feat(x)
        else:
            x, trans, trans_feat = self.feat(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.dropout(self.conv4(x))))
        x = self.conv5(x)
        x = x.transpose(2,1)

        if self.global_feat:
            return pc_feat, x, trans, trans_feat
        else:
            return x, trans, trans_feat

class StudentNetfeat(nn.Module):
    def __init__(self):
        super(StudentNetfeat, self).__init__()
        self.conv1a = nn.Conv1d(3, 16, 1)
        self.conv1b = nn.Conv1d(16, 16, 1)
        self.conv2 = nn.Conv1d(16, 16, 1)
        self.conv3 = nn.Conv1d(16, 32, 1)
        self.conv4 = nn.Conv1d(32, 256, 1)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 32)
        self.bn1a = nn.BatchNorm1d(16)
        self.bn1b = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(32)

    def forward(self, x):
        n_pts = x.size()[2]
        
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        points_feat = F.relu(self.bn4(self.conv4(x)))

        pc_feat = torch.max(points_feat, 2, keepdim=True)[0]
        pc_feat = pc_feat.view(-1, 256)
        pc_feat = F.relu(self.bn5(self.fc1(pc_feat)))
        pc_feat = F.relu(self.bn6(self.fc2(pc_feat)))

        pc_feat_repeat = pc_feat.view(-1, 32, 1).repeat(1, 1, n_pts)
        return pc_feat, torch.cat([points_feat, pc_feat_repeat], 1)

class StudentNetDenseCls(nn.Module):
    def __init__(self, k = 3):
        super(StudentNetDenseCls, self).__init__()
        self.k = k
        self.feat = StudentNetfeat()
        self.fc = nn.Linear(128, 32)
        self.conv1 = nn.Conv1d(288, 64, 1)
        self.conv2 = nn.Conv1d(64, 32, 1)
        self.conv3 = nn.Conv1d(32, 16, 1)
        self.conv4 = nn.Conv1d(16, 16, 1)
        self.conv5 = nn.Conv1d(16, self.k, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(16)
        self.bn_fc = nn.BatchNorm1d(32)

    def forward(self, x, pc_feat_teacher=None):
        batchsize = x.size()[0]
        n_pts = x.size()[2]

        pc_feat_student, x = self.feat(x)
        if pc_feat_teacher is not None:
            pc_feat_teacher = F.relu(self.bn_fc(self.fc(pc_feat_teacher)))

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        x = x.transpose(2,1)
        return pc_feat_student, pc_feat_teacher, x


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

def get_loss(pred, label, patch_rot=None):
    if patch_rot is not None:
        pred = torch.bmm(pred, patch_rot.transpose(2, 1))
    
    label = label / torch.norm(label, p=2, dim=2, keepdim=True)
    pred = pred / torch.norm(pred, p=2, dim=2, keepdim=True)

    cos_angle = torch.abs(torch.sum(pred * label, dim=2))
    loss = torch.mean((1. - cos_angle).pow(2))

    cos_angle = torch.clamp(cos_angle, 0., 1.)
    angle_dif = torch.acos(cos_angle) / 3.14 * 180
    rms_error = torch.sqrt(torch.mean(angle_dif.pow(2)))

    return loss, rms_error

def get_hint_loss(pc_feat_teacher, pc_feat_student):
    return F.mse_loss(pc_feat_student, pc_feat_teacher)

# quaternion a + bi + cj + dk should be given in the form [a,b,c,d]
def batch_quat_to_rotmat(q, out=None):

    batchsize = q.size(0)

    if out is None:
        out = q.new_empty(batchsize, 3, 3)

    # 2 / squared quaternion 2-norm
    s = 2/torch.sum(q.pow(2), 1)

    # coefficients of the Hamilton product of the quaternion with itself
    h = torch.bmm(q.unsqueeze(2), q.unsqueeze(1))

    out[:, 0, 0] = 1 - (h[:, 2, 2] + h[:, 3, 3]).mul(s)
    out[:, 0, 1] = (h[:, 1, 2] - h[:, 3, 0]).mul(s)
    out[:, 0, 2] = (h[:, 1, 3] + h[:, 2, 0]).mul(s)

    out[:, 1, 0] = (h[:, 1, 2] + h[:, 3, 0]).mul(s)
    out[:, 1, 1] = 1 - (h[:, 1, 1] + h[:, 3, 3]).mul(s)
    out[:, 1, 2] = (h[:, 2, 3] - h[:, 1, 0]).mul(s)

    out[:, 2, 0] = (h[:, 1, 3] - h[:, 2, 0]).mul(s)
    out[:, 2, 1] = (h[:, 2, 3] + h[:, 1, 0]).mul(s)
    out[:, 2, 2] = 1 - (h[:, 1, 1] + h[:, 2, 2]).mul(s)

    return out


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
