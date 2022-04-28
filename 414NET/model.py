import os
import torch
import torch.nn as nn
import torch.nn.parallel as P
import torch.nn.functional as F
import torch.utils.data as D
import numpy as np
import json
import argparse
from tqdm import tqdm

class Y3DNet(nn.Module):
    def __init__(self):
        super(Y3DNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        grad_tensor = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))
        grad_tensor = torch.autograd.Variable(grad_tensor)
        grad_tensor = grad_tensor.view(1, 9).repeat(inputs.size()[0], 1)
        grad_tensor = grad_tensor.cuda() if inputs.is_cuda else grad_tensor

        inputs = self.bn1(self.conv1(inputs))
        inputs = F.relu(inputs)

        inputs = self.bn2(self.conv2(inputs))
        inputs = F.relu(inputs)

        inputs = self.bn3(self.conv3(inputs))
        inputs = F.relu(inputs)

        inputs = torch.max(inputs, 2, True)[0].view(-1, 1024)

        inputs = self.bn4(self.fc1(inputs))
        inputs = F.relu(inputs)

        inputs = self.bn5(self.fc2(inputs))
        inputs = F.relu(inputs)

        inputs = self.fc3(inputs)

        inputs += grad_tensor
        return inputs.view(-1, 3, 3)

class FeatureNet(nn.Module):
    def __init__(self, flag = True):
        super(FeatureNet, self).__init__()
        self.yee = Y3DNet()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.flag = flag

    def forward(self, inputs):
        points_num = inputs.size()[2]

        mat2 = self.yee(inputs)
        inputs = inputs.transpose(2, 1)

        inputs = torch.bmm(inputs, mat2)

        inputs = inputs.transpose(2, 1)
        
        inputs = self.bn1(self.conv1(inputs))
        inputs = F.relu(inputs)
        features = inputs

        inputs = self.bn2(self.conv2(inputs))
        inputs = F.relu(inputs)

        inputs = self.bn3(self.conv3(inputs))
        
        inputs = torch.max(inputs, 2, True)[0].view(-1, 1024)

        if not self.flag:
            inputs = inputs.view(-1, 1024, 1).repeat(1, 1, points_num)
            inputs = torch.cat([inputs, features], 1)
        
        return inputs, mat2
    
class SegmentNet(nn.Module):
    def __init__(self, out = 2):
        super(SegmentNet, self).__init__()
        self.out = out
        self.features = FeatureNet(flag=False)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.out, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, inputs):
        points_num = inputs.size()[2]
        batch = inputs.size()[0]

        inputs, mat2 = self.features(inputs)

        inputs = self.bn1(self.conv1(inputs))
        inputs = F.relu(inputs)

        inputs = self.bn2(self.conv2(inputs))
        inputs = F.relu(inputs)

        inputs = self.bn3(self.conv3(inputs))
        inputs = F.relu(inputs)

        inputs = self.conv4(inputs)

        inputs = inputs.transpose(2, 1).contiguous()
        inputs = F.log_softmax(inputs.view(-1, self.out), dim=-1)

        inputs = inputs.view(batch, points_num, self.out)
        return inputs, mat2

class DatasetGenerator(D.Dataset):
    def __init__(self, path, train_flag=True, object_type=None, run_type='train'):
        self.category_dict = {}
        self.train_flag = train_flag

        with open(os.path.join(path, 'synsetoffset2category.txt'), 'r') as lines:
            for line in lines:
                pair = line.strip().split()
                if pair[0] == object_type:
                    self.category_dict[object_type] = pair[1]
        print("cat_dict", self.category_dict)

        self.reverse_category_dict = dict((value, key) for key, value in self.category_dict.items())
        print("reverse_dict", self.reverse_category_dict)

        train_test_split = os.path.join(path, 'train_test_split', 'shuffled_%s_file_list.json' % run_type)
        files = json.load(open(train_test_split, 'r'))
        
        self.data_path = []
        for file in files:
            folder, category, uuid = file.split('/')
            if category in self.category_dict.values():
                self.data_path.append((object_type, 
                                        os.path.join(path, category, 'points', uuid + '.pts'), 
                                        os.path.join(path, category, 'points_label', uuid + '.seg')))
        print("data path", self.data_path)

        self.seg_dict = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as lines:
            for line in lines:
                seg_pair = line.strip().split()
                self.seg_dict[seg_pair[0]] = int(seg_pair[1])
        
        self.num_seg_dict = self.seg_dict[object_type]
        
        print("seg_dict", self.seg_dict)
        print("num_seg_dict", self.num_seg_dict)

    def __getitem__(self, index):
        path = self.data_path[index]
        points = np.loadtxt(path[1]).astype(np.float32)
        segment = np.loadtxt(path[2]).astype(np.int64)

        choice = np.random.choice(len(segment), 2500, replace=True)

        points = points[choice, :]

        points = points - np.expand_dims(np.mean(points, axis=0), 0)
        distance = np.max(np.sqrt(np.sum(points**2, axis=1)), 0)
        points = points / distance

        if self.train_flag:
            theta = np.random.uniform(0, np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            points[:,[0,2]] = points[:,[0,2]].dot(rotation_matrix)
            points += np.random.normal(0, 0.02, size=points.shape)

        points = torch.from_numpy(points)
        segment = torch.from_numpy(segment[choice])
        
        return points, segment

    def __len__(self):
        return len(self.data_path)