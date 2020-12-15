import io
import mysql.connector
import cv2
import torch
import torchvision
import os
import math
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
import numpy as np
import torch.nn as nn
from datetime import datetime
from torchvision import transforms
from detect import Detect
device = torch.device('cuda')


class VggFace(nn.Module):
    def __init__(self):
        super(VggFace, self).__init__()
        self.vgg = torchvision.models.vgg16()
        self.vgg.classifier[6] = torch.nn.Linear(
            in_features=4096, out_features=2622, bias=True)
        self.linear1 = nn.Linear(in_features=5244, out_features=100, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.linear2 = nn.Linear(in_features=100, out_features=1, bias=True)
        self.sigmoid1 = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def forward(self, input1, input2, classname):
        output1 = self.vgg(input1)
        output2 = self.vgg(input2)
        target = torch.cat([output1, output2], dim=1)
        target = self.linear1(target)
        target = self.relu1(target)
        target = self.dropout1(target)
        target = self.linear2(target)
        target = self.sigmoid1(target)
        target = target.view(-1)
        if classname == 0:
            loss = self.loss(target, torch.FloatTensor(
                target.shape[0]).fill_(0).to(device))
        else:
            loss = self.loss(target, torch.FloatTensor(
                target.shape[0]).fill_(1).to(device))
        return loss


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, root='/home/dung/Project/AI/train-faces'):
        self.folder = []
        heads = os.listdir(root)
        for i, head in enumerate(heads[:100]):
            mids = os.listdir(f'{root}/{head}')
            for mid in mids:
                imgs = os.listdir(f'{root}/{head}/{mid}')
                if len(imgs) > 1:
                    self.folder.append(f'{root}/{head}/{mid}')

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, idx):
        imgs = os.listdir(self.folder[idx])
        ims = []
        for img in imgs:
            im = cv2.imread(f'{self.folder[idx]}/{img}')
            im = cv2.resize(im, (224, 224))/225
            ims.append(im)
        imgs = torch.tensor(ims, dtype=torch.float32)
        imgs = imgs.permute(0, 3, 1, 2)
        input = imgs[0]
        output = imgs[1:]
        return input, output


class DifferenceDataset(torch.utils.data.Dataset):
    def __init__(self, not_idx=None, root='/home/dung/Project/AI/train-faces'):
        self.folder = []
        heads = os.listdir(root)
        for i, head in enumerate(heads[:100]):
            if i == not_idx:
                continue
            mids = os.listdir(f'{root}/{head}')
            for mid in mids:
                imgs = os.listdir(f'{root}/{head}/{mid}')
                if len(imgs) > 1:
                    self.folder.append(f'{root}/{head}/{mid}')

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, idx):
        imgs = os.listdir(self.folder[idx])
        ims = []
        for img in imgs:
            im = cv2.imread(f'{self.folder[idx]}/{img}')
            im = cv2.resize(im, (224, 224))/225
            ims.append(im)
        imgs = torch.tensor(ims, dtype=torch.float32)
        imgs = imgs.permute(0, 3, 1, 2)
        return imgs


detect = Detect()
model = VggFace()
model.to(device)
model.train()
# model.load_state_dict(torch.load('recognize_2.pth'))
# loss = model(i1.to(device), i2.to(device))
# i1 = cv2.imread('2.jpg')
# i2 = cv2.imread('3.jpg')
# i1 = detect(i1)
# i1 = torch.tensor(i1, dtype=torch.float32)
# i1 = i1.permute(2, 0, 1).to(device)
# i2 = detect(i2)
# i2 = torch.tensor(i2, dtype=torch.float32).to(device)
# i2 = i2.permute(2, 0, 1).to(device)
# i1 = torch.unsqueeze(i1, 0)
# i2 = torch.unsqueeze(i2, 0)
# loss = model(i1, i2)

dataset = FaceDataset()
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=0)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

epochs = 1000
for epoch in range(epochs):
    print(f'epoch {epoch} running')
    loss = None
    for i, (input, output) in enumerate(data_loader):

        loss = model(input.repeat(
            output.shape[1], 1, 1, 1).to(device), output[0].to(device), 0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        difference = DifferenceDataset(not_idx=i)
        difference_loader = torch.utils.data.DataLoader(
            difference, batch_size=1, shuffle=False, num_workers=0)
        for i, diff in enumerate(difference_loader):
            loss = model(input.repeat(diff.shape[1], 1, 1, 1).to(
                device), diff[0].to(device), 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'loss = {loss}')
        torch.save(model.state_dict(), 'recognize_1.pth')
