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
from detect import Detect
from torchvision import transforms

detect = Detect()


class VggFace(nn.Module):
    def __init__(self):
        super(VggFace, self).__init__()
        self.vgg = torchvision.models.vgg16()
        self.vgg.classifier[6] = torch.nn.Linear(
            in_features=4096, out_features=2622, bias=True)

    def forward(self, input1, input2):
        if self.training:
            output1 = self.vgg(input1)
            output2 = self.vgg(input2)
            return nn.L1Loss()(output1, output2)
        else:
            return self.vgg(input1)


class FaceRecognizeDataset(torch.utils.data.Dataset):
    def __init__(self, folder='/home/dung/Project/AI/archive/training', transform=None):
        self.folder = folder
        self.files = os.listdir(folder)
        self.a = len(self.files)

    def __len__(self):
        return int(len(self.files)/3)

    def __getitem__(self, idx):
        extracts = []
        for file in self.files[idx*3:(idx+1)*3]:
            img = cv2.imread(f'{self.folder}/{file}')
            extracts.append(detect(img))
        return extracts


device = torch.device('cuda')
model = VggFace()
model.to(device)
model.train()
dataset = FaceRecognizeDataset()
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=0)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
epochs = 1000
for epoch in range(epochs):
    print(f'epoch {epoch} running')
    for i, extracts in enumerate(data_loader):
        loss = None
        for extract in extracts[1:3]:
            loss = model(extracts[0], extract)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if(loss != None):
            print(f'mean loss in folder {i} = {loss}')
        torch.save(model.state_dict(), 'recognize_1.pth')
