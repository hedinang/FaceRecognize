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


class Detect(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = torchvision.models.vgg16(pretrained=False).features
        backbone.out_channels = 512
        anchor_sizes = ((8, 16, 32, 64, 128, 256, 512),)
        aspect_ratios = ((1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/math.sqrt(2), 1,
                          2, math.sqrt(2), 3, 4, 5, 6, 7, 8),)
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3', '4'],
                                                        output_size=7,
                                                        sampling_ratio=2)
        self.model = FasterRCNN(backbone,
                                num_classes=2,
                                rpn_anchor_generator=anchor_generator,
                                box_roi_pool=roi_pooler,
                                box_score_thresh=0.95)
        self.device = torch.device('cuda')
        self.model.load_state_dict(torch.load('3.pth'))
        self.model.to(self.device)
        self.model.eval()

    def forward(self, img):
        im = img
        img = torch.tensor(img, dtype=torch.float32)/255
        img = torch.unsqueeze(img, 0)
        img = img.permute((0, 3, 1, 2))
        img = img.to(self.device)
        output = self.model(img)
        boxes = output[0]['boxes'].detach().cpu().numpy()
        if len(boxes) != 1:
            return None
        else:
            x0, y0, x1, y1 = boxes[0]
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            im = cv2.resize(im[y0:y1, x0:x1, :], (224, 224))/255
            im = torch.tensor(im, dtype=torch.float32)
            im = im.permute(2, 0, 1)
            return im


detect = Detect()


class VggFace(nn.Module):
    def __init__(self):
        super(VggFace, self).__init__()
        self.vgg = torchvision.models.vgg16()
        print(self.vgg)
        self.vgg.classifier[6] = torch.nn.Linear(
            in_features=4096, out_features=2622, bias=True)

    def forward(self, input1, input2, classname):
        output1 = self.vgg(input1)
        output2 = self.vgg(input2)
        target = torch.cat([output1, output2], dim=1)
        target = nn.Linear(
            in_features=5244, out_features=100, bias=True)(target)
        target = nn.ReLU(inplace=True)(target)
        target = nn.Dropout(p=0.5, inplace=False)(target)
        target = nn.Linear(
            in_features=100, out_features=1, bias=True)(target)
        target = nn.Sigmoid()(target)
        if self.training:
            return nn.L1Loss()(target, classname)
        else:
            return target


class SimilarityFaceDataset(torch.utils.data.Dataset):
    def __init__(self, folder='/home/dung/AI/archive/training'):
        self.folder = folder
        self.files = os.listdir(folder)
        self.a = len(self.files)

    def __len__(self):
        return int(len(self.files)/3)

    def __getitem__(self, idx):
        extracts = []
        for file in self.files[idx*3:(idx+1)*3]:
            img = cv2.imread(f'{self.folder}/{file}')
            extract = detect(img)
            if extract != None:
                extracts.append(extract)
        return extracts


class DifferentFaceDataset(torch.utils.data.Dataset):
    def __init__(self,  not_idx, folder='/home/dung/AI/archive/training'):
        self.folder = folder
        self.files = []
        for i, file in enumerate(os.listdir(folder)):
            j = not_idx*3
            if i not in [j, j+1, j+2]:
                self.files.append(file)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(f'{self.folder}/{self.files[idx]}')
        extract = detect(img)
        if extract != None:
            return extract


device = torch.device('cuda')
model = VggFace()
model.to(device)
i1 = cv2.imread('2.jpg')
i2 = cv2.imread('4.jpg')
i1 = detect(i1).to(device)
i2 = detect(i2).to(device)
i1 = torch.unsqueeze(i1, 0)
i2 = torch.unsqueeze(i2, 0)
model.train()
loss = model(i1, i2, torch.tensor(0, dtype=torch.float32))

similarityDataset = SimilarityFaceDataset()
similarity_data_loader = torch.utils.data.DataLoader(
    similarityDataset, batch_size=1, shuffle=False, num_workers=0)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
epochs = 1000
similarity_batch = 1
different_batch = 50
for epoch in range(epochs):
    print(f'epoch {epoch} running')
    for i, extracts in enumerate(similarity_data_loader):
        loss = None
        for extract in extracts[1:3]:
            loss = model(extracts[0].to(device), extract.to(
                device), torch.tensor(0, dtype=torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # toi uu voi anh khac
        differentDataset = DifferentFaceDataset(i)
        different_data_loader = torch.utils.data.DataLoader(
            differentDataset, batch_size=different_batch, shuffle=False, num_workers=0)
        for difference in different_data_loader:
            loss = model(extracts[0].repeat(different_batch, 3, 224, 224).to(device), difference.to(
                device), torch.tensor(1, dtype=torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'mean loss {i} = {loss}')
    torch.save(model.state_dict(), 'recognize_1.pth')
