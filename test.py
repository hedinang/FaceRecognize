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

class Detect:
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
        self.device = torch.device('cpu')
        self.model.load_state_dict(torch.load(
            '3.pth'))
        self.model.to(self.device)
        self.model.eval()

    def forward(self, img):
        fontScale = 1
        color = (255, 0, 0)
        thickness = 1
        im = img
        img = torch.tensor(img, dtype=torch.float32)/255
        img = img.permute((2, 0, 1))
        output = self.model([img.to(self.device)])
        boxes = output[0]['boxes'].detach().numpy()
        labels = output[0]['labels'].detach().numpy()
        scores = output[0]['scores'].detach().numpy()
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            print('{} {} {} {}'.format(x0,y0,x1,y1))
            cv2.rectangle(im, (x0, y0), (x1, y1), (235, 20, 20), 1)
            cv2.putText(im, str(i), (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                        color, thickness, cv2.LINE_AA)
        cv2.imshow('aaa', im)
        cv2.waitKey(0)
        return result


img = cv2.imread('5.jpg')
detect = Detect()
ds = detect.forward(img)
