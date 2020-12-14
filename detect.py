import cv2
import torch
import torchvision
import os
import math
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
import numpy as np
from torchvision import transforms
import torch.nn as nn


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
            return 'Error'
        else:
            x0, y0, x1, y1 = boxes[0]
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            return cv2.resize(im[y0:y1, x0:x1, :], (224, 224))/255
