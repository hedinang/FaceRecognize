import cv2
import torch
import torchvision
import os
import io
import math
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
import numpy as np
import sqlite3
# import face_recognition
# results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
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
        self.device = torch.device('cuda')
        self.model.load_state_dict(torch.load(
            '/home/dung/Project/AI/3.pth'))
        self.model.to(self.device)
        self.model.eval()
        self.extract = self.model.backbone

    def forward(self, img):
        fontScale = 1
        color = (255, 0, 0)
        thickness = 1
        im = img
        img = torch.tensor(img, dtype=torch.float32)/255
        img = img.permute((2, 0, 1))
        output = self.model([img.to(self.device)])
        boxes = output[0]['boxes'].detach().cpu().numpy()
        labels = output[0]['labels'].detach().cpu().numpy()
        scores = output[0]['scores'].detach().cpu().numpy()
        result = []
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            result.append((x0, y0, x1, y1))

        return result


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)
con = sqlite3.connect('example.db', detect_types=sqlite3.PARSE_DECLTYPES)
cur = con.cursor()
cur.execute("select arr from test")
datas = cur.fetchone()

img = cv2.imread('4.jpg')
detect = Detect()
extract = detect.extract

ds = detect.forward(img)
color = (255, 0, 0)
thickness = 1
fontScale = 1
for box in ds:
    x0, y0, x1, y1 = box
    im = cv2.resize(img[y0:y1, x0:x1, ], (224, 224))/255
    im = torch.tensor(im, dtype=torch.float32)
    im = torch.unsqueeze(im, 0)
    im = im.permute(0, 3, 1, 2)
    output = extract(im.to(torch.device('cuda')))
    dis_tmp = 10
    name = None
    box = []
    for data in datas:
        dis = torch.nn.MSELoss()(output, torch.tensor(
            data, dtype=torch.float32).to(torch.device('cuda')))
        if dis < 0.2 and dis < dis_tmp:
            name = 'dung'
    if name != None:
        cv2.rectangle(img, (x0, y0), (x1, y1), (100, 200, 120), 1)
        cv2.putText(img, name, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                    color, thickness, cv2.LINE_AA)
cv2.imshow('aa', img)
cv2.waitKey(0)
