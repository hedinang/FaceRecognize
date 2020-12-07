import io
import sqlite3
import cv2
import torch
import torchvision
import os
import math
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
import numpy as np
from PIL import Image

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
model = FasterRCNN(backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler,
                   box_score_thresh=0.95)
print(model)
device = torch.device('cpu')
model.load_state_dict(torch.load(
    '/home/dung/Project/AI/3.pth'))
model.to(device)
model.eval()
extract = model.backbone
# 483 669 591 827
img = cv2.imread('2.jpg')[148:471, 323:555, :]
img = cv2.resize(img, (224, 224))/255
transform = torchvision.transforms.Compose([
    # torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        0, [1, 1, 1])])
img = torch.tensor(img, dtype=torch.float32)
img = torch.unsqueeze(img, 0)
img = img.permute(0, 3, 1, 2)
output = extract(img.to(device))
x = output.detach().numpy()


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)
con = sqlite3.connect('example.db', detect_types=sqlite3.PARSE_DECLTYPES)
cur = con.cursor()
cur.execute("create table test (arr array)")
cur.execute("insert into test (arr) values (?)", (x, ))
con.commit()
cur.execute("select arr from test")
data = cur.fetchone()[0]
con.close()
print('aaa')
