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
        self.model.load_state_dict(torch.load('3.pth'))
        self.model.to(self.device)
        self.model.eval()
        self.extract = self.model.backbone

    def forward(self, frame):
        fontScale = 1
        color = (255, 0, 0)
        thickness = 1
        im = frame
        frame = torch.tensor(frame, dtype=torch.float32)/255
        frame = frame.permute((2, 0, 1))
        output = self.model([frame.to(self.device)])
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

detect = Detect()
extract = detect.extract

color = (255, 0, 0)
thickness = 1
fontScale = 1
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"
url = 'rtsp://192.168.1.122:5554'
cap = cv2.VideoCapture(url)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
result = cv2.VideoWriter('filename.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
while(True):
    ret, frame = cap.read()
    if frame is not None:
        ds = detect.forward(frame)

        for box in ds:
            x0, y0, x1, y1 = box
            # cv2.rectangle(frame, (x0, y0), (x1, y1), (100, 200, 120), 1)
            # cv2.imshow('aa',frame)
            # cv2.waitKey(0)
            im = cv2.resize(frame[y0:y1, x0:x1, ], (224, 224))/255
            im = torch.tensor(im, dtype=torch.float32)
            im = torch.unsqueeze(im, 0)
            im = im.permute(0, 3, 1, 2)
            output = extract(im.to(torch.device('cuda')))
            dis_tmp = 10
            name = None
            box = []
            for data in datas:
                dis=  np.linalg.norm(output.detach().cpu().numpy() - data, axis=1)
                data = torch.tensor(
                    data, dtype=torch.float32).to(torch.device('cuda'))
                dis = torch.nn.MSELoss()(output, data)
                if dis < 0.4 and dis < dis_tmp:
                    name = 'dung'
            if name != None:
                cv2.rectangle(frame, (x0, y0), (x1, y1), (100, 200, 120), 1)
                cv2.putText(frame, name, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                            color, thickness, cv2.LINE_AA)
        result.write(frame)
        cv2.imshow('frame', frame)

    q = cv2.waitKey(1)
    if q == ord("q"):
        break
cv2.destroyAllWindows()
