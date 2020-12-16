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
import mysql.connector
import torch.nn as nn
from vidgear.gears import WriteGear
from torch.nn import PairwiseDistance
from face_net import InceptionResnetV1
device = torch.device('cuda')




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


def select():
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="face"
    )
    cursor = db.cursor()
    cursor.execute(
        "select feature, name from person where del_flag = 0 and active = 1")
    rows = cursor.fetchall()
    cursor.close()
    db.close()
    return rows


detect = Detect()

extract = InceptionResnetV1()
extract.load_state_dict(torch.load('/home/dung/AI/face_net.pth'))
extract.eval()
extract.to(device)
cosin = nn.CosineSimilarity()
color = (255, 255, 255)
thickness = 1
fontScale = 1
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"
url = 'rtsp://192.168.1.122:5554'
cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_FPS, 15)
output_params = {"-vcodec": "libx264", "-crf": 0, "-preset": "fast"}
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
result = cv2.VideoWriter('filename.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

writer = WriteGear(output_filename='test/output.m3u8', compression_mode=True, logging=True,
                   **output_params)  # Define writer with output filename 'Output.mp4'
while(True):
    ret, frame = cap.read()
    if frame is not None:
        ds = detect.forward(frame)

        for box in ds:
            x0, y0, x1, y1 = box
            im = cv2.resize(frame[y0:y1, x0:x1, ], (224, 224))/255
            im = torch.tensor(im, dtype=torch.float32)
            im = torch.unsqueeze(im, 0)
            im = im.permute(0, 3, 1, 2)

            output = extract(im.to(device))
            dis_tmp = 0
            name = 'undefined'
            box = []
            for data in select():
                feature = np.frombuffer(data[0], dtype=np.float32)
                feature = torch.tensor(feature, dtype=torch.float32)
                feature = torch.unsqueeze(feature, 0).to(device)
                similarity = cosin(feature, output.to(device))
                if similarity > 0.8 and similarity > dis_tmp:
                    dis_tmp = similarity
                    name = data[1]

            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 1)
            cv2.putText(frame, name, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                        color, thickness, cv2.LINE_AA)

        # writer.write(frame)
        result.write(frame)
        cv2.imshow('frame', frame)

    q = cv2.waitKey(1)
    if q == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()
writer.close()
