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
import torch.nn as nn
from vidgear.gears import WriteGear
device = torch.device('cuda')


class Vgg_face_dag(nn.Module):

    def __init__(self):
        super(Vgg_face_dag, self).__init__()

        # self.meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688],
        #              'std': [1, 1, 1],
        #              'imageSize': [224, 224, 3]}
        # self.conv1_1 = nn.Conv2d(
        #     3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)).to(device)
        # self.relu1_1 = nn.ReLU(inplace=True).to(device)
        # self.conv1_2 = nn.Conv2d(64, 64, kernel_size=[
        #                          3, 3], stride=(1, 1), padding=(1, 1)).to(device)
        # self.relu1_2 = nn.ReLU(inplace=True).to(device)
        # self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[
        #                           2, 2], padding=0, dilation=1, ceil_mode=False).to(device)
        # self.conv2_1 = nn.Conv2d(64, 128, kernel_size=[
        #                          3, 3], stride=(1, 1), padding=(1, 1)).to(device)
        # self.relu2_1 = nn.ReLU(inplace=True).to(device)
        # self.conv2_2 = nn.Conv2d(128, 128, kernel_size=[
        #                          3, 3], stride=(1, 1), padding=(1, 1)).to(device)
        # self.relu2_2 = nn.ReLU(inplace=True).to(device)
        # self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[
        #                           2, 2], padding=0, dilation=1, ceil_mode=False).to(device)
        # self.conv3_1 = nn.Conv2d(128, 256, kernel_size=[
        #                          3, 3], stride=(1, 1), padding=(1, 1)).to(device)
        # self.relu3_1 = nn.ReLU(inplace=True).to(device)
        # self.conv3_2 = nn.Conv2d(256, 256, kernel_size=[
        #                          3, 3], stride=(1, 1), padding=(1, 1)).to(device)
        # self.relu3_2 = nn.ReLU(inplace=True).to(device)
        # self.conv3_3 = nn.Conv2d(256, 256, kernel_size=[
        #                          3, 3], stride=(1, 1), padding=(1, 1)).to(device)
        # self.relu3_3 = nn.ReLU(inplace=True).to(device)
        # self.pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=[
        #                           2, 2], padding=0, dilation=1, ceil_mode=False).to(device)
        # self.conv4_1 = nn.Conv2d(256, 512, kernel_size=[
        #                          3, 3], stride=(1, 1), padding=(1, 1)).to(device)
        # self.relu4_1 = nn.ReLU(inplace=True).to(device)
        # self.conv4_2 = nn.Conv2d(512, 512, kernel_size=[
        #                          3, 3], stride=(1, 1), padding=(1, 1)).to(device)
        # self.relu4_2 = nn.ReLU(inplace=True).to(device)
        # self.conv4_3 = nn.Conv2d(512, 512, kernel_size=[
        #                          3, 3], stride=(1, 1), padding=(1, 1)).to(device)
        # self.relu4_3 = nn.ReLU(inplace=True).to(device)
        # self.pool4 = nn.MaxPool2d(kernel_size=[2, 2], stride=[
        #                           2, 2], padding=0, dilation=1, ceil_mode=False).to(device)
        # self.conv5_1 = nn.Conv2d(512, 512, kernel_size=[
        #                          3, 3], stride=(1, 1), padding=(1, 1)).to(device)
        # self.relu5_1 = nn.ReLU(inplace=True).to(device)
        # self.conv5_2 = nn.Conv2d(512, 512, kernel_size=[
        #                          3, 3], stride=(1, 1), padding=(1, 1)).to(device)
        # self.relu5_2 = nn.ReLU(inplace=True).to(device)
        # self.conv5_3 = nn.Conv2d(512, 512, kernel_size=[
        #                          3, 3], stride=(1, 1), padding=(1, 1)).to(device)
        # self.relu5_3 = nn.ReLU(inplace=True).to(device)
        # self.pool5 = nn.MaxPool2d(kernel_size=[2, 2], stride=[
        #                           2, 2], padding=0, dilation=1, ceil_mode=False).to(device)
        # self.fc6 = nn.Linear(
        #     in_features=25088, out_features=4096, bias=True).to(device)
        # self.relu6 = nn.ReLU(inplace=True).to(device)
        # self.dropout6 = nn.Dropout(p=0.5).to(device)
        # self.fc7 = nn.Linear(
        #     in_features=4096, out_features=4096, bias=True).to(device)
        # self.relu7 = nn.ReLU(inplace=True).to(device)
        # self.dropout7 = nn.Dropout(p=0.5).to(device)
        # self.fc8 = nn.Linear(
        #     in_features=4096, out_features=2622, bias=True).to(device)

        self.meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.conv1_1 = nn.Conv2d(
            3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=[
                                 3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[
                                  2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=[
                                 3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=[
                                 3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[
                                  2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=[
                                 3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=[
                                 3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=[
                                 3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=[
                                  2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=[
                                 3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=[
                                 3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=[
                                 3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=[2, 2], stride=[
                                  2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=[
                                 3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=[
                                 3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=[
                                 3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=[2, 2], stride=[
                                  2, 2], padding=0, dilation=1, ceil_mode=False)
        self.fc6 = nn.Linear(
            in_features=25088, out_features=4096, bias=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(
            in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(
            in_features=4096, out_features=2622, bias=True)

    def forward(self, x0):
        x1 = self.conv1_1(x0)
        x2 = self.relu1_1(x1)
        x3 = self.conv1_2(x2)
        x4 = self.relu1_2(x3)
        x5 = self.pool1(x4)
        x6 = self.conv2_1(x5)
        x7 = self.relu2_1(x6)
        x8 = self.conv2_2(x7)
        x9 = self.relu2_2(x8)
        x10 = self.pool2(x9)
        x11 = self.conv3_1(x10)
        x12 = self.relu3_1(x11)
        x13 = self.conv3_2(x12)
        x14 = self.relu3_2(x13)
        x15 = self.conv3_3(x14)
        x16 = self.relu3_3(x15)
        x17 = self.pool3(x16)
        x18 = self.conv4_1(x17)
        x19 = self.relu4_1(x18)
        x20 = self.conv4_2(x19)
        x21 = self.relu4_2(x20)
        x22 = self.conv4_3(x21)
        x23 = self.relu4_3(x22)
        x24 = self.pool4(x23)
        x25 = self.conv5_1(x24)
        x26 = self.relu5_1(x25)
        x27 = self.conv5_2(x26)
        x28 = self.relu5_2(x27)
        x29 = self.conv5_3(x28)
        x30 = self.relu5_3(x29)
        x31_preflatten = self.pool5(x30)
        # x31 = x31_preflatten.view(x31_preflatten.size(0), -1)
        x31 = x31_preflatten.reshape(-1)
        x31 = torch.unsqueeze(x31, 0)

        x32 = self.fc6(x31)
        x33 = self.relu6(x32)
        x34 = self.dropout6(x33)
        x35 = self.fc7(x34)
        x36 = self.relu7(x35)
        x37 = self.dropout7(x36)
        x38 = self.fc8(x37)
        return x38

    def forward_two(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        #euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        difference = output1 - output2
        return difference


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
        self.model.load_state_dict(torch.load('/home/dung/Project/AI/3.pth'))
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
sqlite3.register_converter("array", convert_array)
con = sqlite3.connect('example.db', detect_types=sqlite3.PARSE_DECLTYPES)
cur = con.cursor()
cur.execute("select * from test")
datas = cur.fetchall()

detect = Detect()
extract = Vgg_face_dag()
state_dict = torch.load('/home/dung/Project/AI/a.pth')
extract.load_state_dict(state_dict)
extract.to(device)
color = (255, 0, 0)
thickness = 1
fontScale = 1
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"
url = 'rtsp://192.168.1.155:5554'
cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_FPS, 15)
output_params = {"-vcodec": "libx264", "-crf": 0, "-preset": "fast"}
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))

# size = (frame_width, frame_height)
# result = cv2.VideoWriter('filename.avi',
#                          cv2.VideoWriter_fourcc(*'MJPG'),
#                          10, size)

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
            name = None
            box = []
            for data in datas:
                # dis=  np.linalg.norm([data]- output.detach().cpu().numpy(), axis=1)
                feature = torch.tensor(
                    data[0], dtype=torch.float32).to(device)
                similarity = nn.CosineSimilarity(
                    dim=1, eps=1e-6)(feature, output.to(device))
                # dis = torch.nn.MSELoss()(output, data)
                if similarity > 0.75 and similarity > dis_tmp:
                    dis_tmp = similarity
                    name = data[1]
            if name != None:
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 1)
                cv2.putText(frame, name, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                            color, thickness, cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 1)
                cv2.putText(frame, 'undefied', (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                            color, thickness, cv2.LINE_AA)
        writer.write(frame)
        cv2.imshow('frame', frame)

    q = cv2.waitKey(1)
    if q == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()
writer.close()
