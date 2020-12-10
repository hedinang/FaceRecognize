import cv2
import torch
import torchvision
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import numpy as np
import pandas as pd
import io
import sqlite3

orig_frame = cv2.imread('/home/dung/AI/facial/input/test/Aidan_Quinn_20.jpg')[ 70:210, 70:161 ,:]
# orig_frame = cv2.imread('2.jpg')
frame_height, frame_width, _ = orig_frame.shape
image = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
# cv2.imshow('aa',image)
# cv2.waitKey(0)
cus = image
image = image / 255.0
image = torch.tensor(image, dtype=torch.float32)
image = torch.unsqueeze(image, 0)
image = image.permute(0, 3, 1, 2)

model = torchvision.models.resnet50()
model.fc = nn.Linear(2048, 136)
model.load_state_dict(torch.load('b.pth'))
result = model(image)
x = result.detach().numpy()
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
# sqlite3.register_adapter(np.ndarray, adapt_array)

# # Converts TEXT to np.array when selecting
# sqlite3.register_converter("array", convert_array)
# con = sqlite3.connect('example.db', detect_types=sqlite3.PARSE_DECLTYPES)
# cur = con.cursor()
# cur.execute("create table test (arr array)")
# cur.execute("insert into test (arr) values (?)", (x, ))
# con.commit()
# cur.execute("select arr from test")
# data = cur.fetchone()[0]
# con.close()
print('aaa')


result = result.reshape(-1, 2)
keypoints = result
for p in range(keypoints.shape[0]):
    cv2.circle(cus, (int(keypoints[p, 0]), int(keypoints[p, 1])),
               1, (0, 0, 255), -1, cv2.LINE_AA)
orig_frame = cv2.resize(cus, (frame_width, frame_height))
cv2.imshow('Facial Keypoint Frame', cus)
cv2.waitKey(0)
print('aaa')
