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
from face_net import InceptionResnetV1
device = torch.device('cuda')
model = InceptionResnetV1()
model.load_state_dict(torch.load('/home/dung/AI/face_net.pth'))
model.eval()
model.to(device)

detect = Detect()


class Person:
    def __init__(self):
        self.db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="password",
            database="face"
        )
        self.cursor = self.db.cursor()
    def create(self, name):
        self.cursor.execute(
            'insert into person (name, active, del_flag) values (%s,%s,%s)', (name, 0, 0))
        self.db.commit()
        self.cursor.execute(
            f'select * from person where id = {self.cursor.lastrowid}')
        result = self.cursor.fetchone()
        self.cursor.close()
        self.db.close()
        response = {}
        response['id'] = result[0]
        response['name'] = result[1]
        response['img'] = result[2]
        response['feature'] = result[3]
        response['active'] = result[4]
        response['del_flag'] = result[5]
        return response

    def extract(self, file_dir):
        img = cv2.imread(file_dir)
        face = detect.forward(img)
        face = torch.tensor(face, dtype=torch.float32)
        face = torch.unsqueeze(face, 0)
        face = face.permute(0, 3, 1, 2).to(device)
        output = model(face)
        return output.detach().cpu().numpy().tostring()

    def update(self, id, name_replace=None, im=None):
        self.cursor.execute(
            f'select * from person where id = {id} and del_flag = 0')
        record = self.cursor.fetchone()
        timestamp = datetime.now().timestamp()
        name = record[1]
        img = record[2]
        feature = record[3]
        active = record[4]
        if name_replace != None:
            name = name_replace
        if im != None:
            img = f'image/{id}_{timestamp}.jpg'
            im.save(img)
            feature = self.extract(img)
            active = 1
        sql = f'update person set name = %s, img = %s, feature = %s, active = %s where id = %s'
        self.cursor.execute(sql, (name, img, feature, active, id))
        self.db.commit()
        self.cursor.close()
        self.db.close()


class Recognize:
    def __init__(self):
        pass

    def process(self):
        pass
