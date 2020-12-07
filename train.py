import cv2
import torch
import torchvision
import os
import math
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN

device = torch.device('cuda')

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
                   box_roi_pool=roi_pooler)
model.load_state_dict(torch.load('3.pth'))
model.to(device)
# model.eval()

train_num = 1000
step = 29382


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, file_dir='/home/dung/AI/WIDER_train/wider_face_train_annot.txt', root_url='/home/dung/AI/WIDER_train/images', transform=None):
        self.root_url = root_url
        self.lines = open(file_dir).readlines()
        self.index = []
        self.last = None
        for i, line in enumerate(self.lines[step:]):
            if len(self.index) == train_num:
                self.last = i+step
                print('i = {}'.format(i))
                break
            if '.jpg' in line:
                self.index.append(i)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        img = None
        d = {}
        boxes = []
        labels = []
        array = []
        if idx < len(self.index) - 1:
            array = self.lines[step+self.index[idx]: step+self.index[idx+1]]
        else:
            array = self.lines[step+self.index[idx]: self.last]
        for i, line in enumerate(array):
            if i == 0:
                img = cv2.imread(
                    '{}/{}'.format(self.root_url, line.rstrip('\n')))
                img = torch.tensor(img, dtype=torch.float32) / 255
                img = img.permute(2, 0, 1)
            if i == 1:
                continue
            if i > 1:
                x, y, w, h = line.strip().rstrip('\n').split(' ')
                x, y, w, h = float(x), float(y), float(w), float(h)
                if w == 0 or h == 0:
                    continue
                x0, y0, x1, y1 = math.floor(x), math.floor(
                    y), math.ceil(x+w), math.ceil(y+h)
                boxes.append([x0, y0, x1, y1])
                labels.append(1)
                d['boxes'] = torch.tensor(boxes, dtype=torch.int64)
                d['labels'] = torch.tensor(labels, dtype=torch.int64)
        return img, d


dataset = FaceDataset()
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=0)
# criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)


for i in range(1000):
    print('Epoch {}\n'.format(i))
    for j, (images, targets) in enumerate(data_loader):
        if len(targets) == 0:
            continue
        images = images.to(device)
        a = {}
        a['boxes'] = targets['boxes'][0].to(device)
        a['labels'] = targets['labels'][0].to(device)
        output = model(images, [a])
        losses = sum(loss for loss in output.values())
        if j % 300 == 0:
            print('Step {} -- loss_classifier = {} -- loss_box_reg = {} -- loss_objectness = {} -- loss_rpn_box_reg = {}\n'.format(j,
                                                                                                                                   output['loss_classifier'].item(), output['loss_box_reg'].item(), output['loss_objectness'].item(), output['loss_rpn_box_reg'].item()))
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    if i % 1 == 0:
        print('save model')
        torch.save(model.state_dict(), '3.pth')
print('done')
