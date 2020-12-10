import cv2
import torch
import torchvision
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import numpy as np
import pandas as pd

a = pd.read_csv('/home/dung/AI/facial/input/training_frames_keypoints.csv')


class FaceKeypointDataset(Dataset):
    def __init__(self, root_dir='/home/dung/AI/facial/input/training',
                 path_file='/home/dung/AI/facial/input/training_frames_keypoints.csv'):
        self.root_dir = root_dir
        self.file = pd.read_csv(path_file)
        self.resize = 224

    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        path = self.file.iloc[index][0]
        image = cv2.imread(f'{self.root_dir}/{path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w, channel = image.shape
        # resize the image into `resize` defined above
        image = cv2.resize(image, (self.resize, self.resize))
        # again reshape to add grayscale channel format
        image = image / 255.0
        # transpose for getting the channel size to index 0
        image = np.transpose(image, (2, 0, 1))
        # get the keypoints
        keypoints = self.file.iloc[index][1:]
        keypoints = np.array(keypoints, dtype='float32')
        # reshape the keypoints
        keypoints = keypoints.reshape(-1, 2)
        # rescale keypoints according to image resize
        keypoints = keypoints * [self.resize / orig_w, self.resize / orig_h]
        return {
            'images': torch.tensor(image, dtype=torch.float),
            'keypoints': torch.tensor(keypoints, dtype=torch.float),
        }


device = torch.device('cuda')
model = torchvision.models.resnet50()
model.fc = nn.Linear(2048, 136)
model.load_state_dict(torch.load('a.pth'))
model.to(device)
epochs = 1000
dataset = FaceKeypointDataset()
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=100, shuffle=False, num_workers=0)
# criterion = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.SmoothL1Loss()
for epoch in range(epochs):
    print(f'Epoch {epoch}\n')
    for i, data in enumerate(data_loader):
        images = data['images'].to(device)
        keypoints = data['keypoints']
        keypoints = keypoints.view(keypoints.size(0), -1).to(device)

        outputs = model(images)
        loss = criterion(outputs, keypoints)
        if i % 10 == 0:
            print(f'loss = {loss}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 1 == 0:
        print('save model')
        torch.save(model.state_dict(), 'b.pth')
