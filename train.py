import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional as F
from PIL import Image

from backend.data_mapper import get_set
import os
import json

# Define your custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, targets):
        self.image_paths = image_paths
        self.targets = targets

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open('./data/' + image_path).convert("RGB")
        target = self.targets[idx]

        image = image.resize(size=(640, 640))

        return F.to_tensor(image), target

# get data
batch_size = 1

abs_path_to_set = os.path.abspath("./data/labels.json")

image_paths, targets = get_set(abs_path_to_set)

custom_dataset = CustomDataset(image_paths, targets)
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)


## training
import torch.optim as optim
from backend.model import FRCNNObjectDetector

with open('data/info.json', 'r') as file:
        info = json.load(file)

model = FRCNNObjectDetector(num_classes=len(info['classes']))

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

import numpy

images = []
targets = []
for image, target in data_loader:
    target['boxes'] = target['boxes'][0]
    image = image[0]

    images.append(image)
    targets.append(target)

outputs = model(images[:3], targets[:3])
print(outputs)