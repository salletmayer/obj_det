import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional as F
from PIL import Image

from sklearn.model_selection import train_test_split
from backend.data_mapper import get_set
import os
import json
import random

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def images_from_paths(image_paths):
    images = []

    for path in image_paths:
        image = Image.open('./data/' + path).convert("RGB")
        image = image.resize(size=(640, 640))

        images.append(F.to_tensor(image))
    
    return images

def generate_batches(image_paths, targets, batch_size):
    num_samples = len(image_paths)
    num_batches = num_samples // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        
        batch_image_paths = image_paths[start_idx:end_idx]
        batch_targets = targets[start_idx:end_idx]
        
        yield batch_image_paths, batch_targets

    # Handle the last batch if it's not a full batch
    if num_samples % batch_size != 0:
        start_idx = num_batches * batch_size
        batch_image_paths = image_paths[start_idx:]
        batch_targets = targets[start_idx:]
        
        yield batch_image_paths, batch_targets

# get data
batch_size = 2

abs_path_to_set = os.path.abspath("./data/labels.json")

image_paths, targets = get_set(abs_path_to_set)

data = list(zip(image_paths, targets))
random.shuffle(data)
image_paths, targets = zip(*data)

train_image_paths, test_image_paths, train_targets, test_targets = train_test_split(image_paths, targets, test_size=0.1, random_state=42)



## load model
import torch.optim as optim
from backend.model import FRCNNObjectDetector

with open('data/info.json', 'r') as file:
        info = json.load(file)

model = FRCNNObjectDetector(num_classes=len(info['classes']))

optimizer = optim.Adam(model.parameters(), lr=0.001)

## train model
from vision.util.engine import train_one_epoch, evaluate 
import vision.util.utils
import math
import sys

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

num_epochs = 1

for epoch in range(num_epochs):

    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(train_image_paths) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for image_paths, targets in generate_batches(train_image_paths, train_targets, 2):
        images = images_from_paths(image_paths)
        
        # train model
        model.train()

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=False):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = vision.util.utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        model.eval()

        test_images = images_from_paths(test_image_paths)
        pred = model(test_images)

        print(pred)

print('Done')    