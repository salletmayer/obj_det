import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional as F
from PIL import Image

# Define your custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, targets):
        self.image_paths = image_paths
        self.targets = targets

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        target = self.targets[idx]

        return F.to_tensor(image), target

# Define your custom DataLoader
batch_size = 4
custom_dataset = CustomDataset(image_paths, targets)
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
