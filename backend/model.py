from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchsummary import summary

class FRCNNObjectDetector(FasterRCNN):
    def __init__(self, num_classes=91, **kwargs):
        backbone = resnet_fpn_backbone('resnet50', True)
        super(FRCNNObjectDetector, self).__init__(backbone, num_classes, **kwargs)

if __name__  == '__main__':
    model = FRCNNObjectDetector()
    print(summary(model, (3, 224, 224)))