import torch
import torch.nn.functional as F


class Net_VGG(torch.nn.Module):

    """
    This class is uses VGG netowrk as pretrained network
    """

    def __init__(self, num_classes=5):
        """
        This function initializes the model
        """
        super(Net_VGG, self).__init__()
        self.model = torch.hub.load("pytorch/vision:v0.6.0", "vgg16", pretrained=True)
        self.model.classifier[6] = torch.nn.Linear(4096, num_classes)
        self.backbone = self.model

    def forward(self, x):
        """
        This function defines the forward pass of the model
        """
        x = self.backbone(x)
        x = F.normalize(x)
        x = F.log_softmax(x, dim=1)
        return x
