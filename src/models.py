import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.architectures import *

models = {
    "resnet": Net_ResNet,
    "dense": Net_DenseNet,
    "inception": Net_Inception,
    "vgg": Net_VGG,
    "mobilenet": Net_MobileNet,
    "vit": Net_ViT
}
