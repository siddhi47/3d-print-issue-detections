import torch
import torchvision
import pandas as pd
import torchvision.transforms as transforms
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image
import cv2

class CDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None,):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv2.imread(img_path,)
        label = self.img_labels.iloc[idx, -1]
        if self.transform:
            image = self.transform(image)
        return image, label


IMAGE_SIZE = (256,256)
#IMAGE_SIZE = (32,32)

transform = transforms.Compose(

    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize(IMAGE_SIZE)
    ]
)

from torch.utils.data.sampler import SubsetRandomSampler

BATCH_SIZE = 32
validation_split=.2
shuffle_dataset = True
random_seed=12


dataset = CDataset(
    '/mnt/d/internship/early-detection-of-3d-printing-issues/train.csv',
    '/mnt/d/internship/early-detection-of-3d-printing-issues/images',
    transform
)

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                                sampler=valid_sampler)




import torch.nn as nn
import torch.nn.functional as F


torch.cuda.empty_cache()

NUM_CLASSES=2
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


if os.path.exists(os.path.join('models', 'vgg16.pth')):
    print('Saved model found, loading...')
    net = Net_VGG(NUM_CLASSES)
    net.load_state_dict(torch.load(os.path.join('models', 'vgg16.pth')))
    print('Model loaded')
else:
    print('No saved model found, training...')
    net = Net_VGG(NUM_CLASSES)
    print('Model created')

net.to(device)



import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(10):  

    running_loss = 0.0
    accuracy=[]
    for i, data in enumerate(train_loader, 0):
        net.train()
        net.to(device)
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        
        outputs = net(inputs)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        accuracy.append(acc)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(
            "\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t accuracy: {:.6f}".format(
            epoch,
            i * len(data),
            len(train_loader),
            100.0 * i / len(train_loader),
            loss.item(),
            acc.item(),
            end="",
        ))
    #save model
    os.makedirs("models", exist_ok=True)
    torch.save(net.state_dict(), os.path.join('models',f'vgg16.pth'))

    
    if epoch%1 == 0:
        print("validation")
        net.eval()
        net.to(device)
        val_loss = 0
        correct = 0
        with torch.no_grad():
            val_acc = []
            for i, data in enumerate(validation_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                output = net(inputs)
                val_loss += criterion(output, labels).item()
                acc = (output.argmax(dim=1) == labels).float().mean()
                val_acc.append(acc.item())
            val_loss /= len(validation_loader)
            print(
                "Val set: Average loss: {:.4f}, Accuracy: {}\n".format(
                    val_loss,
                    np.mean(val_acc),
                )
            )
    

print('Finished Training')


