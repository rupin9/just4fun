
# Using folder name
import torch
import torchvision
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
import numpy as np

# ./class/tiger and ./class/lion
transf = tr.Compose([tr.Resize(16), tr.ToTensor()])
trainset = torchvision.datasets.ImageFolder(root='./data/class', transform=transf)
trainloader = DataLoader(trainset, batch_size = 2, shuffle=False, num_workers=2)
print(len(trainloader))
trainset[0][0].size()

