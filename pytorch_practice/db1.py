
# CIFAR10

import torch
import torchvision
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Preparing for data set
transf = tr.Compose([tr.Resize(8), tr.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transf)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transf)


#print(trainset[0][0].size())

trainld = DataLoader(trainset, batch_size=50, shuffle=True, num_workers=0)
testld = DataLoader(testset, batch_size=50, shuffle=True, num_workers=0)


dataiter = iter(trainld)
images, labels = dataiter.next()
print(images.size())


