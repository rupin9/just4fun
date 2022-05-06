
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

first_pair = ds[0]
img_data, img_label = first_pair
print(type(img_data), type(img_label))
x = img_data.permute(1,2,0) # to HWC
print(f"x.shape = {x.shape}")
print(f"img_label.shape = {img_label.shape}")
figure = plt.figure()
plt.imshow(x, cmap="gray")
plt.show()


