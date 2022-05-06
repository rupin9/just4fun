
# Using Custom dataset with preprocessing (not using tr.transform)

import torch
import torchvision
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

#import preprocessing
train_images = np.random.randint(256, size=(20,32,32,3))
train_labels = np.random.randint(2, size=(20,1))

#preprocessing ....
#train_mages, train_labels - preprocessing(train_images, train_labels)

# without transform
class TensorData(Dataset):
    
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.x_data = self.x_data.permute(0,3,1,2) # to NCHW
        self.y_data = torch.tensor(x_data, dtype=torch.int32)
        self.len = self.y_data.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

train_data = TensorData(train_images, train_labels)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
train_data[0][0].size()

dataiter = iter(train_loader)
images, labels = dataiter.next()
print(f"images = {images.size()}")


#############################################33
# Custom dataset class with transform
class MyDataset(Dataset):
    
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform
        self.len = len(y_data)
    
    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def __len__(self):
        return self.len
    
class MyToTensor:
    def __call__(self, sample):
        inputs, lables = sample
        inputs = torch.tensor(inputs, dtype=torch.float32)
        lables = torch.tensor(lables, dtype=torch.int32)
        inputs = inputs.permute(2,0,1) # HWC -> CHW
        return inputs, labels
    
class LinearTensor:
    def __init__(self, slope=1, bias=0):
        self.slope = slope
        self.bias = bias
    
    def __call__(self, sample):
        inputs, labels = sample
        inputs = self.slope * inputs + self.bias
        return inputs, labels

trans = tr.Compose([MyToTensor(), LinearTensor(2,5)])
ds1 = MyDataset(train_images, train_labels, transform=trans)
train_loader1 = DataLoader(ds1, batch_size = 10, shuffle = True)

first_data = ds1[0]
features, labels = first_data
print(type(features), type(labels))

dataiter1 = iter(train_loader1)
images1, labels1 = dataiter1.next()

print(f"images1 = {images1.size()}")
print(f"images1[0] = {images1[0].size()}")

x = (images1[0]%256)/256
y = x.permute(1,2,0) # to HWC
print(f"y.shape = {y.shape}")

figure = plt.figure()
plt.imshow(y)
plt.show()

#############################################33
# Custom dataset class with transform
# MyDataset + use tr.transform

train_images = np.random.randint(256, size=(20,32,32,3))
train_labels = np.random.randint(2, size=(20,1))

class MyDatasetTr(Dataset):
    
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform
        self.len = len(y_data)
    
    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def __len__(self):
        return self.len 
class MyTransform:
    def __call__(self, sample):
        inputs, lables = sample
        inputs = torch.tensor(inputs, dtype=torch.float32)
        lables = torch.tensor(lables, dtype=torch.int32)
        inputs = inputs.permute(2,0,1) # HWC -> CHW
        
        transf = tr.Compose([tr.ToPILImage(), tr.Resize(128), tr.ToTensor(), tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        final_output = transf(inputs)
        return final_output, labels

ds2 = MyDatasetTr(train_images, train_labels, transform=MyTransform())
train_loader2 = DataLoader(ds2, batch_size = 10, shuffle = True)
first_data = ds2[0]
features, labels = first_data
print(type(features), type(labels))

dataiter2 = iter(train_loader2)
images2, labels2 = dataiter2.next()
print(f"images2.size() = {images2.size()}")



