
## FashionMNIST 데이터셋의 이미지들을 분류하는 신경망

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

dv = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {dv} device")

class cMyNN(nn.Module):
    def __init__(self):
        super(cMyNN, self).__init__()
        self.myflat = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.myflat(x)
        logits = self.linear_relu_stack(x)
        return logits

model = cMyNN().to(dv)
ifm = torch.rand(30, 28, 28, device=dv)
logits = model(ifm)
pred_probab = nn.Softmax(dim=1)(logits) # dim=1 => axis=1 at NCHW
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
print(f"logits.shape = {logits.shape}")
print(f"pred_probab.shape = {pred_probab.shape}")

print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")



