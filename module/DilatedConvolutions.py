import numpy as np
from torchvision.transforms import Compose, ToTensor
from torch import nn
import torch.nn.init as init
def transform():
    return Compose([
        ToTensor(),
        # Normalize((12,12,12),std = (1,1,1)),
    ])

arr = range(1,26)
arr = np.reshape(arr,[5,5])
arr = np.expand_dims(arr,2)
arr = arr.astype(np.float32)
# arr = arr.repeat(3,2)
print(arr.shape) #(5, 5, 1)
arr = transform()(arr)
arr = arr.unsqueeze(0)
print(arr)
print(arr.shape)#torch.Size([1, 1, 5, 5])
conv1 = nn.Conv2d(1, 1, 3, stride=1, bias=False, dilation=1, padding=0)  # 普通卷积
conv2 = nn.Conv2d(1, 1, 3, stride=1, bias=False, dilation=2, padding=0)  # dilation就是空洞率，即间隔
init.constant_(conv1.weight, 1)
init.constant_(conv2.weight, 1)
out1 = conv1(arr)
print(out1.shape) #torch.Size([1, 1, 3, 3])
out2 = conv2(arr)
print(out2.shape) #torch.Size([1, 1, 1, 1])
print('standare conv:\n', out1.detach().numpy())
print('dilated conv:\n', out2.detach().numpy())
