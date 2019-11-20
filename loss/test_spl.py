import torch
import pytorch_spl_loss as spl

# Gradient Profile Loss
GPL =  spl.GPLoss()

# Color Profile Loss
# You can define the desired color spaces in the initialization
# default is True for all
CPL =  spl.CPLoss(rgb=True,yuv=True,yuvgrad=True)

target    = torch.randn(1,3,256,256)
generated = torch.randn(1,3,256,256)

gpl_value = GPL(generated,target)
cpl_value = CPL(generated,target)

spl_value = gpl_value + cpl_value
print(gpl_value)
print(cpl_value)
print(spl_value)