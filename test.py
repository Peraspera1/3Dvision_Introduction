import torch
from datagenarate import Dataset



psi_sch = torch.load('psi_train_sch.pth')
velocity_sch = torch.load('velocity_sch.pth')

print(psi_sch.dtype)
print(psi_sch.shape)
print(velocity_sch.dtype)
print(velocity_sch.shape)

'''
torch.float32
torch.Size([99, 4, 256, 256])
torch.float32
torch.Size([99, 2, 256, 256])
'''

# 加载数据
# psi_train = torch.load('psi_train.pth')
# velocity = torch.load('velocity.pth')
# viscous_item = torch.load('viscous_item.pth')

# print(psi_train.dtype)
# print(psi_train.shape)

# print(velocity.dtype)
# print(velocity.shape)

# print(viscous_item.dtype)
# print(viscous_item.shape)
'''
torch.float32
torch.Size([200, 4, 256, 256])
torch.float32
torch.Size([200, 2, 256, 256])
torch.float32
torch.Size([200, 2, 256, 256])
'''

