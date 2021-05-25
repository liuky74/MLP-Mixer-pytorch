import torch
from torch import nn
class TmpModel(nn.Module):
    def __init__(self):
        super(TmpModel, self).__init__()
        self.conv_1 = nn.Conv2d(3,64,)



"""
0 0 0 0 0 0
0 1 1 1 1 1
0 1 1 1 1 1
0 1 1 1 1 1
0 1 1 1 1 1 
0 1 1 1 1 1
0 0 0 0 0 0 

"""