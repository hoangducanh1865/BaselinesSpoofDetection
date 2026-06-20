import torch.nn as nn
from .blocks.frontend import DWS_Frontend_SE, DWS_Frontend_SE_small
from .blocks.classifier import TfSepNet
import torch

try:
    from torchinfo import summary
except ModuleNotFoundError:
    summary = None


class RawTFNet(nn.Module):
    
    def __init__(self, sample_rate: int = 16000):
        super(RawTFNet, self).__init__()
        self.front_end = DWS_Frontend_SE(sinc_kernel_size=128, sample_rate=sample_rate)
        
        
        self.classifier = TfSepNet(depth=10, width=32, dropout_rate=0.2, shuffle=True, shuffle_groups=8)

    def forward(self, x):
        x = self.front_end(x)
        x = self.classifier(x)        
        return x

class RawTFNet_small(nn.Module):
    
    def __init__(self, sample_rate: int = 16000):
        super(RawTFNet_small, self).__init__()
        self.front_end = DWS_Frontend_SE_small(sinc_kernel_size=128, sample_rate=sample_rate)
        
        
        self.classifier = TfSepNet(depth=18, width=16, dropout_rate=0.2, shuffle=True, shuffle_groups=8)

    def forward(self, x):
        x = self.front_end(x)
        x = self.classifier(x)        
        return x
    
if __name__ == "__main__":
    args = {}
    model = RawTFNet_small(args, device="cpu")
    x = torch.randn(2, 16000*4)
    if summary is not None:
        summary(model, (2, 16000*4))
