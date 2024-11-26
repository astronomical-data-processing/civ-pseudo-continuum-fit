import torch
from torch import sigmoid
import torch.nn as nn


class UNet1Sigmoid(nn.Module):
    def __init__(self, n_channels, n_classes, hidden=32, kernel_size=None):
        super(type(self), self).__init__()
        self.inc    = inconv(n_channels, hidden, kernel_size)
        self.down1  = down(hidden, hidden * 2, kernel_size)
        self.up8    = up(hidden * 2, hidden, kernel_size)
        self.outc   = outconv(hidden, n_classes)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x = self.up8(x2, x1)
        x = self.outc(x)
        return sigmoid(x)

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(type(self), self).__init__()
        self.conv = double_conv(in_ch, out_ch, kernel_size)
    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(type(self), self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool1d(2),
            double_conv(in_ch, out_ch, kernel_size)
        )
    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(type(self), self).__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch, kernel_size)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(type(self), self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1)
    def forward(self, x):
        x = self.conv(x)
        return x
    
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        print('in_ch,kernel_size =',in_ch,kernel_size)
        super(type(self), self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=int(((in_ch-1)*1 - in_ch + kernel_size)/2)), 
            nn.BatchNorm1d(out_ch, momentum=0.005),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=int(((in_ch-1)*1 - in_ch + kernel_size)/2)),
            nn.BatchNorm1d(out_ch, momentum=0.005),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class BCELosswithLogits(nn.Module):
    def __init__(self):
        super(BCELosswithLogits, self).__init__()
        
    def forward(self, logits, target, fuyangben_weight):
        import torch

        eps = 1e-7
        logits = torch.clamp(logits, eps, (1 - eps))

        loss = 0.5 * (logits - target)**2

        loss = loss.mean() 

        return loss
