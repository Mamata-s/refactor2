
import torch.nn as nn
import torch

class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def save(self,model_weights,opt,path,optimizer_weights,epoch):
        torch.save({
                    'training_type':opt.training_type,
                    'epoch': epoch,
                    'model_state_dict': model_weights,
                    'optimizer_state_dict': optimizer_weights,
                    }, path) 