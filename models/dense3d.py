import torch
from torch import nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding = kernel_size // 2)
        self.relu = nn.LeakyReLU(0.2)
        

    def forward(self, x):
        return self.relu(self.conv(x))

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.block = [ConvLayer(in_channels, growth_rate, kernel_size=3)]
        for i in range(num_layers - 1):
            self.block.append(DenseLayer(growth_rate * (i + 1), growth_rate, kernel_size=3))
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return torch.cat([x, self.block(x)], 1)


class SR3DDenseNet(nn.Module):
    def __init__(self, num_channels=1, growth_rate=5, num_blocks=2, num_layers=4):
        super(SR3DDenseNet, self).__init__()

        # low level features
        self.conv = ConvLayer(num_channels, growth_rate * num_layers, 3)

        # high level features
        self.dense_blocks = []
        for i in range(num_blocks):
            self.dense_blocks.append(DenseBlock(growth_rate * num_layers * (i + 1), growth_rate, num_layers))
        self.dense_blocks = nn.Sequential(*self.dense_blocks)

        # bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv3d(growth_rate * num_layers + growth_rate * num_layers * num_blocks, 6, kernel_size=1),
            nn.LeakyReLU(0.2)
        )
      
        # reconstruction layer
        self.reconstruction = nn.Conv3d(6, num_channels, kernel_size=3, padding=3 // 2)

        # self._initialize_weights()
       
        self.tanh = nn.Tanh()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
    
        x = self.conv(x)
    
        x = self.dense_blocks(x)
       
        x = self.bottleneck(x)
        
        # x = self.deconv(x)  removed
        x = self.tanh(self.reconstruction(x))

        return x

    def save(self,model_weights,opt,path,optimizer_weights,epoch):
         torch.save({
                    'training_type':opt.training_type,
                    'epoch': epoch,
                    'growth_rate': opt.growth_rate,
                    'num_blocks':opt.num_blocks,
                    'num_layers':opt.num_layers,
                    'model_state_dict': model_weights,
                    'optimizer_state_dict': optimizer_weights,
                    }, path) 