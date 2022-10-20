import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x))


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.LeakyReLU(0.2, True)

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


class SRDenseNet(nn.Module):
    def __init__(self, num_channels=1, growth_rate=3, num_blocks=2, num_layers=2):
        super(SRDenseNet, self).__init__()

        # low level features
        self.conv = ConvLayer(num_channels, growth_rate * num_layers, 3)

        # high level features
        self.dense_blocks = []
        for i in range(num_blocks):
            self.dense_blocks.append(DenseBlock(growth_rate * num_layers * (i + 1), growth_rate, num_layers))
        self.dense_blocks = nn.Sequential(*self.dense_blocks)

        # bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(growth_rate * num_layers + growth_rate * num_layers * num_blocks, growth_rate * num_layers * num_blocks, kernel_size=1),
            nn.LeakyReLU(0.2, True)
        )

        # reconstruction layer
        self.reconstruction = nn.Conv2d(growth_rate * num_layers * num_blocks, num_channels, kernel_size=3, padding=3 // 2)
        self.tanh = nn.Tanh()
        # self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.combine_layer = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, padding='same'),
            nn.Tanh()
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
      print('initializing weights of a model')
      for module in self.modules():
          if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
              nn.init.kaiming_normal_(module.weight)
              # module.weight.data *= 1.2
              if module.bias is not None:
                  nn.init.constant_(module.bias, 0)

    def forward(self, input_edge , lr_image, train = True):
        x1 = self.conv(input_edge)
        x2 = self.dense_blocks(x1)
        x3 = self.bottleneck(x2)
        x4 = self.reconstruction(x3)
        x5 = self.tanh(x4)

        x6 = x5+ lr_image
        
        if train:
            return x5, x6
        else:
            return x6

    def save(self,model_weights,opt,path,optimizer_weights,epoch):
         torch.save({
                    'epoch': epoch,
                    'growth_rate': opt.growth_rate,
                    'num_blocks':opt.num_blocks,
                    'num_layers':opt.num_layers,
                    'model_state_dict': model_weights,
                    'optimizer_state_dict': optimizer_weights,
                    }, path) 