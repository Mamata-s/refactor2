import torch
from torch import nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.conv(x))


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return torch.cat([x, self.leaky_relu(self.conv(x))], 1)


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
    def __init__(self, num_channels=1, growth_rate=4, num_blocks=4, num_layers=3):
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
            nn.Conv2d(growth_rate * num_layers + growth_rate * num_layers * num_blocks, 256, kernel_size=1),
            nn.LeakyReLU(0.1)
        )

        # deconvolution layers
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=3 // 2, output_padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=3 // 2, output_padding=1),
            nn.LeakyReLU(0.1)
        )

        # reconstruction layer
        self.reconstruction = nn.Conv2d(256, num_channels, kernel_size=3, padding=3 // 2)
        self.tanh = nn.Tanh()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.conv(x)
        x = self.dense_blocks(x)
        x = self.bottleneck(x)
        # x = self.deconv(x)
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


class SRDenseNetUpscale(nn.Module):
    def __init__(self, num_channels=1, growth_rate=4, num_blocks=4, num_layers=3):
        super(SRDenseNetUpscale, self).__init__()

        # low level features
        self.conv = ConvLayer(num_channels, growth_rate * num_layers, 3)

        # high level features
        self.dense_blocks = []
        for i in range(num_blocks):
            self.dense_blocks.append(DenseBlock(growth_rate * num_layers * (i + 1), growth_rate, num_layers))
        self.dense_blocks = nn.Sequential(*self.dense_blocks)

        # bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(growth_rate * num_layers + growth_rate * num_layers * num_blocks, 256, kernel_size=1),
            nn.LeakyReLU(0.1)
        )

        # deconvolution layers
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=3 // 2, output_padding=1),
            nn.LeakyReLU(0.1),
            # nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=3 // 2, output_padding=1),
            # nn.LeakyReLU(0.1)
        )

        # reconstruction layer
        self.reconstruction = nn.Conv2d(256, num_channels, kernel_size=3, padding=3 // 2)
        self.tanh = nn.Tanh()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.conv(x)
        x = self.dense_blocks(x)
        x = self.bottleneck(x)
        x = self.deconv(x)
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