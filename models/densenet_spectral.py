# import torch
# import torch.nn as nn
# from torch.nn.utils import spectral_norm


# class ConvLayer(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size):
#         super(ConvLayer, self).__init__()
#         self.conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
#         self.relu = nn.LeakyReLU(0.1)

#     def forward(self, x):
#         return self.relu(self.conv(x))


# class DenseLayer(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size):
#         super(DenseLayer, self).__init__()
#         self.conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
#         self.relu = nn.LeakyReLU(0.1)

#     def forward(self, x):
#         return torch.cat([x, self.relu(self.conv(x))], 1)


# class DenseBlock(nn.Module):
#     def __init__(self, in_channels, growth_rate, num_layers):
#         super(DenseBlock, self).__init__()
#         self.block = [ConvLayer(in_channels, growth_rate, kernel_size=3)]
#         for i in range(num_layers - 1):
#             self.block.append(DenseLayer(growth_rate * (i + 1), growth_rate, kernel_size=3))
#         self.block = nn.Sequential(*self.block)

#     def forward(self, x):
#         return torch.cat([x, self.block(x)], 1)


# class SRDenseNet(nn.Module):
#     def __init__(self, num_channels=1, growth_rate=7, num_blocks=5, num_layers=5):
#         super(SRDenseNet, self).__init__()

#         # low level features
#         self.conv = ConvLayer(num_channels, growth_rate * num_layers, 3)

#         # high level features
#         self.dense_blocks = []
#         for i in range(num_blocks):
#             self.dense_blocks.append(DenseBlock(growth_rate * num_layers * (i + 1), growth_rate, num_layers))
#         self.dense_blocks = nn.Sequential(*self.dense_blocks)

#         # bottleneck layer
#         self.bottleneck = nn.Sequential(
#             spectral_norm(nn.Conv2d(growth_rate * num_layers + growth_rate * num_layers * num_blocks, 256, kernel_size=1)),
#             nn.LeakyReLU(0.1)
#         )

#         # reconstruction layer
#         self.reconstruction = nn.Conv2d(256, num_channels, kernel_size=3, padding=3 // 2)
#         self.tanh = nn.Tanh()
#         self.leaky_relu = nn.LeakyReLU(0.1)

#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#                 nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias.data)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.dense_blocks(x)
#         x = self.bottleneck(x)
#         x = self.tanh(self.reconstruction(x))
#         return x




import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2,bias=False)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x))


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2,bias=False)
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
    def __init__(self, num_channels=1, growth_rate=3, num_blocks=2, num_layers=3):
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
            nn.Conv2d(growth_rate * num_layers + growth_rate * num_layers * num_blocks, 256, kernel_size=1,bias=False),
            nn.LeakyReLU(0.2, True)
        )

        # reconstruction layer
        self.reconstruction = nn.Conv2d(256, num_channels, kernel_size=3, padding=3 // 2,bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
      for module in self.modules():
          if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
              nn.init.kaiming_normal_(module.weight)
              module.weight.data *= 0.1
              if module.bias is not None:
                  nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.dense_blocks(x)
        x = self.bottleneck(x)
        x =self.reconstruction(x)
        return x