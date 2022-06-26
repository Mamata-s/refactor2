import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torch.nn.utils import spectral_norm
# from torchvision.models.feature_extraction import create_feature_extractor
# import torchvision.models.feature_extraction.create_feature_extractor as create_feature_extractor
from torchvision import transforms

__all__ = [
    "ResidualDenseBlock", "ResidualResidualDenseBlock",
    "Discriminator", "Generator",
    "ContentLoss"
]


class ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.
    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(ResidualDenseBlock, self).__init__()
        self.conv1 =spectral_norm( nn.Conv3d(channels + growth_channels * 0, growth_channels, 3, 1, 1))
        self.conv2 =spectral_norm( nn.Conv3d(channels + growth_channels * 1, growth_channels, 3,1,1))
        self.conv3 = spectral_norm(nn.Conv3d(channels + growth_channels * 2, growth_channels, 3,1,1))
        self.conv4 = spectral_norm(nn.Conv3d(channels + growth_channels * 3, growth_channels, 3,1,1))
        self.conv5 = spectral_norm(nn.Conv3d(channels + growth_channels * 4, channels,3,1,1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

        # Initialize model weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))
        out = torch.mul(out5, 0.2)
        out = torch.add(out, identity)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class ResidualResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.
    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(ResidualResidualDenseBlock, self).__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = torch.mul(out, 0.2)
        out = torch.add(out, identity)

        return out



class RRDBNet3D(nn.Module):
    def __init__(self,num_block=7) -> None:
        super(RRDBNet3D, self).__init__()
        # The first layer of convolutional layer.
        self.conv1 = nn.Conv3d(1, 64, 3,1,1)

        # Feature extraction backbone network.
        trunk = []
        self.num_block=num_block
        for _ in range(self.num_block):
            trunk.append(ResidualResidualDenseBlock(64, 32))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks.
        self.conv2 = nn.Conv3d(64, 64, 3,1,1)

        # Upsampling convolutional layer.
        self.upsampling1 = nn.Sequential(
            nn.Conv3d(64, 64, 3,1,1),
            nn.LeakyReLU(0.2, True)
        )
        self.upsampling2 = nn.Sequential(
            nn.Conv3d(64, 64, 3,1,1),
            nn.LeakyReLU(0.2, True)
        )

        # Reconnect a layer of convolution block after upsampling.
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 64, 3,1,1),
            nn.LeakyReLU(0.2, True)
        )

        # Output layer.
        self.conv4 = nn.Conv3d(64, 1,3,1,1)

    # The model should be defined in the Torch.script method.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)

        # out = self.upsampling1(F.interpolate(out, scale_factor=1, mode="nearest"))
        # out = self.upsampling2(F.interpolate(out, scale_factor=2, mode="nearest"))

        #replaced to make the size of input and output same
        out = self.upsampling1(out)
        out = self.upsampling2(out)

        out = self.conv3(out)
        out = self.conv4(out)

        out = torch.clamp_(out, 0.0, 1.0)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


    def save(self,model_weights,opt,path,optimizer_weights,epoch):
         torch.save({
                    'epoch': epoch,
                    'num_blocks':opt.num_blocks,
                    'model_state_dict': model_weights,
                    'optimizer_state_dict': optimizer_weights,
                    }, path) 
# model = Generator()
# input = torch.randn((3, 1, 64, 64, 64))
# # input.shape

# output = model(input)
# print(output.shape)