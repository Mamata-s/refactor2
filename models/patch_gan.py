# https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8
import torch
from torch import nn, optim
from loss.ganloss import GANLoss
from models.unet import Unet


class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down - 1) else 2)
                  for i in range(n_down)]  # the 'if' statement is taking care of not using
        # stride of 2 for the last block in this loop
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False,
                                  act=False)]  # Make sure to not use normalization or
        # activation for the last layer of the model
        self.model = nn.Sequential(*model)

    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True,
                   act=True):  # when needing to make some repeatitive blocks of layers,
        layers = [
            nn.Conv2d(ni, nf, k, s, p, bias=not norm)]  # it's always helpful to make a separate method for that purpose
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def init_weights(net, init='norm', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
    
    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net


def init_model(model, device,init="norm"):
    model = model.to(device);print(model)
    model = init_weights(model,init=init)
    return model


class PatchGAN(nn.Module):
    def __init__(self,opt, net_G = None,
                 beta1=0.5, beta2=0.999):
        super().__init__()

        self.device = opt.device
        self.lambda_L1 = opt.lambda_L1
        
        if net_G is None:
            self.net_G = init_model(Unet(in_channels= 1,
                 out_channels= 1,
                 n_blocks=opt.n_blocks,
                 start_filters=opt.start_filters,
                 activation=opt.activation,
                 normalization=opt.normalization,
                 conv_mode = opt.conv_mode,
                 dim= opt.dim,
                 up_mode=opt.up_mode), self.device,init=opt.init)
        else:
            self.net_G = net_G.to(self.device)
        self.net_D = init_model(PatchDiscriminator(input_c=1, n_down=opt.n_down, num_filters=opt.num_filters), self.device)
        self.GANcriterion = GANLoss(gan_mode=opt.gan_mode).to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr= opt.lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr= opt.lr_D, betas=(beta1, beta2))

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, images,labels):
        self.images =images.to(self.device)
        self.labels = labels.to(self.device)

    def forward(self):
        self.fake_images = self.net_G(self.images)

    def backward_D(self):
        fake_preds = self.net_D(self.fake_images.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_preds = self.net_D(self.labels)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_preds = self.net_D(self.fake_images)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_images, self.labels) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def backward_G_l1(self):
        self.loss_G_L1 = self.L1criterion(self.fake_images, self.labels)
        self.loss_G_L1.backward()

    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

        

    def optimize_l1(self):
        self.forward()
        self.net_G.train()
        self.opt_G.zero_grad()
        self.backward_G_l1()
        self.opt_G.step()