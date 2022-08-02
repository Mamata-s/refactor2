'''
checkpoints to load the models
model_name to load correct model and use their function to load weights from checkpoints
factor: determines on which downsampled factor images model is tested
plot-dir: dir where plots is saved---leave as default
output-dir: dir where outputs are saved --- leave as deault
addition: boolean (not implemented) if True add the pred with images to get the final output of the model
edges: boolean(not implemented) --if True load canny edges as input to model

Possible error: Make sure factor_2, factor_4 folder of downsample images are not empty, if not exists then it will generate

'''

import torch
import torch.nn as nn
import  cv2
import matplotlib.pyplot as plt
import argparse

from utils.load_model import load_model_main
from utils.preprocess import tensor2image, image2tensor
from utils.prepare_test_set_image import crop_pad_kspace, prepare_image_fourier,prepare_image_gaussian
from utils.image_quality_assessment import PSNR,SSIM
from utils.general import min_max_normalize, NRMSELoss
from models.densenet_new import SRDenseNet
from utils.debug_utils import compare_models
import os
import numpy as np

import warnings
warnings.filterwarnings("ignore")


label_path ='test_set/labels/'


def load_model(checkpoint):
    checkpoint = torch.load(checkpoint,map_location=torch.device(opt.device))
    model = SRDenseNet()
    state_dict = model.state_dict()
    for n, p in checkpoint['model_state_dict'].items():
        new_key = n[7:]
        if new_key in state_dict.keys():
            state_dict[new_key].copy_(p)
        else:
            raise KeyError(new_key)
    return model




# ****************************************************************************************************************************************


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model demo')
    parser.add_argument('--checkpoint1', type=str, metavar='',required=False,help='checkpoint path',
    default='outputs/resolution_dataset25_small4/srdense/hrdownsample_z_axis25_small4_mask_training_addition_f4_105_0.0001/checkpoints/z_axis/factor_4/epoch_100_f_4.pth')

    parser.add_argument('--checkpoint2', type=str, metavar='',required=False,help='checkpoint path',
    default='outputs/resolution_dataset25_small4/srdense/hrdownsample_z_axis25_small4_mask_training_addition_f4_105_0.0001/checkpoints/z_axis/factor_4/epoch_80_f_4.pth')


    opt = parser.parse_args()
  

    '''set device'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device

    model1 = load_model(opt.checkpoint1)
    model2 = load_model(opt.checkpoint2)

    # if device != 'cpu':
    #     num_of_gpus = torch.cuda.device_count()
    #     model1 = nn.DataParallel(model1,device_ids=[*range(num_of_gpus)])
    #     model2 = nn.DataParallel(model2,device_ids=[*range(num_of_gpus)])

    model1.to(device)
    model1.eval()

    model2.to(device)
    model2.eval()


    model3 = model1.state_dict()
    for n, p in model1.state_dict().items():
        # print(n)
        if n in model2.state_dict().keys():
            new_val = p + model2.state_dict()[n]
            model3[n].copy_(new_val)
            # print(p[0][0])
            # print(new_val[0][0])
            # print(model2.state_dict()[n][0][0])
            print(p.min().item(),p.max().item())
            print(new_val.min().item(),new_val.max().item())
        else:
            raise KeyError(n)
