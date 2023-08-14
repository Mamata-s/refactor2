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

from email.mime import image
from tabnanny import check
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
# from models.densenet_edges import SRDenseNet
from models.densenet_smchannel import SRDenseNet
import os
import numpy as np
from utils.train_utils import normalize_edges, denormalize_edges
from utils.preprocess import hfen_error
from utils.train_utils import read_dictionary

import warnings
warnings.filterwarnings("ignore")
import random

label_path = 'upsample/combine/hr_25/z_axis'

def load_model(opt):
    checkpoint = torch.load(opt.checkpoint,map_location=torch.device(opt.device))
    growth_rate = checkpoint['growth_rate']
    num_blocks = checkpoint['num_blocks']
    num_layers =checkpoint['num_layers']
    model = SRDenseNet(growth_rate=growth_rate,num_blocks=num_blocks,num_layers=num_layers)
    state_dict = model.state_dict()
    for n, p in checkpoint['model_state_dict'].items():
        new_key = n[7:]
        # new_key = n
        # print(new_key)
        if new_key in state_dict.keys():
            # new_p = p*random.uniform(2., 7.)
            state_dict[new_key].copy_(p)
        else:
            raise KeyError(new_key)
    return model


def save_array(path,array):
    f = open(path, "w")

    for d in array:
        f.write(f"{d}\n")
    f.close()


# define function that combines all three image quality metrics
def compare_images(target, ref,psnr,ssim):
    mse = nn.MSELoss()
    nrmse = NRMSELoss()
    scores = []
    scores.append(psnr(target, ref).item())
    scores.append(mse(target, ref).item())
    scores.append(ssim(target, ref).item())
    scores.append(nrmse(ref, target).item())

    ref_arr = (ref.squeeze().detach().cpu().numpy())*255.
    target_arr = (target.squeeze().detach().cpu().numpy())*255.
    scores.append(hfen_error(ref_arr,target_arr).item())
    
    return scores


# ****************************************************************************************************************************************
# for dictionary having lr image name as key


def predict(model,image_path,device,psnr,ssim,label_path):
    
    degraded = cv2.imread(image_path)
    degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # ref = cv2.imread('test_set/labels/{}'.format(file))

    ref = cv2.imread(label_path)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)

    print("Image path", image_path)
    print("Label path", label_path)
    print("**********************************************************************************************************************")

    degraded = degraded/255.
    ref = ref/255.
    degraded = image2tensor(degraded).unsqueeze(dim=0).to(device)
    ref = image2tensor(ref).unsqueeze(dim=0).to(device)
    
    # perform super-resolution with srcnn
    pre = model(degraded)
    
    # image quality calculations
    scores = []
    scores.append(compare_images(degraded, ref,psnr,ssim))
    scores.append(compare_images(pre, ref,psnr,ssim))
    

    # return images and scores
    return ref, degraded, pre, scores

def save_results(model,opt):
    for file in opt.dictionary:
        image_name = file
        label_name = opt.dictionary[file]
        image_path = os.path.join(opt.image_path,image_name)
        label_path = os.path.join(opt.label_path,label_name)

        print("Image name", image_path)
        print("label name", label_path)
        print("********************************************************************************")

        # perform super-resolution
        ref, degraded, output, scores = predict(model=model,image_path=image_path,device=opt.device,psnr=opt.psnr,ssim=opt.ssim,label_path=label_path)
        
        # display images as subplots
        fig, axs = plt.subplots(1, 3, figsize=(20, 8))
        axs[0].imshow(tensor2image(degraded),cmap='gray')
        axs[0].set_title('Degraded')
        axs[0].set(xlabel = 'PSNR: {:.6f} \nMSE: {:.6f} \nSSIM: {:.6f}'.format(scores[0][0], scores[0][1], scores[0][2]))
        axs[1].imshow(tensor2image(output),cmap='gray')
        axs[1].set_title(opt.model_name)
        axs[1].set(xlabel = 'PSNR: {:.6f} \nMSE: {:.6f} \nSSIM: {:.6f}'.format(scores[1][0], scores[1][1], scores[1][2]))
        axs[2].imshow(tensor2image(ref),cmap='gray')
        axs[2].set_title('Original')

        # remove the x and y ticks
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        
        print('Saving {}'.format(file))
        print(fig.savefig(opt.plot_dir+'{}.png'.format(os.path.splitext(file)[0])) )
        print(cv2.imwrite(opt.preds_dir+'{}.png'.format(os.path.splitext(file)[0]),tensor2image(output)))
        plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model demo')
    parser.add_argument('--checkpoint', type=str, metavar='',required=False,help='checkpoint path', default='outputs/gaussian_dataset25_mul_wo_circular_mask/srdense/gaussian_mul_wo_circular_mask/checkpoints/z_axis/factor_2/epoch_1000_f_2.pth')
    parser.add_argument('--plot-dir', type=str, metavar='',help='plot save dir',default='test_set/plots_2/')
    parser.add_argument('--preds-dir', type=str, metavar='',help='plot save dir',default='test_set/preds_2/')
    opt = parser.parse_args()

    index_value = 0
    checkpoints =[
       'outputs/gaussian_dataset25_sigma100/srdense/gaussian_mul_wo_circular_mask_sigma100/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
       'outputs/dataset_images/hanning_dataset25/srdense/srdense_hanning_blur/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
       'outputs/dataset_images/hamming_dataset25/srdense/srdense_hamming_blur/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
       'outputs/dataset_images/bicubic_dataset25/srdense/bicubic_up_and_down_factor_2/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
       'outputs/dataset_images/mean_blur_dataset25/srdense/srdense_mean_blur/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
       'outputs/dataset_images/median_blur_dataset25/srdense/srdense_median_blur/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
        'outputs/dataset_images/combine_all_degradation_fix_dataset/srdense/srdense_combine_all_degradation_fix_dataset/checkpoints/z_axis/factor_2/epoch_450_f_2.pth',
       'outputs/dataset_images/combine_all_degradation_large_dataset/srdense/srdense_combine_all_degradation_large_dataset/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'
    ] 
    model_names = ['gaussian_sigma', 'hann_model', 'hamm_model', 'bicubic_model', 'mean_model', 'median_model', 'combine_fix', 'combine_large']
    dataset_names = ['gaussian_sigma', 'hann', 'hamm', 'bicubic', 'mean', 'median']

    image_path = [
        'plot_set/gaussian_sigma100',
        'plot_set/hann',
        'plot_set/hamm',
        'plot_set/biubic',
        'plot_set/mean_blur',
        'plot_set/median_blur'
    ]
    label_path = 'plot_set/label'
    annotation_paths=  [

    ]

    opt.model_name = model_names[index_value]
    opt.label_path = label_path
  

    '''set device'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device
    opt.label_path = label_path

    psnr = PSNR()
    ssim = SSIM()

    psnr = psnr.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)
    ssim = ssim.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)

    opt.psnr= psnr
    opt.ssim=ssim

    for index,dataset in enumerate(image_path): # looping through all dataset

        # parameter for each dataset
        opt.image_path = image_path[index]
        opt.dictionary = read_dictionary(annotation_paths[index]) 
        opt.image_path = image_path[index]
        opt.dataset_name = dataset_names[index]

        # loading the model defined by index_value variable
        model = load_model(opt)
        if device != 'cpu':
            num_of_gpus = torch.cuda.device_count()
            model = nn.DataParallel(model,device_ids=[*range(num_of_gpus)])

        model.to(device)

        model.eval()
        opt.plot_dir = os.path.join('plot_set',opt.model_name, opt.dataset_name, 'plots')
        opt.plot_dir = os.path.join('plot_set',opt.model_name, opt.dataset_name,'preds')
        if not os.path.exists(opt.plot_dir):
            os.makedirs(opt.plot_dir)
        if not os.path.exists(opt.preds_dir):
            os.makedirs(opt.preds_dir)
        save_results(model,opt)
