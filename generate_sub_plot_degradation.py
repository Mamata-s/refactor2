'''
This file is created to test the model trained on corresponding pair of 25 and 50 micron images
 based on the minimum mse, nrmse,hfen or maximum psnr,ssim
'''

from email.mime import image
from tabnanny import check
import torch
import torch.nn as nn
import  cv2
import matplotlib.pyplot as plt
import argparse
import itertools

from utils.load_model import load_model_main
from utils.preprocess import tensor2image, image2tensor
from utils.prepare_test_set_image import crop_pad_kspace, prepare_image_fourier,prepare_image_gaussian
from utils.image_quality_assessment import PSNR,SSIM
from utils.general import min_max_normalize, NRMSELoss
from models.densenet_smchannel import SRDenseNet
import os
import numpy as np
from utils.train_utils import normalize_edges, denormalize_edges
from utils.preprocess import hfen_error
from utils.train_utils import read_dictionary
import matplotlib.patches as patches

import warnings
warnings.filterwarnings("ignore")
import random


# generates a subplots as single image vs all models

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
        if new_key in state_dict.keys():
            pass
        else:
            new_key = n
        state_dict[new_key].copy_(p)
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

    # print("Image path", image_path)
    # print("Label path", label_path)
    # print("**********************************************************************************************************************")

    degraded = degraded/255.
    ref = ref/255.
    degraded = image2tensor(degraded).unsqueeze(dim=0).to(device)
    ref = image2tensor(ref).unsqueeze(dim=0).to(device)
    
    # perform super-resolution with srcnn
    pre = model(degraded)
    
    # image quality calculations
    # scores = []
    # scores.append(compare_images(degraded, ref,psnr,ssim))
    print('initial scores',compare_images(degraded, ref,psnr,ssim))
    # scores.append(compare_images(pre, ref,psnr,ssim))

    scores = compare_images(degraded, ref,psnr,ssim)
    

    # return images and scores
    return ref, degraded, pre, scores

def save_results(result_image_list,result_image_list_key,label_image,save_dir,save_name, scores):
    first_index = 0
    second_index = 2 # 3 means nrmse

    start_h =  8       
    end_h = 190                

    start_w = 284    
    end_w = 466      

    plt.figure(figsize=(15, 7))
    fnsize =15
    ax1 = plt.subplot(2,4,2)
    ax1.imshow(result_image_list[0][start_h:end_h, start_w:end_w], cmap='gray')
    ax1.set_title(result_image_list_key[0]+ ' ({:.1f}/{:.1f})'.format(scores[0][first_index], scores[0][second_index]),fontsize=fnsize)
    # ax1.set_xlabel('({:.3f},{:.3f})'.format(scores[0][first_index], scores[0][second_index]),fontsize=fnsize)

    ax2 = plt.subplot(2,4,3)
    ax2.imshow(result_image_list[1][start_h:end_h, start_w:end_w], cmap='gray')
    ax2.set_title(result_image_list_key[1]+' ({:.1f},{:.1f})'.format(scores[1][first_index], scores[1][second_index]),fontsize=fnsize)
    # ax2.set_xlabel('({:.3f},{:.3f})'.format(scores[1][first_index], scores[1][second_index]),fontsize=fnsize)

    ax3 = plt.subplot(2,4,4)
    ax3.imshow(result_image_list[2][start_h:end_h, start_w:end_w], cmap='gray')
    ax3.set_title(result_image_list_key[2]+ ' ({:.1f}/{:.1f})'.format(scores[2][first_index], scores[2][second_index]),fontsize=fnsize)
    # ax3.set_xlabel('({:.3f},{:.3f})'.format(scores[2][first_index], scores[2][second_index]),fontsize=fnsize)

    ax4 = plt.subplot(2,4,6)
    ax4.imshow(result_image_list[3][start_h:end_h, start_w:end_w], cmap='gray')
    ax4.set_title(result_image_list_key[3]+' ({:.1f},{:.1f})'.format(scores[3][first_index], scores[3][second_index]),fontsize=fnsize)
    # ax4.set_xlabel('({:.3f},{:.3f})'.format(scores[3][first_index], scores[3][second_index]),fontsize=fnsize)

    ax5 = plt.subplot(2,4,7)
    ax5.imshow(result_image_list[4][start_h:end_h, start_w:end_w], cmap='gray')
    ax5.set_title(result_image_list_key[4]+' ({:.1f}/{:.1f})'.format(scores[4][first_index], scores[4][second_index]),fontsize=fnsize)
    # ax5.set_xlabel('({:.3f},{:.3f})'.format(scores[4][first_index], scores[4][second_index]),fontsize=fnsize)

    ax6 = plt.subplot(2,4,8)
    ax6.imshow(result_image_list[5][start_h:end_h, start_w:end_w], cmap='gray')
    ax6.set_title(result_image_list_key[5]+ ' ({:.1f}/{:.1f})'.format(scores[5][first_index], scores[5][second_index]),fontsize=fnsize)
    # ax6.set_xlabel('({:.3f},{:.3f})'.format(scores[5][first_index], scores[5][second_index]),fontsize=fnsize)

    # ax7 = plt.subplot(2,4,9)
    # ax7.imshow(result_image_list[6][328:510, 224:406], cmap='gray')
    # ax7.set_title(result_image_list_key[6])
    # ax7.set_xlabel('({:.3f},{:.3f})'.format(scores[6][first_index], scores[6][second_index]))

    # ax8 = plt.subplot(2,5,10)
    # ax8.imshow(result_image_list[7][328:510, 224:406], cmap='gray')
    # ax8.set_title(result_image_list_key[7])
    # ax8.set_xlabel('({:.3f},{:.3f})'.format(scores[7][first_index], scores[7][second_index]))

    ax11 = plt.subplot(1,4,1)
    ax11.imshow(label_image, cmap='gray')
    ax11.set_title('ground truth (PSNR/SSIM)',fontsize=fnsize)
    rect = patches.Rectangle((290, 30), 182, 182, linewidth=1, edgecolor='r', facecolor='none')  # first is horizontal, second is verticle
    # ax11.set_xlabel('(PSNR/NRMSE)',fontsize=fnsize)


    # Add the patch to the Axes
    ax11.add_patch(rect)


    axes = [ax1, ax2, ax3, ax4, ax5, ax6,ax11]
    for ax in axes:
        ax.xaxis.set_tick_params(length=0,labelbottom=False)
        ax.yaxis.set_tick_params(length=0,labelbottom=False)
        ax.set_xticklabels([]),ax.set_yticklabels([])
    

    print('Saving {}'.format(save_name))
    print(plt.savefig(save_dir+'{}.png'.format(save_name)) )
    plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model demo')
    parser.add_argument('--evaluate', type=str, metavar='',required=False,help='just for using opt', default='default')
    opt = parser.parse_args()

    index_value = 0
    checkpoint='outputs/dataset_images_srdense/L1_loss/hamming_dataset25/srdense/srdense_hamming_dataset25_l1/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'

    '''set device'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device

    psnr = PSNR()
    ssim = SSIM()

    psnr = psnr.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)
    ssim = ssim.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)

    opt.psnr= psnr
    opt.ssim=ssim


    label_path = 'dataset_images/dataset_index/hr_label/hr_f4_149_z_112.png'


    image_paths = [
     'plot_set/gaussian_sigma100/lr_f4_149_z_112_g100.png',
     'plot_set/hann/lr_f4_149_2_z_ham112.png',
     'plot_set/hamm/lr_f4_149_2_z_ham112.png',
     'plot_set/bicubic/lr_f4_149_2_z_112.png',
     'plot_set/mean_blur/lr_f4_149_2_z_b112.png',
     'plot_set/median_blur/lr_f4_149_2_z_b112.png'
    ]
    

    label_image =  cv2.imread(label_path)
    label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2GRAY).astype(np.float32)


    result_image_list = [label_image]
    result_image_list_key = ['ref']

    result_image_list = []
    result_image_list_key = []
    keys = ['gaussian','hanning','hamming','bicubic','mean','median']


    ref_score = [0,0,0,0,0,0,0]
    score_list = []
    # score_list.append(ref_score)

   
    for index,image_path in enumerate(image_paths): # looping through the model checkpoints

        opt.checkpoint = checkpoint
        print("Evaluating")
        print('checkpoint', checkpoint)
        print("*******************************************************************************************************")

        model = load_model(opt)
        if device != 'cpu':
            num_of_gpus = torch.cuda.device_count()
            model = nn.DataParallel(model,device_ids=[*range(num_of_gpus)])
        model.to(device)
        model.eval()

        ref, degraded, output, scores = predict(model=model,image_path=image_path,device=opt.device,psnr=opt.psnr,ssim=opt.ssim,label_path=label_path)
        result_image_list.append(tensor2image(degraded))
        result_image_list_key.append(keys[index])
        score_list.append(scores)


    # after getting result from all the model create a subplot
    save_dir='plot_set/subplots/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_name='degradation_type'

    save_results(result_image_list=result_image_list,result_image_list_key=result_image_list_key,label_image=label_image,save_dir=save_dir,save_name=save_name, scores=score_list)



