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
import itertools

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
        # new_key = n
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

    scores = compare_images(pre, ref,psnr,ssim)
    

    # return images and scores
    return ref, degraded, pre, scores

def save_results(result_image_list,result_image_list_key,label_image,save_dir,save_name, scores):
    # print(scores)
    # print(len(scores))
    # # print(scores[8][0],scores[8][3])
    # for index,score in enumerate(scores):
    #     print(index)
    #     print('\n')
    #     print(scores[index])

    start_h =37   #369
    end_h = 147

    start_w =303    #265
    end_w = 413
  
    first_index = 0
    second_index = 2 # 3 means nrmse
    fn_size = 15

    fig = plt.figure(figsize=(20, 7))
    fig.set_dpi(900)


    ax1 = plt.subplot(2,5,2)
    ax1.imshow(result_image_list[0][start_h:end_h, start_w:end_w], cmap='gray')
    ax1.set_title(result_image_list_key[0],fontsize=fn_size)

        #Input
    # ax2 = plt.subplot(2,5,3)
    # ax2.imshow(result_image_list[1][328:510, 224:406], cmap='gray')
    # ax2.set_title(result_image_list_key[1])
    # ax2.set_xlabel('({:.3f},{:.3f})'.format(scores[1][first_index], scores[1][second_index]))

    ax3 = plt.subplot(2,5,3)
    ax3.imshow(result_image_list[2][start_h:end_h, start_w:end_w], cmap='gray')
    ax3.set_title(result_image_list_key[2] +' ({:.1f}/{:.1f})'.format(scores[2][first_index], scores[2][second_index]),fontsize=fn_size)
    # ax3.set_xlabel('({:.2f},{:.2f})'.format(scores[2][first_index], scores[2][second_index]),fontsize=fn_size)

    ax4 = plt.subplot(2,5,4)
    ax4.imshow(result_image_list[3][start_h:end_h, start_w:end_w], cmap='gray')
    ax4.set_title(result_image_list_key[3]+' ({:.1f}/{:.1f})'.format(scores[3][first_index], scores[3][second_index]),fontsize=fn_size)
    # ax4.set_xlabel('({:.2f},{:.2f})'.format(scores[3][first_index], scores[3][second_index]),fontsize=fn_size)

    ax5 = plt.subplot(2,5,5)
    ax5.imshow(result_image_list[4][start_h:end_h, start_w:end_w], cmap='gray')
    ax5.set_title(result_image_list_key[4]+' ({:.1f}/{:.1f})'.format(scores[4][first_index], scores[4][second_index]),fontsize=fn_size)
    # ax5.set_xlabel('({:.2f},{:.2f})'.format(scores[4][first_index], scores[4][second_index]),fontsize=fn_size)

    ax6 = plt.subplot(2,5,7)
    ax6.imshow(result_image_list[5][start_h:end_h, start_w:end_w], cmap='gray')
    ax6.set_title(result_image_list_key[5]+' ({:.1f}/{:.1f})'.format(scores[5][first_index], scores[5][second_index]),fontsize=fn_size)
    # ax6.set_xlabel('({:.2f},{:.2f})'.format(scores[5][first_index], scores[5][second_index]),fontsize=fn_size)

    ax7 = plt.subplot(2,5,8)
    ax7.imshow(result_image_list[6][start_h:end_h, start_w:end_w], cmap='gray')
    ax7.set_title(result_image_list_key[6]+' ({:.1f}/{:.1f})'.format(scores[6][first_index], scores[6][second_index]),fontsize=fn_size)
    # ax7.set_xlabel('({:.2f},{:.2f})'.format(scores[6][first_index], scores[6][second_index]),fontsize=fn_size)

    ax8 = plt.subplot(2,5,9)
    ax8.imshow(result_image_list[7][start_h:end_h, start_w:end_w], cmap='gray')
    ax8.set_title(result_image_list_key[7]+' ({:.1f}/{:.1f})'.format(scores[7][first_index], scores[7][second_index]),fontsize=fn_size)
    # ax8.set_xlabel('({:.2f},{:.2f})'.format(scores[7][first_index], scores[7][second_index]),fontsize=fn_size)

    ax9 = plt.subplot(2,5,10)
    ax9.imshow(result_image_list[8][start_h:end_h, start_w:end_w], cmap='gray')
    ax9.set_title(result_image_list_key[8]+ ' ({:.1f}/{:.1f})'.format(scores[8][first_index], scores[8][second_index]),fontsize=fn_size)
    # ax9.set_xlabel('({:.2f},{:.2f})'.format(scores[8][first_index], scores[8][second_index]),fontsize=fn_size)

    # ax10 = plt.subplot(2,6,12)
    # ax10.imshow(result_image_list[9][328:510, 224:406], cmap='gray')
    # ax10.set_title(result_image_list_key[9])

    ax11 = plt.subplot(1,5,1)
    ax11.imshow(label_image, cmap='gray')
    ax11.set_title('ground truth (PSNR/SSIM) ',fontsize=fn_size)
    rect = patches.Rectangle((310, 60), 110, 110, linewidth=1, edgecolor='r', facecolor='none')  # first is horizontal dir, second vert dir
    # ax11.set_xlabel('(PSNR,SSIM)',fontsize=fn_size)

    # Add the patch to the Axes
    ax11.add_patch(rect)


    axes = [ax1, ax3, ax4, ax5, ax6, ax7, ax8, ax9,ax11]
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
    checkpoints =[
    #    'outputs/gaussian_dataset25_sigma100/srdense/gaussian_mul_wo_circular_mask_sigma100/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
    #    'outputs/dataset_images/hanning_dataset25/srdense/srdense_hanning_blur/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
    #    'outputs/dataset_images/hamming_dataset25/srdense/srdense_hamming_blur/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
    #    'outputs/dataset_images/bicubic_dataset25/srdense/bicubic_up_and_down_factor_2/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
    #    'outputs/dataset_images/mean_blur_dataset25/srdense/srdense_mean_blur/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
    #    'outputs/dataset_images/median_blur_dataset25/srdense/srdense_median_blur/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
    # 'outputs/dataset_images/combine_all_degradation_fix_dataset/srdense/srdense_combine_all_degradation_fix_dataset/checkpoints/z_axis/factor_2/epoch_450_f_2.pth',

        'outputs/gaussian_dataset25_sigma100/srdense/gaussian_mul_wo_circular_mask_sigma100/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
       'outputs/dataset_images_srdense/hanning_dataset25/srdense/srdense_hanning_blur/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
       'outputs/dataset_images_srdense/hamming_dataset25/srdense/srdense_hamming_blur/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
       'outputs/dataset_images_srdense/bicubic_dataset25/srdense/bicubic_up_and_down_factor_2/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
       'outputs/dataset_images_srdense/mean_blur_dataset25/srdense/srdense_mean_blur/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
       'outputs/dataset_images_srdense/median_blur_dataset25/srdense/srdense_median_blur/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
       'outputs/dataset_images_srdense/combine_all_degradation_fix_dataset/srdense/srdense_combine_all_degradation_fix_dataset/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',



    #    'outputs/dataset_images/combine_all_degradation_large_dataset/srdense/srdense_combine_all_degradation_large_dataset/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'
    ] 
    # model_names = ['gaussian model', 'hann model', 'hamm model', 'bicubic model', 'mean model', 'median model', 'combine fix', 'combine large']

    model_names = ['gaussian', 'hanning', 'hamming', 'bicubic ', 'mean', 'median', 'combine']
    dataset_names = ['gaussian', 'hann', 'hamm', 'bicubic', 'mean', 'median']

    '''set device'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device

    psnr = PSNR()
    ssim = SSIM()

    psnr = psnr.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)
    ssim = ssim.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)

    opt.psnr= psnr
    opt.ssim=ssim

    label_path = 'plot_set/label/hr_f4_149_z_176.png'
    # label_path = 'plot_set/label/hr_f4_149_z_112.png'c
    # label_path = 'dataset_images/hamming_dataset25/z_axis/label/test/hr_f4_149_z_200.png'

    # image_path = 'plot_set/gaussian_sigma100/lr_f4_149_z_212_g100.png'


    # image_path = 'plot_set/gaussian_sigma100/lr_f4_149_z_176_g100.png'
    # image_path = 'plot_set/hann/lr_f4_149_2_z_ham176.png'
    # image_path = 'plot_set/hamm/lr_f4_149_2_z_ham212.png'
    image_path = 'plot_set/bicubic/lr_f4_149_2_z_176.png'
    # image_path = 'dataset_images/hanning_dataset25/z_axis/factor_2/test/lr_f4_149_2_z_ham200.png'
    # image_path = 'plot_set/median_blur/lr_f4_149_2_z_b176.png'
    

    label_image =  cv2.imread(label_path)
    label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    degraded_image = cv2.imread(image_path)
    degraded_image = cv2.cvtColor(degraded_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
   

    result_image_list = [label_image, degraded_image]
    result_image_list_key = ['ref', 'input']
    ref_score = [0,0,0,0,0,0,0]
    # input_score = [30.57894590512997, 0.0008751961286179721, 0.7774414480673757, 0.10160665214061737, 0.06954708695411682]
    input_score = [30.928589391368845, 0.0008074972429312766, 0.7577382203683922, 0.10333500057458878, 0.07132922112941742]

    score_list = []
    score_list.append(ref_score)
    score_list.append(input_score)
   
    for index,checkpoint in enumerate(checkpoints): # looping through the model checkpoints

        opt.checkpoint = checkpoint
        model_name = model_names[index]
        print("Evaluating")
        print('checkpoint', checkpoint)
        print("model name ", model_name)
        print("*******************************************************************************************************")

        model = load_model(opt)
        if device != 'cpu':
            num_of_gpus = torch.cuda.device_count()
            model = nn.DataParallel(model,device_ids=[*range(num_of_gpus)])
        model.to(device)
        model.eval()

        ref, degraded, output, scores = predict(model=model,image_path=image_path,device=opt.device,psnr=opt.psnr,ssim=opt.ssim,label_path=label_path)
        result_image_list.append(tensor2image(output))
        result_image_list_key.append(model_name)
        # print("each scores", scores)
        # score_flattened = list(itertools.chain(*scores))
        score_list.append(scores)


    # after getting result from all the model create a subplot
    save_dir='plot_set/subplots/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_name='bicubic'

    save_results(result_image_list=result_image_list,result_image_list_key=result_image_list_key,label_image=label_image,save_dir=save_dir,save_name=save_name, scores=score_list)



