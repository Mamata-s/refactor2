import argparse
from utils.load_model import load_model_3d_main
from utils.general import min_max_normalize
from utils.image_quality_assessment import PSNR,SSIM
import torch
import numpy as np
import torch.nn as nn
from utils.preprocess import tensor2image, image2tensor
import matplotlib.pyplot as plt
import os
import cv2

# define function that combines all three image quality metrics
def compare_images(target, ref,psnr,ssim):
    mse = nn.MSELoss()
    scores = []
    scores.append(psnr(target, ref).item())
    scores.append(mse(target, ref).item())
    scores.append(ssim(target, ref).item())
    
    return scores


def save_results(degraded,ref,pre,opt,filename):
        
    # image quality calculations
    scores = []
    scores.append(compare_images(degraded, ref,psnr,ssim))
    scores.append(compare_images(pre, ref,psnr,ssim))

    ref=tensor2image(ref)
    degraded = tensor2image(degraded)
    pre = tensor2image(pre)

    # ref = ref.cpu()
    # degraded = degraded.cpu()
    # pre = pre.cpu()

    # display images as subplots
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))
    axs[0].imshow(ref,cmap='gray')
    axs[0].set_title('Original')
    axs[1].imshow(degraded,cmap='gray')
    axs[1].set_title('Degraded')
    axs[1].set(xlabel = 'PSNR: {:.3f}% \nMSE: {:.3f}% \nSSIM: {:.3f}%'.format(scores[0][0], scores[0][1], scores[0][2]))
    axs[2].imshow(pre,cmap='gray')
    axs[2].set_title(opt.model_name)
    axs[2].set(xlabel = 'PSNR: {:.3f}% \nMSE: {:.3f}% \nSSIM: {:.3f}%'.format(scores[1][0], scores[1][1], scores[1][2]))

    # remove the x and y ticks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    
    print(fig.savefig(opt.plot_dir+'{}.png'.format(filename)) )
    print(cv2.imwrite(opt.output_dir+'{}.png'.format(filename),pre))
    print(cv2.imwrite(opt.input_dir+'{}.png'.format(filename),degraded))
    print(cv2.imwrite(opt.label_dir+'{}.png'.format(filename),ref))
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3d net demo')
    parser.add_argument('--checkpoint',type=str,required=False,default='outputs/3d_dataset_50/srdense/3DSRDENSE_GR7NB5NL5S4_KAI_NE805_BS16_LR0.001_Z_F2_ADAM_MSE/checkpoints/patch/patch-64/factor_2/epoch_800_f_2.pth',help='path of model checkpoint')
    parser.add_argument('--model-name',type=str,required=False,default='dense',help='name of model to load')
    parser.add_argument('--lr-dir',type=str,required=False,default='3d_dataset50/factor_2/val/f5_153_input_p0.npy',help='path of degraded array')
    parser.add_argument('--hr-dir',type=str,required=False,default='3d_dataset50/label/val/f5_153_label_p0.npy',help='path of hr array')
    parser.add_argument('--plot-dir',type=str,required=False,default='test_3d/plots/',help='path to save the result plots')
    parser.add_argument('--input-dir',type=str,required=False,default='test_3d/inputs/',help='path to save the degraded images')
    parser.add_argument('--label-dir',type=str,required=False,default='test_3d/labels/',help='path to save the label images')
    parser.add_argument('--output-dir',type=str,required=False,default='test_3d/outputs/',help='path to save the output images')
    opt = parser.parse_args()


    '''set device'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device


    psnr = PSNR()
    ssim = SSIM()

    psnr = psnr.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)
    ssim = ssim.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)

    opt.psnr= psnr
    opt.ssim=ssim

    if not os.path.exists(opt.input_dir):
        os.makedirs(opt.input_dir)
    if not os.path.exists(opt.label_dir):
        os.makedirs(opt.label_dir)
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    if not os.path.exists(opt.plot_dir):
        os.makedirs(opt.plot_dir)
   
    # load model
    model = load_model_3d_main(opt)

    #load data array lr and hr
    lr_arr = np.load(opt.lr_dir)
    hr_arr = np.load(opt.hr_dir)

    image = torch.from_numpy(min_max_normalize(lr_arr)).unsqueeze(0).unsqueeze(0).type(torch.cuda.FloatTensor)
    label = torch.from_numpy(min_max_normalize(hr_arr)).unsqueeze(0).unsqueeze(0).type(torch.cuda.FloatTensor)

    #apply model
    output = model(image)

    # print(image.shape)
    # print(label.shape)
    # print(output.shape)
    # print('input', image.max(),image.min())
    # print('label',label.max(),label.min())
    # print('output',output.min(),output.max())

    # image = image.squeeze(dim=0)
    # output = output.squeeze(dim=0)
    # label = label.squeeze(dim=0)

    print(image.shape)
    print(label.shape)
    print(output.shape)

    x_indexes = [1,2]
    y_indexes = [4,7]
    z_indexes = [24,25]

    for x in x_indexes:
        img = image[:,:,x,:,:]  #batch,channel,x,y,z
        lbl = label[:,:,x,:,:]
        out = output[:,:,x,:,:]
        filename = 'x_{}'.format(x)
        save_results(img,lbl,out,opt,filename)
    
    for y in y_indexes:
        img = image[:,:,:,y,:]  #batch,channel,x,y,z
        lbl = label[:,:,:,y,:]
        out = output[:,:,:,y,:]
        filename = 'y_{}'.format(y)
        save_results(img,lbl,out,opt,filename)

    for z in z_indexes:
        img = image[:,:,:,:,z]  #batch,channel,x,y,z
        lbl = label[:,:,:,:,z]
        out = output[:,:,:,:,z]
        filename = 'z_{}'.format(z)
        save_results(img,lbl,out,opt,filename)


    # 

