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
from utils.prepare_test_set_image import prepare_image_fourier
from utils.image_quality_assessment import PSNR,SSIM
import os
import numpy as np

import warnings
warnings.filterwarnings("ignore")
 



# define function that combines all three image quality metrics
def compare_images(target, ref,psnr,ssim):
    mse = nn.MSELoss()
    scores = []
    scores.append(psnr(target, ref).item())
    scores.append(mse(target, ref).item())
    scores.append(ssim(target, ref).item())
    
    return scores

def predict(model,path,factor,device,psnr,ssim):
    
    image_path = path
    # load the degraded and reference images
    path, file = os.path.split(image_path)
    degraded = cv2.imread(image_path)
    degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ref = cv2.imread('test_set/labels/{}'.format(file))
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)

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
    return tensor2image(ref), tensor2image(degraded), tensor2image(pre), scores
    
def save_results(model,path,opt):
    for file in os.listdir(path):
        
        # perform super-resolution
        ref, degraded, output, scores = predict(model,path='{}/{}'.format(path,file),factor=opt.factor,device=opt.device,psnr=opt.psnr,ssim=opt.ssim)
        
        # display images as subplots
        fig, axs = plt.subplots(1, 3, figsize=(20, 8))
        axs[0].imshow(ref,cmap='gray')
        axs[0].set_title('Original')
        axs[1].imshow(degraded,cmap='gray')
        axs[1].set_title('Degraded')
        axs[1].set(xlabel = 'PSNR: {:.3f}% \nMSE: {:.3f}% \nSSIM: {:.3f}%'.format(scores[0][0], scores[0][1], scores[0][2]))
        axs[2].imshow(output,cmap='gray')
        axs[2].set_title(opt.model_name)
        axs[2].set(xlabel = 'PSNR: {:.3f}% \nMSE: {:.3f}% \nSSIM: {:.3f}%'.format(scores[1][0], scores[1][1], scores[1][2]))

        # remove the x and y ticks
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        
        print('Saving {}'.format(file))
        print(fig.savefig(opt.plot_dir+'{}.png'.format(os.path.splitext(file)[0])) )
        print(cv2.imwrite(opt.preds_dir+'{}.png'.format(os.path.splitext(file)[0]),output))
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model demo')
    parser.add_argument('--checkpoint', type=str, metavar='',required=False,help='checkpoint path',
    default='outputs1/resolution_dataset50/srdense/srdense_kaiming_init/checkpoints/z_axis/factor_2/epoch_500_f_2.pth')

    parser.add_argument('--model_name', type=str, metavar='',help='name of model',default='dense')
    # for loading the val dataset path
    parser.add_argument('--factor', type=int, metavar='',required=False,help='resolution factor',default=4)
    parser.add_argument('--addition',
                    help='add the output of model to input images for getting result image', action='store_true')
    parser.add_argument('--edges',
                    help='add the output of model to input images for getting result image', action='store_true')

    parser.add_argument('--plot-dir', type=str, metavar='',help='plot save dir',default='test_set/plots_4/')
    parser.add_argument('--preds-dir', type=str, metavar='',help='plot save dir',default='test_set/preds_4/')

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

    model = load_model_main(opt)

    if device != 'cpu':
        num_of_gpus = torch.cuda.device_count()
        model = nn.DataParallel(model,device_ids=[*range(num_of_gpus)])

    model.to(device)

    model.eval()
    path_images = 'test_set/images/factor_{}'.format(opt.factor)
    if not os.path.exists(path_images):
        os.makedirs(path_images)
        prepare_image_fourier('test_set/labels',path_images,opt.factor)
    if not os.path.exists(opt.plot_dir):
        os.makedirs(opt.plot_dir)
    if not os.path.exists(opt.preds_dir):
        os.makedirs(opt.preds_dir)

    save_results(model,path_images,opt)
