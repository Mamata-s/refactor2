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
# from models.densenet_new import SRDenseNet
from models.densenet_smchannel import SRDenseNet
import os
import numpy as np
from utils.train_utils import normalize_edges, denormalize_edges

import warnings
warnings.filterwarnings("ignore")
import random


label_path ='test_set/labels/'
# label_path ='resolution_dataset25_small4/z_axis/label/train/'
# label_path ='gaussian_dataset25/z_axis/label/train/'

def load_model(opt):
    checkpoint = torch.load(opt.checkpoint,map_location=torch.device(opt.device))
    model = SRDenseNet()
    state_dict = model.state_dict()
    for n, p in checkpoint['model_state_dict'].items():
        # new_key = n[7:]
        new_key = n
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
    
    return scores

def make_downsample_images(opt):
    factor = opt.factor+opt.factor
    path_images = 'test_set/images/factor_{}'.format(factor)
    if not os.path.exists(path_images):
        os.makedirs(path_images)
    # prepare_image_fourier(label_path,path_images,factor)
    prepare_image_gaussian(label_path,path_images,factor)
    return path_images


def predict_canny_edges(model,path,factor,device,psnr,ssim):
    image_path = path
    # load the degraded and reference images
    path, file = os.path.split(image_path)
    degraded = cv2.imread(image_path)
    degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(degraded, (15,15), 0) 
    lr_edges = cv2.Canny(image = image_blur, threshold1=1, threshold2=20)
    # ref = cv2.imread('test_set/labels/{}'.format(file))
    ref_path = os.path.join(label_path,file)
    ref = cv2.imread(ref_path)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    # label_edges = ref-degraded #not effective while plotting
    
    degraded = degraded.astype(np.float32)/255.
    ref = ref.astype(np.float32)/255.
    lr_edges= lr_edges.astype(np.float32)/255.

    degraded = image2tensor(degraded).unsqueeze(dim=0).to(device)
    ref = image2tensor(ref).unsqueeze(dim=0).to(device)
    input_edges=image2tensor(lr_edges).unsqueeze(dim=0).to(device)
    
    label_edges = ref-degraded 
    pre_edges = model(input_edges)
    pre=pre_edges+degraded

    pre = pre.clamp(0.,1.)

    print('Before normalization')
    print('range of prediction edges',pre_edges.min(),pre_edges.max())
    print('range of input edges',input_edges.min(),input_edges.max())
    print('range of label edges',label_edges.min(),label_edges.max())


    label_edges= min_max_normalize(label_edges)
    pre_edges= min_max_normalize(pre_edges)

    initial_baseline = degraded+input_edges
    
    # image quality calculations
    scores = []
    scores.append(compare_images(degraded, ref,psnr,ssim))
    scores.append(compare_images(initial_baseline, ref,psnr,ssim))
    scores.append(compare_images(pre, ref,psnr,ssim))
    

    # return images and scores
    return tensor2image(ref), tensor2image(degraded), tensor2image(pre), tensor2image(input_edges),tensor2image(pre_edges),tensor2image(label_edges), tensor2image(initial_baseline), scores

# correct version
def predict_downsample_edges(opt,model,path,factor,device,psnr,ssim):

    image_path = path
    print('image_path', image_path)


    # load the degraded and reference images
    path, file = os.path.split(image_path)
    degraded = cv2.imread(image_path)
    degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ref_path = os.path.join(label_path,file)
    ref = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    down_path = os.path.join(opt.downsample_path,file)

    downsample = cv2.imread(down_path)
    downsample = cv2.cvtColor(downsample, cv2.COLOR_BGR2GRAY).astype(np.float32)

    print('lr path', path)
    print('ref' , ref_path)
    print('doensample path', down_path)


    
    degraded = degraded/255.
    ref = ref/255.
    downsample=downsample/255.
   

    degraded = image2tensor(degraded).unsqueeze(dim=0).to(device)
    ref = image2tensor(ref).unsqueeze(dim=0).to(device)
    downsample = image2tensor(downsample).unsqueeze(dim=0).to(device)
  
    
    input_edges = degraded - downsample
    label_edges = ref - degraded

    input_edges1 = normalize_edges(input_edges)

    # perform super-resolution with srcnn
    pre_edges = model(input_edges1)

    pre_edges = denormalize_edges(pre_edges)

    # print('RANGE OF Edges before ', pre_edges.min().item(), pre_edges.max().item())
    # pre_edges = pre_edges*50.
    # print('multiplication')
    pre = pre_edges+degraded
    # print('RANGE OF PREDICTION before clamping or minmax', pre.min().item(), pre.max().item())

    pre = pre.clamp(0.,1.)
    # pre = min_max_normalize(pre)


    initial_baseline = degraded + input_edges1
    initial_baseline = initial_baseline.clamp(0.,1.)

    # print(image_path)
    # print('RANGE OF PREDICTION after', pre.min().item(), pre.max().item())
    print('range of prediction edges',pre_edges.min().item(),pre_edges.max().item())
    print('range of input edges',input_edges.min().item(),input_edges.max().item())
    print('range of label edges',label_edges.min().item(),label_edges.max().item())
    print('********************************************************************************************************************************************')

    pre_edges = min_max_normalize(pre_edges)
    input_edges= min_max_normalize(input_edges)
    label_edges = min_max_normalize(label_edges)


    # image quality calculations
    scores = []
    scores.append(compare_images(degraded, ref,psnr,ssim))
    scores.append(compare_images(initial_baseline, ref,psnr,ssim))
    scores.append(compare_images(pre, ref,psnr,ssim))

    # return images and scores
    return tensor2image(ref), tensor2image(degraded), tensor2image(pre), tensor2image(input_edges),tensor2image(pre_edges),tensor2image(label_edges), tensor2image(initial_baseline) ,scores

def save_results_edges(model,path,opt):
    print('save result for edges training')
    # if opt.edge_type in ['downsample']:
    #     downsample_path = make_downsample_images(opt)
    for file in os.listdir(path):
        
        # perform super-resolution
        if opt.edge_type in ['downsample']:
            # ref, degraded, output, input_edges, output_edges, label_edges, intial_baseline, scores = predict_downsample_edges(model,path='{}/{}'.format(path,file),downsample_path=downsample_path,factor=opt.factor,device=opt.device,psnr=opt.psnr,ssim=opt.ssim)
            ref, degraded, output, input_edges, output_edges, label_edges, intial_baseline, scores = predict_downsample_edges(opt,model,path='{}/{}'.format(path,file),factor=opt.factor,device=opt.device,psnr=opt.psnr,ssim=opt.ssim)
        elif opt.edge_type in ['canny']:
            ref, degraded, output, input_edges, output_edges, label_edges, intial_baseline, scores = predict_canny_edges(model,path='{}/{}'.format(path,file),factor=opt.factor,device=opt.device,psnr=opt.psnr,ssim=opt.ssim)
        else:
            print(f'edge type {opt.edge_type} not implemented')

        # display images as subplots
        fig, axs = plt.subplots(1, 7, figsize=(20, 8))
        axs[0].imshow(ref,cmap='gray')
        axs[0].set_title('Original')

        axs[1].imshow(degraded,cmap='gray')
        axs[1].set_title('Degraded')
        axs[1].set(xlabel = 'PSNR: {:.6f} \nMSE: {:.6f} \nSSIM: {:.6f} \nNRMSE: {:.6f}'.format(scores[0][0], scores[0][1], scores[0][2],scores[0][3]))

        axs[2].imshow(intial_baseline,cmap='gray')
        axs[2].set_title('Initial_baseline')
        axs[2].set(xlabel = 'PSNR: {:.6f} \nMSE: {:.6f}% \nSSIM: {:.6f}% \nNRMSE: {:.6f}'.format(scores[1][0], scores[1][1], scores[1][2],scores[1][3]))

        axs[3].imshow(output,cmap='gray')
        axs[3].set_title(opt.model_name)
        axs[3].set(xlabel = 'PSNR: {:.6f} \nMSE: {:.6f} \nSSIM: {:.6f} \nNRMSE: {:.6f}'.format(scores[2][0], scores[2][1], scores[2][2],scores[2][3]))

        axs[4].imshow(input_edges,cmap='gray')
        axs[4].set_title('Input Edges')

        axs[5].imshow(output_edges,cmap='gray')
        axs[5].set_title('Output Edges')

        axs[6].imshow(label_edges,cmap='gray')
        axs[6].set_title('Label Edges')

        # remove the x and y ticks
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # print('Saving {}'.format(file))
        fig.savefig(opt.plot_dir+'{}.png'.format(os.path.splitext(file)[0])) 

         # display images as subplots
        fig1, axs = plt.subplots(1, 2, figsize=(10, 8))
        axs[0].imshow((ref/255.)-(degraded/255.),cmap='gray')
        axs[0].set_title('Original error (label-input)')

        axs[1].imshow((ref/255.)-(output/255.),cmap='gray')
        axs[1].set_title('Model Error (label -output)')

        # remove the x and y ticks
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        fig1.savefig(opt.plot_dir+'{}_errormap.png'.format(os.path.splitext(file)[0])) 

        cv2.imwrite(opt.preds_dir+'{}.png'.format(os.path.splitext(file)[0]),output)

        cv2.imwrite(opt.pred_edges_dir+'{}.png'.format(os.path.splitext(file)[0]),output_edges)
        cv2.imwrite(opt.input_edges_dir+'{}.png'.format(os.path.splitext(file)[0]),input_edges)
        cv2.imwrite(opt.label_edges_dir+'{}.png'.format(os.path.splitext(file)[0]),label_edges)
        plt.close()





# ****************************************************************************************************************************************


def predict(model,path,factor,device,psnr,ssim):
    
    image_path = path
    # load the degraded and reference images
    path, file = os.path.split(image_path)
    degraded = cv2.imread(image_path)
    degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # ref = cv2.imread('test_set/labels/{}'.format(file))

    ref_path = os.path.join(label_path,file)
    ref = cv2.imread(ref_path)
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
        axs[1].set(xlabel = 'PSNR: {:.6f} \nMSE: {:.6f} \nSSIM: {:.6f}'.format(scores[0][0], scores[0][1], scores[0][2]))
        axs[2].imshow(output,cmap='gray')
        axs[2].set_title(opt.model_name)
        axs[2].set(xlabel = 'PSNR: {:.6f} \nMSE: {:.6f} \nSSIM: {:.6f}'.format(scores[1][0], scores[1][1], scores[1][2]))

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
    parser.add_argument('--checkpoint', type=str, metavar='',required=True,help='checkpoint path')

    parser.add_argument('--model_name', type=str, metavar='',help='name of model',default='dense')
    # for loading the val dataset path
    parser.add_argument('--factor', type=int, metavar='',required=False,help='resolution factor',default=4)

    parser.add_argument('--plot-dir', type=str, metavar='',help='plot save dir',default='test_set/plots_4/')
    parser.add_argument('--preds-dir', type=str, metavar='',help='plot save dir',default='test_set/preds_4/')
    parser.add_argument('--pred-edges-dir', type=str, metavar='',help='plot save dir',default='test_set/preds_edges_4/')
    parser.add_argument('--input-edges-dir', type=str, metavar='',help='plot save dir',default='test_set/input_edges_4/')
    parser.add_argument('--label-edges-dir', type=str, metavar='',help='plot save dir',default='test_set/label_edges_4/')

    parser.add_argument('--edges',
                    help='add the output of model to input images for getting result image', action='store_true')
    parser.add_argument('--edge-type',type=str,default='canny',help='type of edges',required=False,choices=['canny','downsample'])



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

    model = load_model(opt)

    if device != 'cpu':
        num_of_gpus = torch.cuda.device_count()
        model = nn.DataParallel(model,device_ids=[*range(num_of_gpus)])

    model.to(device)

    model.eval()
    path_images = 'test_set/images/factor_{}'.format(opt.factor)
    if not os.path.exists(path_images):
        os.makedirs(path_images)
        # prepare_image_gaussian('test_set/labels',path_images,opt.factor)
        prepare_image_fourier(label_path,path_images,opt.factor)
    if not os.path.exists(opt.plot_dir):
        os.makedirs(opt.plot_dir)
    if not os.path.exists(opt.preds_dir):
        os.makedirs(opt.preds_dir)

    if opt.edges:
        if not os.path.exists(opt.pred_edges_dir):
            os.makedirs(opt.pred_edges_dir)
        if not os.path.exists(opt.input_edges_dir):
            os.makedirs(opt.input_edges_dir)
        if not os.path.exists(opt.label_edges_dir):
            os.makedirs(opt.label_edges_dir)
        if opt.edge_type in ['downsample']:
            opt.downsample_path = make_downsample_images(opt)
            print('downsample path is ', opt.downsample_path)
        save_results_edges(model,path_images,opt)
    
    else:
        save_results(model,path_images,opt)
