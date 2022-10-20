'''
checkpoints to load the models
model_name to load correct model and use their function to load weights from checkpoints
factor: determines on which downsampled factor images model is tested
plot-dir: dir where plots is saved---leave as default
output-dir: dir where outputs are saved --- leave as deault
addition: boolean (not implemented) if True add the pred with images to get the final output of the model
edges: boolean(not implemented) --if True load canny edges as input to model


Make plot for gaussian model where we plot the original kspace pad image as the lr initial image instead of gaussian input image

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


# label_path ='test_set/labels/'
# label_path = 'gaussian_dataset25_mul/z_axis/label/test'
# label_path ='gaussian_test_set/label'


# label_path = 'upsample/combine/hr_25/z_axis'
label_path = 'dataset_images/combined_kspace_gaussian_bicubic/z_axis/label/train'


# label_path = 'gaussian_dataset25_sigma50/z_axis/label/test'

# label_path ='resolution_dataset25_small4/cz_axis/label/train/'
# label_path ='gaussian_dataset25/z_axis/label/train/'


def create_freq_mask(shape,factor=2, lower=False):

  y,x = shape
  mask_lower = np.zeros((y,x), dtype=np.complex_)

  center_y = y//2 #defining the center of image in x and y direction
  center_x = x//2
  startx = center_x-(x//(factor*2))  
  starty = center_y-(y//(factor*2))

  mask_lower[starty:starty+(y//factor),startx:startx+(x//factor)] = 1
  mask_upper = 1-mask_lower
  if lower:
    return mask_lower
  else:
    return mask_upper

def get_high_freq(image,factor=2):
  fshift = np.fft.fftshift(np.fft.fft2(image))
  mask = create_freq_mask(fshift.shape,factor=factor, lower=False)
  high_freq = fshift*mask
  return np.abs(np.fft.ifft2(np.fft.ifftshift(high_freq)))


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

def make_downsample_images(opt):
    factor = opt.factor+opt.factor
    path_images = 'test_set/images/factor_{}'.format(factor)
    if not os.path.exists(path_images):
        os.makedirs(path_images)
    prepare_image_fourier(label_path,path_images,factor)
    # prepare_image_gaussian(label_path,path_images,factor)
    return path_images


def predict_canny_edges(model,image_path,label_path,device,psnr,ssim):
   
    degraded = cv2.imread(image_path)
    degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)
    # image_blur = cv2.GaussianBlur(degraded, (15,15), 0) 
    # lr_edges = cv2.Canny(image = image_blur, threshold1=1, threshold2=20)
    
    lr_edges= get_high_freq(degraded,factor=4).astype(np.float32)

    ref = cv2.imread(label_path)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    
    degraded = degraded.astype(np.float32)/255.
    ref = ref.astype(np.float32)/255.
    lr_edges= lr_edges.astype(np.float32)/255.

    degraded = image2tensor(degraded).unsqueeze(dim=0).to(device)
    ref = image2tensor(ref).unsqueeze(dim=0).to(device)
    input_edges=image2tensor(lr_edges).unsqueeze(dim=0).to(device)
    
    label_edges = ref-degraded 
    pre_edges, pre = model(input_edges, degraded)
    # pre=pre_edges+degraded

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
    return ref, degraded, pre, input_edges,pre_edges,label_edges, initial_baseline, scores

# correct version
def predict_downsample_edges(opt,model,path,factor,device,psnr,ssim, dictionary=None):

    image_path = path
    print('image_path', image_path)


    # load the degraded and reference images
    path, file = os.path.split(image_path)
    degraded = cv2.imread(image_path)
    degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    if dictionary:
        label_name = dictionary[file]
    else:
        label_name = file

    ref_path = os.path.join(label_path,label_name)
    ref = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    down_path = os.path.join(opt.downsample_path,file)

    downsample = cv2.imread(down_path)
    downsample = cv2.cvtColor(downsample, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # print('lr path', path)
    # print('ref' , ref_path)
    # print('downsample path', down_path)
    # quit();


    
    degraded = degraded/255.
    ref = ref/255.
    downsample=downsample/255.
   

    degraded = image2tensor(degraded).unsqueeze(dim=0).to(device)
    ref = image2tensor(ref).unsqueeze(dim=0).to(device)
    downsample = image2tensor(downsample).unsqueeze(dim=0).to(device)
  
    
    input_edges = degraded - downsample
    label_edges = ref - degraded

    # input_edges1 = normalize_edges(input_edges)
    input_edges1 =input_edges

    # perform super-resolution with srcnn
    pre_edges = model(input_edges1)

    # pre_edges = denormalize_edges(pre_edges)

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
    return ref, degraded, pre,input_edges,pre_edges,label_edges, initial_baseline ,scores

def save_results_edges(model,path,opt):
    print('save result for edges training')
    # if opt.edge_type in ['downsample']:
    #     downsample_path = make_downsample_images(opt)
    fnsize=17
    for file in opt.dictionary:
        image_path = os.path.join(opt.image_path,opt.dictionary[file])
        label_path = os.path.join(opt.label_path,file)
        
        # perform super-resolution
        if opt.edge_type in ['downsample']:
            ref, degraded, output, input_edges, output_edges, label_edges, intial_baseline, scores = predict_downsample_edges(opt,model,path='{}/{}'.format(path,file),factor=opt.factor,device=opt.device,psnr=opt.psnr,ssim=opt.ssim,dictionary= opt.dictionary)
        elif opt.edge_type in ['canny']:
            print("Calling this canny function")
            ref, degraded, output, input_edges, output_edges, label_edges, intial_baseline, scores = predict_canny_edges(model,image_path=image_path,label_path=label_path,device=opt.device,psnr=opt.psnr,ssim=opt.ssim)
        else:
            print(f'edge type {opt.edge_type} not implemented')

        # display images as subplots
        fig, axs = plt.subplots(1, 7, figsize=(20, 8))
        axs[0].imshow(tensor2image(ref),cmap='gray')
        axs[0].set_title('Original',fontsize=fnsize)

        axs[1].imshow(tensor2image(degraded),cmap='gray')
        axs[1].set_title('Degraded',fontsize=fnsize)
        axs[1].set(xlabel = 'PSNR: {:.6f} \nMSE: {:.6f} \nSSIM: {:.6f} \nNRMSE: {:.6f} \nHFEN: {:.6f}'.format(scores[0][0], scores[0][1], scores[0][2],scores[0][3],scores[0][4]))

        axs[2].imshow(tensor2image(intial_baseline),cmap='gray')
        axs[2].set_title('Initial_baseline',fontsize=fnsize)
        axs[2].set(xlabel = 'PSNR: {:.6f} \nMSE: {:.6f}% \nSSIM: {:.6f}% \nNRMSE: {:.6f} \nHFEN: {:.6f}'.format(scores[1][0], scores[1][1], scores[1][2],scores[1][3],scores[1][4]))

        axs[3].imshow(tensor2image(output),cmap='gray')
        axs[3].set_title(opt.model_name,fontsize=fnsize)
        axs[3].set(xlabel = 'PSNR: {:.6f} \nMSE: {:.6f} \nSSIM: {:.6f} \nNRMSE: {:.6f} \nHFEN: {:.6f}'.format(scores[2][0], scores[2][1], scores[2][2],scores[2][3],scores[2][4]))

        axs[4].imshow(tensor2image(input_edges),cmap='gray')
        axs[4].set_title('Input Edges',fontsize=fnsize)

        axs[5].imshow(tensor2image(output_edges),cmap='gray')
        axs[5].set_title('Output Edges',fontsize=fnsize)

        axs[6].imshow(tensor2image(label_edges),cmap='gray')
        axs[6].set_title('Label Edges',fontsize=fnsize)

        # remove the x and y ticks
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # print('Saving {}'.format(file))
        fig.savefig(opt.plot_dir+'{}.png'.format(os.path.splitext(file)[0])) 
        print('saving plots')

        cv2.imwrite(opt.preds_dir+'{}.png'.format(os.path.splitext(file)[0]),tensor2image(output))

        # cv2.imwrite(opt.pred_edges_dir+'{}.png'.format(os.path.splitext(file)[0]),tensor2image(output_edges))
        # cv2.imwrite(opt.input_edges_dir+'{}.png'.format(os.path.splitext(file)[0]),tensor2image(input_edges))
        # cv2.imwrite(opt.label_edges_dir+'{}.png'.format(os.path.splitext(file)[0]),tensor2image(label_edges))


        ref = ref.squeeze().detach().cpu().numpy()
        degraded = degraded.squeeze().detach().cpu().numpy()
        output = output.squeeze().detach().cpu().numpy()
        label_edges = label_edges.squeeze().detach().cpu().numpy()
        input_edges = input_edges.squeeze().detach().cpu().numpy()
        output_edges = output_edges.squeeze().detach().cpu().numpy()

         # display errormap as subplots
        fig_error_map, axs = plt.subplots(1, 4, figsize=(25, 8))
        error_i= label_edges-input_edges
        error_o = label_edges-output_edges
        lmin= label_edges.min()
        lmax= label_edges.max()

        axs[0].imshow(error_i,cmap='gray')
        axs[0].set_title('Original error (label edg-inputedg)',fontsize=fnsize)

        axs[1].imshow(error_o,cmap='gray')
        axs[1].set_title('Model Error (labeledg -outputedg)',fontsize=fnsize)

        axs[2].imshow(error_i,cmap='gray',vmin=lmin,vmax=lmax)
        axs[2].set_title('Original error (label edg-input edg) \n with label_edg_min_max',fontsize=fnsize)

        axs[3].imshow(error_o,cmap='gray',vmin=lmin,vmax=lmax)
        axs[3].set_title('Model Error (label edg -output edg) \n with label_edg_min_max',fontsize=fnsize)

        # remove the x and y ticks
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        # fig_error_map.savefig(opt.plot_dir+'{}_errormap.png'.format(os.path.splitext(file)[0])) 

          # display edges as subplots
        fig_edges, axs = plt.subplots(1, 5, figsize=(25, 8))    
        axs[0].imshow(label_edges,cmap='gray')
        axs[0].set_title('label edges',fontsize=fnsize)

        axs[1].imshow(input_edges,cmap='gray')
        axs[1].set_title('input edge',fontsize=fnsize)

        axs[2].imshow(output_edges,cmap='gray')
        axs[2].set_title('output edges',fontsize=fnsize)

        axs[3].imshow(input_edges,cmap='gray',vmin=lmin,vmax=lmax)
        axs[3].set_title('input edge \n with lbledge vmin_vmax',fontsize=fnsize)

        axs[4].imshow(output_edges,cmap='gray',vmin=lmin,vmax=lmax)
        axs[4].set_title('output edges \n with lbledge vmin_vmax',fontsize=fnsize)
        # remove the x and y ticks
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])


        # fig_edges.savefig(opt.plot_dir+'{}_edges.png'.format(os.path.splitext(file)[0])) 


        plt.close()





# ****************************************************************************************************************************************


# def predict(model,path,factor,device,psnr,ssim,dictionary):
    
#     image_path = path
#     # load the degraded and reference images
#     path, file = os.path.split(image_path)
#     degraded = cv2.imread(image_path)
#     degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY).astype(np.float32)
#     # ref = cv2.imread('test_set/labels/{}'.format(file))

#     # read dictionary for label path
#     label_name = dictionary[file]
#     ref_path = os.path.join(label_path,label_name)

#     print("Image path", image_path)
#     print("Label path", ref_path)
#     print("**********************************************************************************************************************")


#     ref = cv2.imread(ref_path)
#     ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)

#     degraded = degraded/255.
#     ref = ref/255.
#     degraded = image2tensor(degraded).unsqueeze(dim=0).to(device)
#     ref = image2tensor(ref).unsqueeze(dim=0).to(device)
    
#     # perform super-resolution with srcnn
#     pre = model(degraded)
    
#     # image quality calculations
#     scores = []
#     scores.append(compare_images(degraded, ref,psnr,ssim))
#     scores.append(compare_images(pre, ref,psnr,ssim))
    

#     # return images and scores
#     return ref, degraded, pre, scores

# def save_results(model,opt):
#     for file in opt.dictionary:
#         print(file)
#         print("*************************************")
        
#         # perform super-resolution
#         ref, degraded, output, scores = predict(model=model,path='{}/{}'.format(opt.image_path,file),factor=opt.factor,device=opt.device,psnr=opt.psnr,ssim=opt.ssim,dictionary=opt.dictionary)
        
#         # display images as subplots
#         fig, axs = plt.subplots(1, 3, figsize=(20, 8))
#         axs[0].imshow(tensor2image(ref),cmap='gray')
#         axs[0].set_title('Original')
#         axs[1].imshow(tensor2image(degraded),cmap='gray')
#         axs[1].set_title('Degraded')
#         axs[1].set(xlabel = 'PSNR: {:.6f} \nMSE: {:.6f} \nSSIM: {:.6f}'.format(scores[0][0], scores[0][1], scores[0][2]))
#         axs[2].imshow(tensor2image(output),cmap='gray')
#         axs[2].set_title(opt.model_name)
#         axs[2].set(xlabel = 'PSNR: {:.6f} \nMSE: {:.6f} \nSSIM: {:.6f}'.format(scores[1][0], scores[1][1], scores[1][2]))

#         # remove the x and y ticks
#         for ax in axs:
#             ax.set_xticks([])
#             ax.set_yticks([])
        
#         print('Saving {}'.format(file))
#         print(fig.savefig(opt.plot_dir+'{}.png'.format(os.path.splitext(file)[0])) )
#         print(cv2.imwrite(opt.preds_dir+'{}.png'.format(os.path.splitext(file)[0]),tensor2image(output)))
#         plt.close()
    

# for dictionary having hr image name as key
def predict(model,image_path,device,psnr,ssim,label_path,original_degraded_path):
    
    degraded = cv2.imread(image_path)
    degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # ref = cv2.imread('test_set/labels/{}'.format(file))

    # original_degraded = cv2.imread(original_degraded_path)
    # original_degraded = cv2.cvtColor(original_degraded, cv2.COLOR_BGR2GRAY).astype(np.float32)

    ref = cv2.imread(label_path)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)

    print("Image path", image_path)
    print("Label path", label_path)
    print("**********************************************************************************************************************")

    degraded = degraded/255.
    ref = ref/255.
    # original_degraded= original_degraded/255.
    degraded = image2tensor(degraded).unsqueeze(dim=0).to(device)
    # original_degraded=image2tensor(original_degraded).unsqueeze(dim=0).to(device)
    ref = image2tensor(ref).unsqueeze(dim=0).to(device)
    
    # perform super-resolution with srcnn
    pre = model(degraded)
    
    # image quality calculations
    scores = []
    scores.append(compare_images(degraded, ref,psnr,ssim))
    scores.append(compare_images(pre, ref,psnr,ssim))
    # scores.append(compare_images(original_degraded, ref,psnr,ssim))
    

    # return images and scores
    # return ref, degraded, pre, original_degraded, scores
    return ref, degraded, pre, scores

def save_results(model,opt):
    for file in opt.dictionary:
        # label_name = file
        # image_name = opt.dictionary[file]

        image_name = file
        label_name = opt.dictionary[file]

        image_path = os.path.join(opt.image_path,image_name)
        label_path = os.path.join(opt.label_path,label_name)

        
        original_degraded_path = os.path.join('upsample/combine/upsample_image',image_name)

        print("Image name", image_path)
        print("label name", label_path)
        print("********************************************************************************")

        # perform super-resolution
        ref, degraded, output,scores = predict(model=model,image_path=image_path,device=opt.device,psnr=opt.psnr,ssim=opt.ssim,label_path=label_path, original_degraded_path= original_degraded_path)
        
        # # display images as subplots
        # fig, axs = plt.subplots(1, 4, figsize=(20, 8))
        # axs[0].imshow(tensor2image(ref),cmap='gray')
        # axs[0].set_title('Label')

        # axs[1].imshow(tensor2image(output),cmap='gray')
        # axs[1].set_title(opt.model_name)
        # axs[1].set(xlabel = 'PSNR: {:.6f} \nMSE: {:.6f} \nSSIM: {:.6f}'.format(scores[1][0], scores[1][1], scores[1][2]))

        # axs[2].imshow(tensor2image(original_degraded),cmap='gray')
        # axs[2].set_title('Original_degraded')
        # axs[2].set(xlabel = 'PSNR: {:.6f} \nMSE: {:.6f} \nSSIM: {:.6f}'.format(scores[2][0], scores[2][1], scores[2][2]))

        # axs[3].imshow(tensor2image(degraded),cmap='gray')
        # axs[3].set_title('Degraded_gaussian')
        # axs[3].set(xlabel = 'PSNR: {:.6f} \nMSE: {:.6f} \nSSIM: {:.6f}'.format(scores[0][0], scores[0][1], scores[0][2]))
        
        # # remove the x and y ticks
        # for ax in axs:
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        
        # print('Saving {}'.format(file))
        # print(fig.savefig(opt.plot_dir+'{}.png'.format(os.path.splitext(file)[0])) )
        print(cv2.imwrite(opt.preds_dir+'{}.png'.format(os.path.splitext(file)[0]),tensor2image(output)))
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model demo')
    parser.add_argument('--checkpoint', type=str, metavar='',required=False,help='checkpoint path', default='outputs/gaussian_dataset25_mul_wo_circular_mask/srdense/gaussian_mul_wo_circular_mask/checkpoints/z_axis/factor_2/epoch_1000_f_2.pth')

    parser.add_argument('--model_name', type=str, metavar='',help='name of model',default='dense')
    # for loading the val dataset pathc
    parser.add_argument('--factor', type=int, metavar='',required=False,help='resolution factor',default=2)

    parser.add_argument('--plot-dir', type=str, metavar='',help='plot save dir',default='test_set/plots_2/')
    parser.add_argument('--preds-dir', type=str, metavar='',help='plot save dir',default='test_set/preds_2/')
    parser.add_argument('--pred-edges-dir', type=str, metavar='',help='plot save dir',default='test_set/preds_edges_4/')
    parser.add_argument('--input-edges-dir', type=str, metavar='',help='plot save dir',default='test_set/input_edges_4/')
    parser.add_argument('--label-edges-dir', type=str, metavar='',help='plot save dir',default='test_set/label_edges_4/')
    parser.add_argument('--dictionary-path', type=str,required=False, metavar='',help='loading the annotation path',default='gaussian_dataset25_sigma75/test_annotation.pkl')
    parser.add_argument('--image-path', type=str, metavar='',required=False,help='gaussian mult image',default='upsample/upsample_50/z_axis')
    parser.add_argument('--downsample-path', type=str, metavar='',help='gaussian mult  downsample image',default='gaussian_dataset25_mul/z_axis/factor_8/test')

    parser.add_argument('--edges',
                    help='add the output of model to input images for getting result image', action='store_true')
    parser.add_argument('--edge-type',type=str,default='canny',help='type of edges',required=False,choices=['canny','downsample'])

    opt = parser.parse_args()

    # print(opt)
    # quit();
    print(opt.image_path)
  

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

    model = load_model(opt)

    opt.dictionary = read_dictionary(opt.dictionary_path)
    if device != 'cpu':
        num_of_gpus = torch.cuda.device_count()
        model = nn.DataParallel(model,device_ids=[*range(num_of_gpus)])

    model.to(device)

    model.eval()
    path_images = 'test_set/images/factor_{}'.format(opt.factor)
    if not os.path.exists(path_images):
        os.makedirs(path_images)
        # prepare_image_gaussian('test_set/labels',path_images,opt.factor)
        # prepare_image_fourier(label_path,path_images,opt.factor)
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
            if not opt.downsample_path:
                opt.downsample_path = make_downsample_images(opt)
                print('downsample path is ', opt.downsample_path)
            if not opt.image_path:
                opt.image_path = path_images
        save_results_edges(model,opt.image_path,opt)
    
    else:
        print("testing on images")
        print(opt)
        opt.dictionary = read_dictionary(opt.dictionary_path)
        # print(opt.dictionary);quit();
        save_results(model,opt)
