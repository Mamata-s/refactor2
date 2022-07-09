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
import statistics
import json

from utils.load_model import load_model_main
from utils.preprocess import tensor2image, image2tensor,create_dictionary
from utils.prepare_test_set_image import crop_pad_kspace, prepare_image_fourier
from utils.image_quality_assessment import PSNR,SSIM
from utils.general import min_max_normalize, NRMSELoss
import os
import numpy as np

import warnings
warnings.filterwarnings("ignore")
 
def predict_canny_edges(model,image_path,label_path,device,psnr,ssim,mse,nrmse):
    # print(image_path)
    # print(label_path)
    # quit();

    degraded = cv2.imread(image_path)
    degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)
    
    image_blur = cv2.GaussianBlur(degraded, (5,5), 0) 
    lr_edges = cv2.Canny(image = image_blur, threshold1=1, threshold2=20)
    ref = cv2.imread(label_path)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    degraded = degraded.astype(np.float32)/255.
    ref = ref.astype(np.float32)/255.
    lr_edges= lr_edges.astype(np.float32)/255.

    degraded = image2tensor(degraded).unsqueeze(dim=0).to(device)
    ref = image2tensor(ref).unsqueeze(dim=0).to(device)
    input_edges=image2tensor(lr_edges).unsqueeze(dim=0).to(device)
    
    pre_edges = model(input_edges)
    pre = pre_edges+degraded
    pre = pre.clamp(0.,1.)
    
    init_psnr = psnr(degraded, ref).item()
    init_ssim = ssim(degraded, ref).item()
    init_mse = mse(degraded, ref).item()
    init_nrmse = nrmse(ref,degraded).item()

    model_psnr = psnr(pre, ref).item()
    model_ssim = ssim(pre, ref).item()
    model_mse = mse(pre, ref).item()
    model_nrmse = nrmse(ref, pre).item()
    return  {'init_psnr':init_psnr,
            'init_ssim': init_ssim,
            'init_mse': init_mse,
            'init_nrmse': init_nrmse,
            'model_psnr':model_psnr,
            'model_ssim':model_ssim,
            'model_mse': model_mse,
            'model_nrmse':model_nrmse}

# correct version
def predict_downsample_edges(opt,model,image_path,label_path,device,psnr,ssim,mse,nrmse):
    degraded = cv2.imread(image_path)
    degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ref = cv2.imread(label_path)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)

    downsample = crop_pad_kspace(degraded,pad=True,factor= opt.factor+2).astype(np.float32) # working

    degraded = degraded/255.
    ref = ref/255.
    downsample=downsample/255.
   

    degraded = image2tensor(degraded).unsqueeze(dim=0).to(device)
    ref = image2tensor(ref).unsqueeze(dim=0).to(device)
    downsample = image2tensor(downsample).unsqueeze(dim=0).to(device)
  
    
    input_edges = degraded-downsample
    label_edges = ref - degraded

    # perform super-resolution with srcnn
    pre_edges = model(input_edges)
    pre=pre_edges+degraded
    pre = pre.clamp(0., 1.)

    init_psnr = psnr(degraded, ref).item()
    init_ssim = ssim(degraded, ref).item()
    init_mse = mse(degraded, ref).item()
    init_nrmse = nrmse(ref,degraded).item()

    model_psnr = psnr(pre, ref).item()
    model_ssim = ssim(pre, ref).item()
    model_mse = mse(pre, ref).item()
    model_nrmse = nrmse(ref, pre).item()
    return  {'init_psnr':init_psnr,
            'init_ssim': init_ssim,
            'init_mse': init_mse,
            'init_nrmse': init_nrmse,
            'model_psnr':model_psnr,
            'model_ssim':model_ssim,
            'model_mse': model_mse,
            'model_nrmse':model_nrmse}
   

def evaluate_model_edges(opt):
    dir_dict = create_dictionary(opt.image_path,opt.label_path)
    initial ={'psnr':[],'ssim':[],'mse':[],'nrmse':[]}
    model = {'psnr':[],'ssim':[],'mse':[],'nrmse':[]}
    for file in os.listdir(opt.image_path):  
        # perform super-resolution
        image_path = os.path.join(opt.image_path,file)
        label_path = os.path.join(opt.label_path, dir_dict[file])

        if opt.edge_type in ['downsample']:
            output = predict_downsample_edges(opt=opt, model=opt.model,image_path=image_path,label_path=label_path,device=opt.device,psnr=opt.psnr,ssim=opt.ssim,mse = opt.mse,nrmse=opt.nrmse)
        elif opt.edge_type in ['canny']:
            output = predict_canny_edges(model=opt.model,image_path= image_path,label_path=label_path,device=opt.device,psnr=opt.psnr,ssim=opt.ssim,mse = opt.mse,nrmse=opt.nrmse)
        else:
            print(f'edge type {opt.edge_type} not implemented')
            
        # append initial metric
        initial['psnr'].append(output['init_psnr'])
        initial['ssim'].append(output['init_ssim'])
        initial['mse'].append(output['init_mse'])
        initial['nrmse'].append(output['init_nrmse'])
        model['psnr'].append(output['model_psnr'])
        model['ssim'].append(output['model_ssim'])
        model['mse'].append(output['model_mse'])
        model['nrmse'].append(output['model_nrmse'])

    #print min, max std and median
    print('metric for initial')
    for key in initial.keys():
        print('key is',key) 
        print('min : ',min(initial[key])) 
        print('max : ',max(initial[key])) 
        print( ' std :', statistics.pstdev(initial[key]))
        print( ' mean :', statistics.mean(initial[key]))
        print( ' median :', statistics.median(initial[key]))
        print('************************************************************************************')

    print('metric for model')
    for key in model.keys():
        print('key is',key) 
        print('min : ',min(model[key])) 
        print('max : ',max(model[key])) 
        print( ' std :', statistics.pstdev(model[key]))
        print( ' mean :', statistics.mean(model[key]))
        print( ' median :', statistics.median(model[key]))
        print('************************************************************************************')

    with open('model.yaml', 'w') as f:
        json.dump(model, f, indent=2)

    with open('initial.yaml', 'w') as f:
        json.dump(initial, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model demo')
    parser.add_argument('--checkpoint', type=str, metavar='',required=False,help='checkpoint path',
    # default='outputs/resolution_dataset25/srdense/dense_downsample_z_axis25_mask_training_original_f4_805_0.0001/checkpoints/z_axis/factor_4/epoch_300_f_4.pth')
    default = 'outputs/resolution_dataset25/srdense/dense_canny_z_axis25_mask_training_original_f2_155_0.0001/checkpoints/z_axis/factor_2/epoch_150_f_2.pth')
    parser.add_argument('--model_name', type=str, metavar='',help='name of model',default='dense')
    # for loading the val dataset path
    parser.add_argument('--factor', type=int, metavar='',required=False,help='resolution factor',default=2)
    parser.add_argument('--label-path',type=str,required=False,help='path for label images',default='resolution_dataset25/z_axis/label/test')
    # parser.add_argument('--edges',
    #                 help='add the output of model to input images for getting result image', action='store_true')
    parser.add_argument('--edge-type',type=str,default='canny',help='type of edges',required=False,choices=['canny','downsample'])


    opt = parser.parse_args()
  
    opt.image_path = f'resolution_dataset25/z_axis/factor_{opt.factor}/test'

    '''set device'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device

    psnr = PSNR()
    ssim = SSIM()

    psnr = psnr.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)
    ssim = ssim.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)

    opt.psnr= psnr
    opt.ssim=ssim
    opt.mse = nn.MSELoss().to(device=opt.device)
    opt.nrmse = NRMSELoss().to(device=opt.device)

    model = load_model_main(opt)

    if device != 'cpu':
        num_of_gpus = torch.cuda.device_count()
        model = nn.DataParallel(model,device_ids=[*range(num_of_gpus)])

    model.to(device)
    model.eval()

    opt.model = model

    evaluate_model_edges(opt)
    print(opt)
