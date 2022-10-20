'''
checkpoints to load the models
model_name to load correct model and use their function to load weights from checkpoints
factor: determines on which downsampled factor images model is tested
plot-dir: dir where plots is saved---leave as default
output-dir: dir where outputs are saved --- leave as deault
addition: boolean (not implemented) if True add the pred with images to get the final output of the model

THIS FILE IS CREATED TO EVALUATE GAUSSIAN IMAGE TRAINED MODEL WITH KSPACE PADDED, KSPACE PADDED AND GAUSSIAN DOWSAMPLE 50 MICRON IMAGES FROM ALL SUBJECTS

'''

from venv import create
import torch
import torch.nn as nn
import  cv2
import matplotlib.pyplot as plt
import argparse
import statistics
import json
import numpy as np


# from models.densenet_new import SRDenseNet

from models.densenet_smchannel import SRDenseNet
from utils.preprocess import tensor2image, image2tensor,create_dictionary
from utils.prepare_test_set_image import crop_pad_kspace, prepare_image_fourier
from utils.image_quality_assessment import PSNR,SSIM
from utils.general import min_max_normalize, NRMSELoss
from utils.train_utils import read_dictionary
import os
import numpy as np
from utils.preprocess import hfen_error

import warnings
warnings.filterwarnings("ignore")



def load_model(checkpoint_path,device):
    checkpoint = torch.load(checkpoint_path,map_location=torch.device(device))
    growth_rate = checkpoint['growth_rate']
    num_blocks = checkpoint['num_blocks']
    num_layers =checkpoint['num_layers']
    model = SRDenseNet(growth_rate=growth_rate,num_blocks=num_blocks,num_layers=num_layers)
    state_dict = model.state_dict()
    for n, p in checkpoint['model_state_dict'].items():
        new_key = n[7:]
        # new_key = n
        if new_key in state_dict.keys():
            state_dict[new_key].copy_(p)
        else:
            raise KeyError(new_key)
    return model
 


def predict_model(model,image_path,label_path,device,psnr,ssim,mse,nrmse):

    # print("image path",image_path)
    # print("label path",label_path)
    # print("**********************************************************************************************")

    degraded = cv2.imread(image_path)
    degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ref = cv2.imread(label_path)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)

    degraded = degraded/255.
    ref = ref/255.   

    degraded = image2tensor(degraded).unsqueeze(dim=0).to(device)
    ref = image2tensor(ref).unsqueeze(dim=0).to(device)  

    pre = model(degraded)

    pre = pre.clamp(0.,1.)

    init_psnr = psnr(degraded, ref).item()
    init_ssim = ssim(degraded, ref).item()
    init_mse = mse(degraded, ref).item()
    init_nrmse = nrmse(ref,degraded).item()
    

    model_psnr = psnr(pre, ref).item()
    model_ssim = ssim(pre, ref).item()
    model_mse = mse(pre, ref).item()
    model_nrmse = nrmse(ref, pre).item()


    ref_arr = (ref.squeeze().detach().cpu().numpy()) *255.
    degraded_arr = (degraded.squeeze().detach().cpu().numpy())*255.
    pre_arr = (pre.squeeze().detach().cpu().numpy())*255.
    init_hfen = hfen_error(ref_arr,degraded_arr).astype(np.float16).item()
    model_hfen = hfen_error(ref_arr,pre_arr).astype(np.float16).item()

    return  {'init_psnr':init_psnr,
            'init_ssim': init_ssim,
            'init_mse': init_mse,
            'init_nrmse': init_nrmse,
            'init_hfen': init_hfen,
            'model_psnr':model_psnr,
            'model_ssim':model_ssim,
            'model_mse': model_mse,
            'model_nrmse':model_nrmse,
            'model_hfen': model_hfen}

def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

def create_each_plot(initial,model,save_name="ssim",y_min=None,y_max=None, ticks=None):
    # initial_pad = [initial_50,initial_75,initial_100,initial_125,initial_150]
    #....

    title = save_name.upper()
    plt.figure()

    bpl = plt.boxplot(initial, positions=np.array(range(len(initial)))*2.0-0.4, sym='', widths=0.4)
    bpr = plt.boxplot(model, positions=np.array(range(len(model)))*2.0+0.1, sym='', widths=0.4)

    set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
    set_box_color(bpr, '#2C7BB6')


    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#D7191C', label='Initial')
    plt.plot([], c='#2C7BB6', label='Model')
    plt.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.title(title+' PLOT')

    plt.show()
    plt.savefig(save_name+'.png')


def plot_boxplot(initial_list, model_list):

    keys = ['psnr', 'ssim', 'mse','nrmse', 'hfen']
    names = ['kspace', 'bicubic', 'gaussian']
    # initial_pad_list list of dictionary , each dictionary is the metric value for each sigma, each dictionary having keys and values for each metric

    ylim_min = [24, 0.6, 0.0001, 0.05, 0.0]
    ylim_max = [37, 0.98, 0.0025, 0.2, 0.2]

    for i,key in enumerate(keys):  # for each metric
        initial = []
        model = []
        for index,element in enumerate(initial_list): # for each sigmas
            print("Index Value", index)
            initial.append(initial_list[index][key])
            model.append(model_list[index][key])
        create_each_plot(initial=initial, model=model,save_name=key,y_min=ylim_min[i],y_max=ylim_max[i],ticks=names)


# new function for evaluating upsample 50 micron as dictionary structure is different
def evaluate_model(opt):
    
    initial ={'psnr':[],'ssim':[],'mse':[],'nrmse':[],'hfen':[]}
    model = {'psnr':[],'ssim':[],'mse':[],'nrmse':[],'hfen':[]}

    for key in opt.dictionary: 
        label_name = opt.dictionary[key]
        image_path = os.path.join(opt.image_path,key)
        label_path = os.path.join(opt.label_path, label_name)

        output = predict_model(model=opt.model,image_path=image_path,label_path=label_path,device=opt.device,psnr=opt.psnr,ssim=opt.ssim,mse = opt.mse,nrmse=opt.nrmse)
            
        # append initial metric
        initial['psnr'].append(output['init_psnr'])
        initial['ssim'].append(output['init_ssim'])
        initial['mse'].append(output['init_mse'])
        initial['nrmse'].append(output['init_nrmse'])
        initial['hfen'].append(output['init_hfen'])
        model['psnr'].append(output['model_psnr'])
        model['ssim'].append(output['model_ssim'])
        model['mse'].append(output['model_mse'])
        model['nrmse'].append(output['model_nrmse'])
        model['hfen'].append(output['model_hfen'])

    #print min, max std and median
    print('metric for initial')
    for key in initial.keys():
        print('key is',key) 
        print('min : ',min(initial[key]))
        print('max : ',max(initial[key]))  
        if key == 'hfen':
            pass
        else:
            print( ' std :', statistics.pstdev(initial[key]))
        print( ' mean :', statistics.mean(initial[key]))
        print( ' median :', statistics.median(initial[key]))
        print('************************************************************************************')

    print('metric for model')
    for key in model.keys():
        print('key is',key) 
        print('min : ',min(model[key])) 
        print('max : ',max(model[key])) 
        if key == 'hfen':
            pass
        else:
            print( ' std :', statistics.pstdev(initial[key]))
        print( ' mean :', statistics.mean(model[key]))
        print( ' median :', statistics.median(model[key]))
        print('************************************************************************************')

    # with open(opt.model_name+'.yaml', 'w') as f:
    #     json.dump(model, f, indent=2)

    # with open(opt.initial_name+'.yaml', 'w') as f:
    #     json.dump(initial, f, indent=2)

    return initial, model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='model demo')
    parser.add_argument('--checkpoint', type=str, metavar='',required=False,help='checkpoint path')
    opt = parser.parse_args()

    sigma_value = [50,75,100,125]
    dataset_names = ['kspace', 'bicubic', 'gaussian']

    checkpoint = 'outputs/dataset_images/resolution_dataset25_full/srdense/kspace_up_and_down_factor_2/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'
   
    # checkpoint = 'outputs/dataset_images/bicubic_dataset25/srdense/bicubic_up_and_down_factor_2/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'
    # checkpoint = 'outputs/gaussian_dataset25_sigma100/srdense/gaussian_mul_wo_circular_mask_sigma100/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'
    # checkpoint = 'outputs/dataset_images/upscaled_50_micron_datset/srdense/srdense_50_micron_degradation/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'
    
    # for sigma in  sigma_value:
    #     checkpoint = 'outputs/gaussian_dataset25_sigma'+str(sigma)+'/srdense/gaussian_mul_wo_circular_mask_sigma'+str(sigma)+'/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'
    #     checkpoint1.append(checkpoint)

    # checkpoint2 = ['outputs/dataset_images/bicubic_dataset25/srdense/bicubic_up_and_down_factor_2/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
    # 'outputs/dataset_images/combined_kspace_gaussian_bicubic/srdense/srdense_combine_kpsace_bicubic_gaussian/checkpoints/z_axis/factor_2/epoch_250_f_2.pth',
    # 'outputs/dataset_images/resolution_dataset25_full/srdense/kspace_up_and_down_factor_2/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
    # 'outputs/dataset_images/simulated_degradation_dataset25/srdense/srdense_simulated_degradation/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
    # 'outputs/dataset_images/upscaled_50_micron_datset/srdense/srdense_50_micron_degradation/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'
    # ]
   
    
    
    names = ['kspace', 'bicubic', 'gaussian', '50_micron']

    # annotations = 'upsample/combine/annotation_hr1_dict.pkl'

    annotations = [
        'dataset_images/resolution_dataset25_full/z_axis/factor_2/test_annotation.pkl',
        'dataset_images/bicubic_dataset25/z_axis/factor_2/annotation_test_dict.pkl',
        'dataset_images/gaussian_dataset25_sigma100/test_annotation.pkl',
        'dataset_images/upscaled_50_micron_datset/test_annotation.pkl'

        # 'dataset_images/gaussian_dataset25_sigma50/test_annotation.pkl',
        # 'dataset_images/gaussian_dataset25_sigma75/test_annotation.pkl',
        # 'dataset_images/gaussian_dataset25_sigma100/test_annotation.pkl',

        # 'dataset_images/bicubic_dataset25/z_axis/factor_2/annotation_test_dict.pkl',
        # 'dataset_images/combined_kspace_gaussian_bicubic/test_annotation.pkl',
        # 'dataset_images/resolution_dataset25_full/z_axis/factor_2/test_annotation.pkl',
        # 'dataset_images/simulated_degradation_dataset25/test_annotation.pkl',
        
    ] 

    image_path_sigma = [
        'dataset_images/resolution_dataset25_full/z_axis/factor_2/test',
        'dataset_images/bicubic_dataset25/z_axis/factor_2/test',
        'dataset_images/gaussian_dataset25_sigma100/z_axis/factor_2/test',
        'dataset_images/upscaled_50_micron_datset/z_axis/factor_2/test'


        # 'dataset_images/gaussian_dataset25_sigma50/z_axis/factor_2/test',
        # 'dataset_images/gaussian_dataset25_sigma75/z_axis/factor_2/test',
        # 'dataset_images/gaussian_dataset25_sigma125/z_axis/factor_2/test',

        # 'dataset_images/combined_kspace_gaussian_bicubic/z_axis/factor_2/test',
        # 'dataset_images/resolution_dataset25_full/z_axis/factor_2/test',
        # 'dataset_images/simulated_degradation_dataset25/z_axis/factor_2/test',      

    ]

    label_path ='dataset_images/combined_kspace_gaussian_bicubic/z_axis/label/test'
    

    '''set device'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device

    metrics = ['psnr', 'ssim', 'mse', 'nrmse','hfen']

    psnr = PSNR()
    ssim = SSIM()

    psnr = psnr.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)
    ssim = ssim.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)

    opt.psnr= psnr
    opt.ssim=ssim
    opt.mse = nn.MSELoss().to(device=opt.device)
    opt.nrmse = NRMSELoss().to(device=opt.device)


    initial_list = []  # list of dictionary for each dataset
    model_list = []

    model = load_model(checkpoint,device)
    if device != 'cpu':
        num_of_gpus = torch.cuda.device_count()
        model = nn.DataParallel(model,device_ids=[*range(num_of_gpus)])
    model.to(device)
    model.eval()
    opt.model=model

    for index,dataset in enumerate(dataset_names):
        opt.dictionary = read_dictionary(annotations[index])
        opt.image_path = image_path_sigma[index]
        if dataset_names[index] == '50_micron':
            opt.label_path = 'dataset_images/upscaled_50_micron_datset/z_axis/label/test'
        else:
            opt.label_path = label_path

        initial, model = evaluate_model(opt)  # each dictionary of list with keys psnr,ssim,mse,nrmse,hfen
        model_list.append(model)
        initial_list.append(initial)
        print("Image path", opt.image_path)
        print("Label path",opt.label_path)
        print("checkpoint", checkpoint)
        print("Evaluating for model", dataset_names[index])
        print("MODEL METRIC")
        for metric in metrics:
            print("Average Values for {} is {} ".format(metric,statistics.mean(model[metric])))
        print("INITIAL METRIC")
        for metric in metrics:
            print("Average Values for {} is {} ".format(metric,statistics.mean(initial[metric])))

        print("**************************************************************************************************")

    plot_boxplot(initial_list=initial_list,model_list=model_list)    



