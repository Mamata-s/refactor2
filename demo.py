# from tabnanny import check
import torch
import torch.nn as nn
import  cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import argparse
import numpy as np
from utils.preprocess import min_max_normalize
from dataset.dataset_cv import MRIDataset, RdnSampler
from utils.train_utils import load_val_dataset,set_val_dir
from utils.load_model import load_model_main
import os

import warnings
warnings.filterwarnings("ignore")

 
from PIL import Image,ImageChops
 
def get_error_image_array(image_arr1,image_arr2):
 image_arr1 = min_max_normalize(image_arr1)
 image_arr2 = min_max_normalize(image_arr2)
 im1obj = Image.fromarray(np.uint8(image_arr1*255))
 im2obj = Image.fromarray(np.uint8(image_arr2*255))
 error_obj = ImageChops.difference(im1obj,im2obj)
 error_arr = np.array(error_obj)
 return error_arr

parser = argparse.ArgumentParser(description='model demo')
parser.add_argument('-tfactor', type=int, metavar='',required=False,help='trained factor',default=2)
parser.add_argument('-checkpoint', type=str, metavar='',required=False,help='checkpoint path',
default='outputs/resolution_dataset50/resunet/RESUNET_KAI_Z_F2_BS32_LR0.001/checkpoints/z_axis/factor_2/best_weights_factor_2_epoch_15.pth')

parser.add_argument('-model_name', type=str, metavar='',help='name of model',default='resunet')
# for loading the val dataset path
parser.add_argument('-factor', type=int, metavar='',required=False,help='resolution factor',default=2)
parser.add_argument('-dataset_name', type=str, metavar='',help='name val dataset',default='full')
parser.add_argument('-dataset_size', type=str, metavar='',help='size of val dataset',default='resolution_dataset50')
parser.add_argument('-save-dir', type=str, metavar='',help='plots saving directory',
default='test_set/')

parser.add_argument('--addition',
                    help='add the output of model to input images for getting result image', action='store_true')

parser.add_argument('--error_map',
                    help='add the output of model to input images for getting result image', action='store_true')

opt = parser.parse_args()

opt.val_batch_size = 1

'''Set the addition argument value for saving epoch images'''
check=True
for arg in vars(opt):
    if arg in ['patch']:
        check=False
if check: opt.patch=False

'''set device'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
opt.device = device

set_val_dir(opt)  #setting the training datasset dir

#load validation dataset
val_dataloaders,val_datasets = load_val_dataset(opt)


dataiter = iter(val_dataloaders)
images, labels = dataiter.next()


model = load_model_main(opt)

if device != 'cpu':
    num_of_gpus = torch.cuda.device_count()
    model = nn.DataParallel(model,device_ids=[*range(num_of_gpus)])
model.to(device)

model.eval()

images = images.to(device)
labels = labels.to(device)
outputs = model(images)

if opt.addition:
    print('additing the output')
    outputs=outputs+images
out = outputs
out = out.squeeze().detach().to('cpu').numpy()
out = min_max_normalize(out)
images = images.squeeze().to('cpu').numpy()
labels = labels.squeeze().detach().to('cpu').numpy()

print('image shape',images.shape)
print('output shape',out.shape)

# plot images
fnsize = 27
fig = plt.figure(figsize=(20,13))
nrows=1
ncols=5

fig.add_subplot(nrows, ncols, 1)
plt.title('label',fontsize=fnsize)
plt.imshow(labels,cmap='gray')
psnr = peak_signal_noise_ratio(labels,labels,data_range=labels.max() - labels.min())
ssim = structural_similarity(labels,labels,multichannel=False,gaussian_weights=True, sigma=1.5, 
                    use_sample_covariance=False, data_range=labels.max() - labels.min())
plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim),fontsize=25)

fig.add_subplot(nrows, ncols, 2)
plt.title('output',fontsize=fnsize)
plt.imshow(out,cmap='gray')
psnr = peak_signal_noise_ratio(labels,out,data_range=labels.max() - labels.min())
ssim = structural_similarity(labels,out,multichannel=False,gaussian_weights=True, sigma=1.5, 
                    use_sample_covariance=False, data_range=labels.max() - labels.min())
plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim),fontsize=fnsize)

fig.add_subplot(nrows, ncols, 3)
plt.title('Input',fontsize=fnsize)
plt.imshow(images,cmap='gray')
psnr = peak_signal_noise_ratio(labels,images,data_range=labels.max() - labels.min())
ssim = structural_similarity(labels,images,multichannel=False,gaussian_weights=True, sigma=1.5, 
                    use_sample_covariance=False, data_range=labels.max() - labels.min())
plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim),fontsize=25)


fig.add_subplot(nrows, ncols, 4)
plt.title('Error : label-output',fontsize=fnsize)
error = get_error_image_array(labels,out)
plt.imshow(error,cmap='gray')
plt.xlabel(' (5)' ,fontsize=fnsize)
plt.tight_layout()


fig.add_subplot(nrows, ncols, 5)
plt.title('Error: label-input',fontsize=fnsize)
error = get_error_image_array(labels,images)
plt.imshow(error,cmap='gray')
plt.xlabel(' (4)' ,fontsize=fnsize)

if not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)
plt.savefig(opt.save_dir+'demo.png')
# plt.savefig('demo.png')
plt.show()
    
   

    