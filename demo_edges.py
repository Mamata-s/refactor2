# from tabnanny import check
import torch
import torch.nn as nn
import  cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import argparse
import numpy as np
from utils.preprocess import min_max_normalize,tensor2image
from dataset.dataset_cv import MRIDatasetEdges, RdnSampler
from utils.load_model import load_model_main
from utils.train_utils import load_eval_dataset_edges
from train_edges import load_dataset_edges
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
default='outputs/resolution_dataset50/srdense/srdense_edge_training_debug/checkpoints/z_axis/factor_4/epoch_900_f_4.pth')

# parser.add_argument('-checkpoint', type=str, metavar='',required=False,help='checkpoint path',
# default='epoch_1500_f_4.pth')

parser.add_argument('-model_name', type=str, metavar='',help='name of model',default='dense')
# for loading the val dataset path
parser.add_argument('-factor', type=int, metavar='',required=False,help='resolution factor',default=2)
parser.add_argument('-dataset_name', type=str, metavar='',help='name val dataset',default='full')
parser.add_argument('-dataset_size', type=str, metavar='',help='size of val dataset',default='resolution_dataset50')
parser.add_argument('-save-dir', type=str, metavar='',help='plots saving directory',
default='')

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
opt.size=50

opt.val_image_dir = 'resolution_dataset50/full/factor_{}/val'.format(opt.factor)
opt.val_label_dir = 'resolution_dataset50/full/label/val' 

#load validation dataset
val_dataloader,val_datasets = load_eval_dataset_edges(opt)


dataiter = iter(val_dataloader)
images,labels,lr_edges = dataiter.next()


model = load_model_main(opt)

if device != 'cpu':
    num_of_gpus = torch.cuda.device_count()
    model = nn.DataParallel(model,device_ids=[*range(num_of_gpus)])
model.to(device)

model.eval()

images = images.to(device)
labels = labels.to(device)
lr_edges = lr_edges.to(device)
pred_edges = model(lr_edges)
outputs = pred_edges+images

label_edges = labels-images

# recover = images+label_edges
print('range of label edges', label_edges.min(), label_edges.max())
print('range of pred edges', pred_edges.min(), pred_edges.max())

label_edges = min_max_normalize(label_edges)
pred_edges = min_max_normalize(pred_edges)

label_edges = tensor2image(label_edges)
out = tensor2image(outputs)
images = tensor2image(images)
labels = tensor2image(labels)
lr_edges = tensor2image(lr_edges)
pred_edges = tensor2image(pred_edges)


# recover = images+label_edges # image added to label edges after image2tensor conversion will not reover hr image 

print('image shape',images.shape)
print('output shape',out.shape)
print('lr edge shape',lr_edges.shape)


# plot images
fnsize = 27
fig = plt.figure(figsize=(25,15))
cols = 6

fig.add_subplot(1, cols, 1)
plt.title('label',fontsize=fnsize)
plt.imshow(labels,cmap='gray')
psnr = peak_signal_noise_ratio(labels,labels,data_range=labels.max() - labels.min())
ssim = structural_similarity(labels,labels,multichannel=False,gaussian_weights=True, sigma=1.5, 
                    use_sample_covariance=False, data_range=labels.max() - labels.min())
plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim),fontsize=25)

fig.add_subplot(1, cols, 2)
plt.title('output',fontsize=fnsize)
plt.imshow(out,cmap='gray')
psnr = peak_signal_noise_ratio(labels,out,data_range=labels.max() - labels.min())
ssim = structural_similarity(labels,out,multichannel=False,gaussian_weights=True, sigma=1.5, 
                    use_sample_covariance=False, data_range=labels.max() - labels.min())
plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim),fontsize=fnsize)

fig.add_subplot(1, cols, 3)
plt.title('lr',fontsize=fnsize)
plt.imshow(images,cmap='gray')
psnr = peak_signal_noise_ratio(labels,images,data_range=labels.max() - labels.min())
ssim = structural_similarity(labels,images,multichannel=False,gaussian_weights=True, sigma=1.5, 
                    use_sample_covariance=False, data_range=labels.max() - labels.min())
plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim),fontsize=25)

fig.add_subplot(1, cols, 4)
plt.title('input edges',fontsize=fnsize)
plt.imshow(lr_edges,cmap='gray')

fig.add_subplot(1, cols, 5)
plt.title('predicted edges',fontsize=fnsize)
plt.imshow(pred_edges,cmap='gray')


fig.add_subplot(1, cols, 6)
plt.title('Label edges',fontsize=fnsize)
plt.imshow(label_edges,cmap='gray')

# if not os.path.exists(opt.save_dir):
#     os.makedirs(opt.save_dir)
plt.savefig('demo.png')
# plt.savefig('demo.png')
plt.show()
    
   

    