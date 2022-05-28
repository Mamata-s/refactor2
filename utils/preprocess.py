
# https://github.com/yjn870/SRDenseNet-pytorch/blob/5cf7af791fdd200441d71de5e3b3d4e8c3941e9c/utils.py (train code and model)
# https://github.com/andreasveit/densenet-pytorch/blob/master/train.py (logger and learning rate adjusting)

import torch
import numpy as np
import os
from PIL import Image
from torchvision.transforms import ToTensor
import glob
from scipy import ndimage

def calc_patch_size(func):
    def wrapper(args):
        if args.scale == 2:
            args.patch_size = 10
        elif args.scale == 3:
            args.patch_size = 7
        elif args.scale == 4:
            args.patch_size = 6
        else:
            raise Exception('Scale Error', args.scale)
        return func(args)
    return wrapper


def calc_psnr(img1, img2):
    return 10. * np.log10(1. / np.mean((img1 - img2) ** 2))


def create_dictionary(image_dir,label_dir):
    lst = []
    for f in os.listdir(image_dir):
        if not f.startswith('.'):
            lst.append(f)
        else:
            pass
    lst.sort()
    label_lst=[]
    for f in os.listdir(label_dir):
        if not f.startswith('.'):
            label_lst.append(f)
        else:
            pass
    label_lst.sort()
   
    dir_dictionary={}
    for i in range(len(lst)):
        dir_dictionary[lst[i]]=label_lst[i]
        
    return dir_dictionary

def min_max_normalize(image):
    max_img = image.max()
    min_img = image.min()
    denom = max_img-min_img
    norm_image = (image-min_img)/denom
    return norm_image 

def normalize(image,max_value=None):
    if max_value:
        return image/max_value
    else:
        return image/image.max()   


def Average(lst):
    return sum(lst) / len(lst)


def check_image(fn):
    try:
        im = Image.open(fn)
        im.verify()
        return True
    except:
        return False
    
def check_image_dir(path):
    for fn in glob.glob(path):
        if not check_image(fn):
            print("Corrupt image: {}".format(fn))
            os.remove(fn)

def save_img(image_tensor, filename):
   image_numpy = image_tensor.squeeze().detach().to('cpu').float().numpy()
   image_numpy = min_max_normalize(image_numpy)
   image_numpy=image_numpy*255
   image_numpy = image_numpy.clip(0, 255)
   image_numpy = image_numpy.astype(np.uint8)
   image_pil = Image.fromarray(image_numpy)
   image_pil.save(filename)
   print("Image saved as {}".format(filename))


def apply_model(model,epoch,opt,addition=False):
    image= Image.open(opt.epoch_image_path)
    image_tensor = ToTensor()(image)
    image_tensor= torch.unsqueeze(image_tensor.float(),0)
    image_tensor =  image_tensor.to(opt.device)
    image_tensor = min_max_normalize(image_tensor)
    output = model(image_tensor)
    if addition:
        output=output+image_tensor
        output = min_max_normalize(output)
    if not os.path.exists(opt.epoch_images_dir):
        os.makedirs(opt.epoch_images_dir)
    path = os.path.join(opt.epoch_images_dir,'epoch_{}.png'.format(epoch))
    # file_name = opt.epoch_images_dir+'/epoch_{}.png'.format(epoch)
    save_img(output,path)
    return True


def hfen_error(original_arr,est_arr,sigma=3):
   original = ndimage.gaussian_laplace(original_arr,sigma=sigma)
   est = ndimage.gaussian_laplace(est_arr,sigma=sigma)
   num = np.sum(np.square(original-est))
   deno = np.sum(np.square(original))
   hfen = np.sqrt(num/deno)
   return hfen