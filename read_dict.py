import pickle as pkl
from venv import create
import yaml
import pickle

# reading the data from the file
# pkl_path ='outputs/resolution_dataset50/srdense/srdense_edge_training_4/losses/z_axis/factor_4/configuration'
# pkl_path='outputs/resolution_dataset50/srdense/srdense_edge_ltype_addition_4/losses/z_axis/factor_4/configuration'
# with open(pkl_path, 'rb') as f:
#     data = pkl.load(f)


# # print(data)
# with open('config.yml', 'w') as yaml_file:
#     yaml.dump(data, yaml_file, default_flow_style=False)

#*****************************************************************************************************************************************************

# interchnage key and value in dictionary
# with open('upsample/f1_160/annotation_hr1_dict.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)

# # print(loaded_dict)

# new_dict = {}
# for key in loaded_dict:
#     value = loaded_dict[key]
#     new_dict[value]=key

# print(new_dict)

# with open('annotation_train_dict.pkl', 'wb') as f:
#     pickle.dump(new_dict, f)

#*************************************************************************************************************************************************


# combine dictionary from bicubic, kspace, gaussian 150 and gaussian 125

# bicubic
# with open('dataset_images/combined_kspace_gaussian_bicubic/annotation_test_dict_bicubic.pkl', 'rb') as f:
#     new_dict = pickle.load(f)


# #kspace
# with open('/home/cidar/Desktop/refactor2/dataset_images/resolution_dataset25_full/z_axis/factor_2/test_annotation.pkl', 'rb') as f:
#     f2_dict = pickle.load(f)

# # print(f2_dict)


# with open('dataset_images/gaussian_dataset25_sigma125/test_annotation.pkl', 'rb') as f:
#     f4_dict = pickle.load(f)
# # print(f4_dict)

# with open('dataset_images/gaussian_dataset25_sigma150/test_annotation.pkl', 'rb') as f:
#     f5_dict = pickle.load(f)

# f1_dict = {}
# for key in new_dict:
#     value = new_dict[key]
#     f1_dict[value]=key


# f1_dict.update(f2_dict)
# f1_dict.update(f4_dict)
# f1_dict.update(f5_dict)

# print(f1_dict)
# with open('test_annotation.pkl', 'wb') as f:
#     pickle.dump(f1_dict, f)

# #**************************************************************************************************************************************************************

# just plotting image from the dictionary
# with open('dataset_images/gaussian_dataset25_sigma50/test_annotation.pkl', 'rb') as f:
#     new_dict = pickle.load(f)

# print(new_dict)
# print(new_dict['hr_f1_160_z_196.png'])

# import cv2
# import numpy as np

# label = cv2.imread('dataset_images/combined_kspace_gaussian_bicubic/z_axis/label/val/hr_f4_149_z_210.png').astype(np.float32) / 255.
# label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
# print(label.shape)

#********************************************************************************************************************************************************

import os

# # create annotation
# def create_dictionary(image_dir,label_dir):
#     lst = []
#     for f in os.listdir(image_dir):
#         if not f.startswith('.'):
#             lst.append(f)
#         else:
#             pass
#     lst.sort()
#     label_lst=[]
#     for f in os.listdir(label_dir):
#         if not f.startswith('.'):
#             label_lst.append(f)
#         else:
#             pass
#     label_lst.sort()
   
#     dir_dictionary={}
#     for i in range(len(lst)):
#         dir_dictionary[lst[i]]=label_lst[i]
        
#     return dir_dictionary

# image_dir = 'dataset_images/upscaled_50_micron_datset/z_axis/factor_2/train'
# label_dir = 'dataset_images/upscaled_50_micron_datset/z_axis/label/train'
# dictionary = create_dictionary(image_dir, label_dir)
# # print(dictionary)
# with open('annotation.pkl', 'wb') as f:
#     pickle.dump(dictionary, f)

# image_dir = 'dataset_images/upscaled_50_micron_datset/z_axis/factor_2/test'
# label_dir = 'dataset_images/upscaled_50_micron_datset/z_axis/label/test'
# dictionary = create_dictionary(image_dir, label_dir)
# # print(dictionary)
# with open('test_annotation.pkl', 'wb') as f:
#     pickle.dump(dictionary, f)

# image_dir = 'dataset_images/upscaled_50_micron_datset/z_axis/factor_2/val'
# label_dir = 'dataset_images/upscaled_50_micron_datset/z_axis/label/val'
# dictionary = create_dictionary(image_dir, label_dir)
# # print(dictionary)
# with open('val_annotation.pkl', 'wb') as f:
#     pickle.dump(dictionary, f)


# ***********************************************************************************************************

#  create dictionary by combining dictionary from all dataset and appending path to it

# read_dictionary 
# append path 
# append it to combine dictionary 
# save new dictionary

def append_subscript(dict, key_append, value_append):
    new_dict = {}
    for key in dict:
        value = dict[key]
        new_key = str(key_append)+ key
        new_value = str(value_append)+ value
        new_dict[new_key]= new_value
    return new_dict


# combine_dictionary = {}

# load_path = [
#     'dataset_images/gaussian_dataset25_sigma25/annotation.pkl',
#     'dataset_images/gaussian_dataset25_sigma50/annotation.pkl',
#     'dataset_images/gaussian_dataset25_sigma75/annotation.pkl',
#     'dataset_images/gaussian_dataset25_sigma100/annotation.pkl',
#     'dataset_images/gaussian_dataset25_sigma125/annotation.pkl',
#     'dataset_images/gaussian_dataset25_sigma150/annotation.pkl'
# ]
# key_appends= [
#     'dataset_images/gaussian_dataset25_sigma25/z_axis/factor_2/train/',
#     'dataset_images/gaussian_dataset25_sigma50/z_axis/factor_2/train/',
#     'dataset_images/gaussian_dataset25_sigma75/z_axis/factor_2/train/',
#     'dataset_images/gaussian_dataset25_sigma100/z_axis/factor_2/train/',
#     'dataset_images/gaussian_dataset25_sigma125/z_axis/factor_2/train/',
#     'dataset_images/gaussian_dataset25_sigma150/z_axis/factor_2/train/'   

# ]
# value_append = 'dataset_images/gaussian_dataset25_mul_wo_circular_mask/z_axis/label/train/'


# load_path = [
#     'dataset_images/gaussian_dataset25_sigma100F/annotation.pkl',
#     'dataset_images/hanning_dataset25/annotation_train_dict.pkl',
#     # 'dataset_images/hamming_dataset25/annotation_train_dict.pkl',
#     'dataset_images/bicubic_dataset25/z_axis/factor_2/annotation_train_dict.pkl',
#     'dataset_images/mean_blur_dataset25/annotation_train_dict.pkl',
#     # 'dataset_images/median_blur_dataset25/annotation_train_dict.pkl'
# ]
# key_appends= [
#     'dataset_images/gaussian_dataset25_sigma100F/z_axis/factor_2/train/',
#     'dataset_images/hanning_dataset25/z_axis/factor_2/train/',
#     # 'dataset_images/hamming_dataset25/z_axis/factor_2/train/',
#     'dataset_images/bicubic_dataset25/z_axis/factor_2/train/',
#     'dataset_images/mean_blur_dataset25/z_axis/factor_2/train/',
#     # 'dataset_images/median_blur_dataset25/z_axis/factor_2/train/'
# ]
# value_append = 'dataset_images/hamming_dataset25/z_axis/label/train/'


# for index,path in enumerate(load_path):
#     with open(path, 'rb') as f:
#         f_dict = pickle.load(f)
#         print(f_dict)
#         print("length of single dict", len(f_dict))
#         f_new_dict = append_subscript(f_dict, key_append=key_appends[index], value_append=value_append)
#         combine_dictionary.update(f_new_dict)


# print("length of combine dict", len(combine_dictionary))
# with open('all_degradation_combine_large_train_annotation.pkl', 'wb') as f:
#     pickle.dump(combine_dictionary, f)

# # print(combine_dictionary)
# print("length of combine dict", len(combine_dictionary))


# train_lengths = [26,26,29,26,26,26]
# train_lengths = [240,240,240] # for 3
# train_lengths = [360,360] #for 2
# train_lengths = [180,180,180,180] #for 4
# import random
# for index,path in enumerate(load_path):
#     with open(path, 'rb') as f:
#         f_dict = pickle.load(f)
#         keys =  list(f_dict.keys()) 
#         random.shuffle(keys)
#         keys = keys[:train_lengths[index]]
#         f_new_dict = {}
#         for key in keys:
#             f_new_dict[key]=f_dict[key]
#         print("length of single dict", len(f_new_dict))
#         f_new_dict = append_subscript(f_new_dict, key_append=key_appends[index], value_append=value_append)
#         combine_dictionary.update(f_new_dict)


# print("length of combine dict", len(combine_dictionary))
# with open('gauss_bicubic_hann_mean_degradation_combine_fix_train_annotation.pkl', 'wb') as f:
#     pickle.dump(combine_dictionary, f)

# print(combine_dictionary)



# read and append the folder path to dictionary

def read_dictionary(path):
    with open(path,'rb') as f:
        annotation_dict = pickle.load(f)
    return annotation_dict

def save_dicionary(path,annotation_dict):
    with open(path,'wb') as f:
        pickle.dump(annotation_dict, f)
        return 1

# path = 'dataset_images/dataset_index/ssim_annotation/train_ssim_annotation.pkl'
# save_path = 'dataset_images/dataset_index/ssim_annotation/train_ssim_annotation1.pkl'

# path = 'dataset_images/gaussian_dataset25_sigma100F/annotation_val_dict.pkl'
# save_path = 'dataset_images/gaussian_dataset25_sigma100F/annotation_val_dict_full_path.pkl'


path = 'dataset_images/bicubic_dataset25/z_axis/factor_2/annotation_test_dict.pkl'
save_path = 'dataset_images/bicubic_dataset25/annotation_test_dict_full_path.pkl'

# key_append = 'dataset_images/dataset_index/train/'
key_append = 'dataset_images/bicubic_dataset25/z_axis/factor_2/test/'
value_append= 'dataset_images/hamming_dataset25/z_axis/label/test/'
dictionary = read_dictionary(save_path)
print(dictionary)
quit()
new_dictionary = append_subscript(dictionary,key_append,value_append)
print(new_dictionary)
save_dicionary(save_path,new_dictionary)