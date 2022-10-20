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

#**************************************************************************************************************************************************************

with open('dataset_images/gaussian_dataset25_sigma50/test_annotation.pkl', 'rb') as f:
    new_dict = pickle.load(f)

print(new_dict)
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

