
# import the modules
import os
from os import listdir
import numpy as np
import cv2
import pickle

def get_window():
    # window1d = np.abs(np.blackman(360))
    # window1yd = np.abs(np.blackman(256))
    window1d = np.abs(np.bartlett(360))
    window1yd = np.abs(np.bartlett(256))
    window2d = np.sqrt(np.outer(window1d,window1yd))
    return window2d
 
# get the path/directory
folder_dir = "dataset_images/hamming_dataset25/z_axis/label/test/"
save_dir = "dataset_images/bartlett_test_set/factor_2/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

blackman_annotation = {}
for images in os.listdir(folder_dir):
    path = os.path.join(folder_dir, images)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    window2d =get_window()

    fshift = np.fft.fftshift(np.fft.fft2(image))
    factor=2
    y,x = fshift.shape
    data_pad = np.zeros((y,x),dtype=np.complex_)
    mask_lower = np.zeros((y,x), dtype=np.float32)

    center_y = y//2 #defining the center of image in x and y direction
    center_x = x//2
    startx = center_x-(x//(factor*2))  
    starty = center_y-(y//(factor*2))

    arr = fshift[starty:starty+(y//factor),startx:startx+(x//factor)]
    data_pad[starty:starty+(y//factor),startx:startx+(x//factor)] = arr

    mask_lower[starty:starty+(y//factor),startx:startx+(x//factor)] = window2d
    mask_upper = 1-mask_lower
    masked_upper = mask_upper*fshift
    masked_lower = mask_lower*fshift

    blackman_image = np.abs(np.fft.ifft2(np.fft.ifftshift(masked_lower)))
    blackman_image =( blackman_image-blackman_image.min())/(blackman_image.max()-blackman_image.min())
    blackman_image = blackman_image*255.
    print(cv2.imwrite(save_dir+'{}.png'.format(os.path.splitext(images)[0]),blackman_image))

    blackman_annotation[images]= images


with open('dataset_images/bartlett_test_set/bartlett_test_annotation.pkl', 'wb') as f:
    pickle.dump(blackman_annotation, f)

