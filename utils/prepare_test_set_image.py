import argparse
import os
import cv2
import numpy as np

    
def crop_pad_kspace(data,pad=False,factor=4):  #function for cropping and/or padding the image in kspace
    F = np.fft.fft2(data)
    fshift = np.fft.fftshift(F)
    
    y,x = fshift.shape
    data_pad = np.zeros((y,x),dtype=np.complex_)
    center_y = y//2 #defining the center of image in x and y direction
    center_x = x//2
    startx = center_x-(x//(factor*2))  
    starty = center_y-(y//(factor*2))
    
    arr = fshift[starty:starty+(y//factor),startx:startx+(x//factor)]
    if pad:
        data_pad[starty:starty+(y//factor),startx:startx+(x//factor)] = arr
        img_reco_cropped = np.fft.ifft2(np.fft.ifftshift(data_pad))
    else:
        img_reco_cropped = np.fft.ifft2(np.fft.ifftshift(arr)) 
    return np.abs(img_reco_cropped )


def prepare_image_fourier(path,save_path,factor):  #read image from given path and downsample for given factor and save it into image/factor_{} folder
    for file in os.listdir(path):
        # open the file
        img = cv2.imread(path + '/' + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)
        
        downsample_image = crop_pad_kspace(img,pad=True,factor=factor)
        downsample_image = downsample_image/downsample_image.max()
        downsample_image = downsample_image*255.
        downsample_image = downsample_image.astype(np.uint8)

        # save the image
        print('Saving {}'.format(file))
        cv2.imwrite(save_path+'/{}'.format(file), downsample_image)


def prepare_images_interpolation(path,save_path,factor):
    
    # loop through the files in the directory
    for file in os.listdir(path):
        
        # open the file
        img = cv2.imread(path + '/' + file)
        
        # find old and new image dimensions
        h, w, _ = img.shape
        new_height = h / factor
        new_width = w / factor
        
        # resize the image - down
        img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_LINEAR)
        
        # resize the image - up
        img = cv2.resize(img, (w, h), interpolation = cv2.INTER_LINEAR)
        
        # save the image
        print('Saving {}'.format(file))
        cv2.imwrite(save_path+'/{}'.format(factor,file), img)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--factor',type=int,required=False,default=2)
    parser.add_argument('--label_path',type=str,required=False,default='test_labels')
    parser.add_argument('--save_path',type=str,required=False,default='test_images')
    args = parser.parse_args()

    prepare_image_fourier(args.label_path,args.save_path,args.factor)
    print('Image processing complete')







