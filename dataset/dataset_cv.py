import torch
import  os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

import utils as ut
import copy
from utils.preprocess import create_dictionary
from utils.preprocess import image2tensor

import random
import cv2


class MRIDataset(Dataset):
    def __init__(self, image_dir, label_dir,transform=None,size=50):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.labels = os.listdir(label_dir)
        assert len(self.images) == len(self.labels), ('images folder and label folder should have the same length, but got '
                                                f'{len(self.images)} and {len(self.labels)}.')
        self.dir_dict = create_dictionary(image_dir,label_dir)
        self.indices = [[] for _ in range(3)]
        self.size = size
        for i, x in enumerate(self.images):
            img_path = os.path.join(self.image_dir, x)
            image = np.array(Image.open(img_path).convert('L'))  #to convert to grayscale
            if self.size==25:
                if image.shape[0]== 512 and image.shape[1]== 304 : self.indices[0].append(i)
                if image.shape[0]== 720 and image.shape[1]== 304 : self.indices[1].append(i)
                if image.shape[0]== 720 and image.shape[1]== 512 : self.indices[2].append(i)
            else:
                if image.shape[0]== 256 and image.shape[1]== 152 : self.indices[0].append(i)
                if image.shape[0]== 360 and image.shape[1]== 152 : self.indices[1].append(i)
                if image.shape[0]== 360 and image.shape[1]== 256 : self.indices[2].append(i)

    def __len__(self):
        return len(self.images)

    def classes(self):
        return self.indices

    def __getitem__(self, index):
        dict_key = self.images[index]
        img_path = os.path.join(self.image_dir, self.images[index])
        label_path = os.path.join(self.label_dir, self.dir_dict[dict_key])

        image = cv2.imread(img_path).astype(np.float32) / 255.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.imread(label_path).astype(np.float32) / 255.
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)


        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        image = image2tensor(image, range_norm=False, half=False)
        label = image2tensor(label, range_norm=False, half=False)

        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)

        # image= torch.unsqueeze(image.float(),0)
        # label = torch.unsqueeze(label.float(),0)
    
        return image,label


class RdnSampler():
    def __init__(self, data_source, batch_size, shuffle=True,classes=[]):
        self.classes = classes
        classes = copy.deepcopy(self.classes)
        self.indices = [[i for _ in range(len(klass))] for i, klass in enumerate(classes)]
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):
        batch_lists = []
        for cluster_indices in self.indices:
            batches = [cluster_indices[i:i + self.batch_size] for i in range(0, len(cluster_indices), self.batch_size)]
            # filter our the shorter batches
            batches = [_ for _ in batches if len(_) == self.batch_size]
            if self.shuffle:
                random.shuffle(batches)
            batch_lists.append(batches)       
        
        # flatten lists and shuffle the batches if necessary
        # this works on batch level
        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)
        # final flatten  - produce flat list of indexes
        lst = self.flatten_list(lst)        
        return iter(lst)

    def __len__(self):
        return len(self.data_source)


class MRIDatasetPatch(Dataset):
    def __init__(self, image_dir, label_dir,transform=None,size=50,normalize=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.labels = os.listdir(label_dir)
        assert len(self.images) == len(self.labels), ('images folder and label folder should have the same length, but got '
                                                f'{len(self.images)} and {len(self.labels)}.')
        self.dir_dict = create_dictionary(image_dir,label_dir)
        self.indices = [[] for _ in range(3)]
        self.size = size
        self.normalize=normalize
    
    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        dict_key = self.images[index]
        img_path = os.path.join(self.image_dir, self.images[index])
        label_path = os.path.join(self.label_dir, self.dir_dict[dict_key])

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        # BGR convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        image = image2tensor(image, range_norm=False, half=False)
        label = image2tensor(label, range_norm=False, half=False)

        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)

        # image= torch.unsqueeze(image.float(),0)
        # label = torch.unsqueeze(label.float(),0)
    
        return image,label



class MRIDatasetEdges(Dataset):
    def __init__(self, image_dir, label_dir,transform=None,size=50):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.labels = os.listdir(label_dir)
        assert len(self.images) == len(self.labels), ('images folder and label folder should have the same length, but got '
                                                f'{len(self.images)} and {len(self.labels)}.')
        self.dir_dict = create_dictionary(image_dir,label_dir)
        self.indices = [[] for _ in range(3)]
        self.size = size
        for i, x in enumerate(self.images):
            img_path = os.path.join(self.image_dir, x)
            image = np.array(Image.open(img_path).convert('L'))  #to convert to grayscale
            if self.size==25:
                if image.shape[0]== 512 and image.shape[1]== 304 : self.indices[0].append(i)
                if image.shape[0]== 720 and image.shape[1]== 304 : self.indices[1].append(i)
                if image.shape[0]== 720 and image.shape[1]== 512 : self.indices[2].append(i)
            else:
                if image.shape[0]== 256 and image.shape[1]== 152 : self.indices[0].append(i)
                if image.shape[0]== 360 and image.shape[1]== 152 : self.indices[1].append(i)
                if image.shape[0]== 360 and image.shape[1]== 256 : self.indices[2].append(i)

    def __len__(self):
        return len(self.images)

    def classes(self):
        return self.indices

    def __getitem__(self, index):
        dict_key = self.images[index]
        img_path = os.path.join(self.image_dir, self.images[index])
        label_path = os.path.join(self.label_dir, self.dir_dict[dict_key])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        lr_edges = cv2.Canny(image = image, threshold1=1, threshold2=20) # Canny Edge Detection
        # lr_edges = 255-lr_edges

        image = image.astype(np.float32) / 255.
        label = label.astype(np.float32) / 255.
        lr_edges = lr_edges.astype(np.float32) / 255.
    

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        image = image2tensor(image, range_norm=False, half=False)
        label = image2tensor(label, range_norm=False, half=False)
        lr_edges = image2tensor(lr_edges, range_norm=False, half=False)

        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)
            lr_edges = self.transform(lr_edges)

        # image= torch.unsqueeze(image.float(),0)
        # label = torch.unsqueeze(label.float(),0)
    
        return image,label,lr_edges


class MRIDatasetDownsampleEdges(Dataset):
    def __init__(self, image_dir, label_dir,downsample_dir,transform=None,size=50):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.downsample_dir = downsample_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.labels = os.listdir(label_dir)
        self.downsamples = os.listdir(downsample_dir)
        assert len(self.images) == len(self.labels), ('images folder and label folder should have the same length, but got '
                                                f'{len(self.images)} and {len(self.labels)}.')
        assert len(self.images) == len(self.downsamples), ('images folder and downsample folder should have the same length, but got '
                                                f'{len(self.images)} and {len(self.downsamples)}.')                                        
        self.dir_dict = create_dictionary(image_dir,label_dir)
        self.dir_downsample_dict = create_dictionary(image_dir,downsample_dir)
        self.indices = [[] for _ in range(3)]
        self.size = size
        for i, x in enumerate(self.images):
            img_path = os.path.join(self.image_dir, x)
            image = np.array(Image.open(img_path).convert('L'))  #to convert to grayscale
            if self.size==25:
                if image.shape[0]== 512 and image.shape[1]== 304 : self.indices[0].append(i)
                if image.shape[0]== 720 and image.shape[1]== 304 : self.indices[1].append(i)
                if image.shape[0]== 720 and image.shape[1]== 512 : self.indices[2].append(i)
            else:
                if image.shape[0]== 256 and image.shape[1]== 152 : self.indices[0].append(i)
                if image.shape[0]== 360 and image.shape[1]== 152 : self.indices[1].append(i)
                if image.shape[0]== 360 and image.shape[1]== 256 : self.indices[2].append(i)

    def __len__(self):
        return len(self.images)

    def classes(self):
        return self.indices

    def __getitem__(self, index):
        dict_key = self.images[index]
        img_path = os.path.join(self.image_dir, self.images[index])
        label_path = os.path.join(self.label_dir, self.dir_dict[dict_key])
        downsample_path = os.path.join(self.downsample_dir,self.dir_downsample_dict[dict_key])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        downsample = cv2.imread(downsample_path)
        downsample = cv2.cvtColor(downsample, cv2.COLOR_BGR2GRAY)

        lr_edges = image-downsample
        # lr_edges = 255-lr_edges

        image = image.astype(np.float32) / 255.
        label = label.astype(np.float32) / 255.
        lr_edges = lr_edges.astype(np.float32) / 255.
    

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        image = image2tensor(image, range_norm=False, half=False)
        label = image2tensor(label, range_norm=False, half=False)
        lr_edges = image2tensor(lr_edges, range_norm=False, half=False)

        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)
            lr_edges = self.transform(lr_edges)

        # image= torch.unsqueeze(image.float(),0)
        # label = torch.unsqueeze(label.float(),0)
    
        return image,label,lr_edges


class MRIDatasetPatchDownsampleEdges(Dataset):
    def __init__(self, image_dir, label_dir,downsample_dir,transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.downsample_dir = downsample_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.labels = os.listdir(label_dir)
        self.downsamples = os.listdir(downsample_dir)
        assert len(self.images) == len(self.labels), ('images folder and label folder should have the same length, but got '
                                                f'{len(self.images)} and {len(self.labels)}.')
        assert len(self.images) == len(self.downsamples), ('images folder and downsample folder should have the same length, but got '
                                                f'{len(self.images)} and {len(self.downsamples)}.')                                        
        self.dir_dict = create_dictionary(image_dir,label_dir)
        self.dir_downsample_dict = create_dictionary(image_dir,downsample_dir)
        self.indices = [[] for _ in range(3)]
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        dict_key = self.images[index]
        img_path = os.path.join(self.image_dir, self.images[index])
        label_path = os.path.join(self.label_dir, self.dir_dict[dict_key])
        downsample_path = os.path.join(self.downsample_dir,self.dir_downsample_dict[dict_key])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        downsample = cv2.imread(downsample_path)
        downsample = cv2.cvtColor(downsample, cv2.COLOR_BGR2GRAY)

        lr_edges = image-downsample
        # lr_edges = 255-lr_edges

        image = image.astype(np.float32) / 255.
        label = label.astype(np.float32) / 255.
        lr_edges = lr_edges.astype(np.float32) / 255.
    

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        image = image2tensor(image, range_norm=False, half=False)
        label = image2tensor(label, range_norm=False, half=False)
        lr_edges = image2tensor(lr_edges, range_norm=False, half=False)

        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)
            lr_edges = self.transform(lr_edges)

        # image= torch.unsqueeze(image.float(),0)
        # label = torch.unsqueeze(label.float(),0)
    
        return image,label,lr_edges

