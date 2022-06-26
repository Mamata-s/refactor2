
import torch
import  cv2, os
from torch.utils.data import Dataset
import numpy as np

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
from utils.general import min_max_normalize
from utils.preprocess import create_dictionary

class MRIDataset3DPatch(Dataset):
    def __init__(self, image_dir, label_dir,transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.labels = os.listdir(label_dir)
        self.images = os.listdir(image_dir)
        self.labels = os.listdir(label_dir)
        assert len(self.images) == len(self.labels), ('images folder and label folder should have the same length, but got '
                                                f'{len(self.images)} and {len(self.labels)}.')
        self.dir_dict = create_dictionary(image_dir,label_dir)
    
    def __len__(self):
        return len(self.images)

    def get_dictionary(self):
        return self.dir_dict
        
    def __getitem__(self, index):
        dict_key = self.images[index]
        img_path = os.path.join(self.image_dir, self.images[index])
        label_path = os.path.join(self.label_dir, self.dir_dict[dict_key])
        image = np.load(img_path)
        image = torch.from_numpy(image)
        label = np.load(label_path)
        label = torch.from_numpy(label)

        # print('dataloader image shape',image.shape)
        # print('dataloader label shape',label.shape)
        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)
        #normalize input and label image
        image = min_max_normalize(image)
        label = min_max_normalize(label)
    
        image= torch.unsqueeze(image.float(),0)
        label = torch.unsqueeze(label.float(),0)
        
        return image,label