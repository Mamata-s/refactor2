'''
This code is written to generate data from the degradation model which was trained to simulate the realistic dergadation
'''
import torch
import torch.nn as nn
import  cv2
import matplotlib.pyplot as plt
import argparse
from utils.preprocess import tensor2image, image2tensor
from models.densenet_smchannel import SRDenseNet
import os
import numpy as np

import warnings
warnings.filterwarnings("ignore")



def load_model(opt):
    checkpoint = torch.load(opt.checkpoint,map_location=torch.device(opt.device))
    growth_rate = checkpoint['growth_rate']
    num_blocks = checkpoint['num_blocks']
    num_layers =checkpoint['num_layers']
    model = SRDenseNet(growth_rate=growth_rate,num_blocks=num_blocks,num_layers=num_layers)
    state_dict = model.state_dict()
    for n, p in checkpoint['model_state_dict'].items():
        new_key = n[7:]
        # new_key = n
        # print(new_key)
        if new_key in state_dict.keys():
            # new_p = p*random.uniform(2., 7.)
            state_dict[new_key].copy_(p)
        else:
            raise KeyError(new_key)
    return model


# ****************************************************************************************************************************************



# for dictionary having hr image name as key
def predict(model,image_path,device):
    
    input = cv2.imread(image_path)
    input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY).astype(np.float32)

    print("Image path", image_path)
    print("**********************************************************************************************************************")

    input = input/255.
    input = image2tensor(input).unsqueeze(dim=0).to(device)
    
    # perform super-resolution with srcnn
    output = model(input)

    # return images and scores
    return input, output



def save_results(model,opt):
    for file in os.listdir(opt.image_path):
        image_path = os.path.join(opt.image_path,file)

        print("Image name", image_path)
        print("********************************************************************************")

        # perform super-resolution
        input, output = predict(model=model,image_path=image_path,device=opt.device)
        
        print(cv2.imwrite(opt.preds_dir+'{}.png'.format(os.path.splitext(file)[0]),tensor2image(output)))
        plt.close()

        print("Image saved")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model demo')
    parser.add_argument('--checkpoint', type=str, metavar='',required=False,help='checkpoint path', default='outputs/degradation/checkpoints/epoch_2500_f_2.pth')
    parser.add_argument('--preds-dir', type=str, metavar='',help='plot save dir',default='test_set/val/')
    parser.add_argument('--image-path', type=str, metavar='',required=False,help='gaussian mult image',default='dataset_images/resolution_dataset25_full/z_axis/label/val')
    opt = parser.parse_args()

    print(opt.image_path)
    
    if not os.path.exists(opt.preds_dir):
        os.makedirs(opt.preds_dir)

    '''set device'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device

    model = load_model(opt)

    if device != 'cpu':
        num_of_gpus = torch.cuda.device_count()
        model = nn.DataParallel(model,device_ids=[*range(num_of_gpus)])

    model.to(device)

    model.eval()

  
    save_results(model,opt)
