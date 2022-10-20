import torch
import argparse
import sys
from dataset.dataset_cv import RdnSampler,MRIDatasetEdges


if __name__ == "__main__":
    '''get the configuration file'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', help="label image path", type=str, required=False, default='gaussian_dataset25_mul/z_axis/label/train')
    parser.add_argument('--image_path', help="lr image path", type=str, required=False, default='gaussian_dataset25_mul/z_axis/factor_2/train')
    opt   = parser.parse_args()
    opt.batch_size=1
    opt.size=25
    opt.apply_mask = False

    train_datasets = MRIDatasetEdges(opt.image_path, opt.label_path,size=opt.size,apply_mask=opt.apply_mask,dir_dict='gaussian_dataset25_mul/annotation.pkl')
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size = opt.batch_size,shuffle=False,
        num_workers=8,pin_memory=False,drop_last=False)

    avg_min = 0.
    avg_max = 0.
    count = 0
    for idx, (data) in enumerate(train_dataloader):
        # print('idx',idx) 
        images = data['image']
        labels = data['label']
        label_error_map = labels-images
        min = label_error_map.min().item()
        max = label_error_map.max().item()
        avg_min = avg_min+min
        avg_max = avg_max+max
        count += 1

    print('Average Min is ', avg_min/count)
    print('Average Max is', avg_max/count)

