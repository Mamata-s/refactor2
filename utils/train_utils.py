
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value 
import torch
# from utils.preprocess import 
import torch.nn as nn
from dataset.dataset import MRIDataset,MRIDatasetPatch,RdnSampler
from utils.config import set_val_dir,set_train_dir
from models.densenet import SRDenseNet,SRDenseNetUpscale
from models.patch_gan import PatchGAN
import torch.optim as optim

def load_dataset(opt):
    set_val_dir(opt)  #setting the training datasset dir
    set_train_dir(opt)  #setting the validation set dir
    if opt.patch:
        train_datasets = MRIDatasetPatch(opt.train_image_dir, opt.train_label_dir)
        val_datasets = MRIDatasetPatch(opt.val_image_dir, opt.val_label_dir)

        train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size = opt.train_batch_size,shuffle=True,
            num_workers=1,pin_memory=False,drop_last=False)
        eval_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size = opt.val_batch_size, shuffle=True,
            num_workers=1,pin_memory=False,drop_last=False)
    else:
        train_datasets = MRIDataset(opt.train_image_dir, opt.train_label_dir)
        val_datasets = MRIDataset(opt.val_image_dir, opt.val_label_dir)

        sampler = RdnSampler(train_datasets,opt.train_batch_size,True,classes=train_datasets.classes())
        val_sampler = RdnSampler(val_datasets,opt.val_batch_size,True,classes=train_datasets.classes())

        train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size = opt.train_batch_size,sampler = sampler,shuffle=False,
            num_workers=1,pin_memory=False,drop_last=False)
        eval_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size = opt.val_batch_size,sampler = val_sampler,shuffle=False,
            num_workers=1,pin_memory=False,drop_last=False)
    return train_dataloader,eval_dataloader,train_datasets,val_datasets


def adjust_learning_rate(optimizer, epoch,opt):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = opt.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    opt.lr=lr
    # log to TensorBoard
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_model(opt):
    if opt.model_name in ['srdense']:
        model =  SRDenseNet(num_channels=1, growth_rate = opt.growth_rate, num_blocks = opt.num_blocks, num_layers=opt.num_layers).to(opt.device) 
    elif opt.model_name in ['gan','patch_gan']:
        model = PatchGAN(opt)
    return model

def get_criterion(opt):
    if opt.criterion in ['mse']:
        criterion = nn.MSELoss()
    elif opt.criterion in ['l1']:
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
    return criterion


def get_optimizer(opt,model):
    if opt.optimizer in ['adam']:
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
        return optimizer
    else:
        print('optimizer not found')
        return None