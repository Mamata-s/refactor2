
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value 
import torch
# from utils.preprocess import 
import torch.nn as nn
from dataset.dataset_cv import MRIDataset,MRIDatasetPatch,RdnSampler
from utils.config import set_val_dir,set_train_dir
from models.densenet import SRDenseNet
from models.patch_gan import PatchGAN,init_model
from models.unet import Unet, UnetSmall
from models.resunet import ResUNet
import torch.optim as optim



''' set the dataset path based on opt.dataset,opt.factor values and load & return the same dataset/dataloader'''
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



'''reduce learning rate of optimizer by half on every  150 and 225 epochs'''
def adjust_learning_rate(optimizer, epoch,lr):
    if epoch % 150 == 0 or epoch % 250==0:
        lr = lr * 0.5
    # log to TensorBoard
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



'''load the model instance based on opt.model_name value'''
def load_model(opt):
    if opt.model_name in ['srdense']:
        model =  SRDenseNet(num_channels=1, growth_rate = opt.growth_rate, num_blocks = opt.num_blocks, num_layers=opt.num_layers).to(opt.device)
        model = init_model( model, opt.device,init=opt.init)
    elif opt.model_name in ['unet']:
        model = Unet(in_channels= 1, out_channels= 1, n_blocks=opt.n_blocks, start_filters=opt.start_filters,activation=opt.activation,
                 normalization=opt.normalization,conv_mode = opt.conv_mode,dim= opt.dim,up_mode=opt.up_mode)
        model = init_model( model, opt.device,init=opt.init)
    elif opt.model_name in ['patch_gan','gan']:
        model= PatchGAN(opt)
    elif opt.model_name in ['unet_small']:
        model =init_model( UnetSmall(in_ch= 1,
                 out_ch= 1), opt.device,init=opt.init)
    elif opt.model_name in ['resunet']:
        model = init_model(ResUNet(in_ch= 1,
                 out_ch= 1),opt.device,init=opt.init)
    else:
        print(f'Model {opt.model_name} not implemented')
    return model




'''get the optimizer based on opt.criterion value'''
def get_criterion(opt):
    if opt.criterion in ['mse']:
        criterion = nn.MSELoss()
    elif opt.criterion in ['l1']:
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
    return criterion



'''get the optimizer based on opt.optimizer value'''
def get_optimizer(opt,model):
    if opt.optimizer in ['adam']:
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
        return optimizer
    else:
        print('optimizer not found')
        return None