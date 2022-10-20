# process_yaml.py file
#imports
import yaml
import argparse
import sys
from utils.train_utils import load_dataset_edges,load_model
import torch
import torch.nn as nn
import copy
from utils.logging_metric import LogMetric,create_loss_meters_gan
from utils.train_utils import adjust_learning_rate,load_dataset_edges,load_dataset_downsample_edges
from utils.train_epoch import train_epoch_patch_gan_edges, validate_patch_gan_edges
# from utils.preprocess import apply_model
from utils.image_quality_assessment import PSNR,SSIM
from utils.general import save_configuration_yaml,LogEdgesOutputs
from utils.config import set_outputs_dir,set_training_metric_dir,set_plots_dir,set_train_dir,set_val_dir,set_downsample_train_val_dir
from models.densenet_smchannel import SRDenseNet

import os
import wandb

os.environ["CUDA_VISIBLE_DEVICES"]='0,1' 


def load_pretrained(checkpoint,device):
    checkpoint = torch.load(checkpoint,map_location=torch.device(device))
    growth_rate = checkpoint['growth_rate']
    num_blocks = checkpoint['num_blocks']
    num_layers =checkpoint['num_layers']
    model = SRDenseNet(growth_rate=growth_rate,num_blocks=num_blocks,num_layers=num_layers).to(device)
    state_dict = model.state_dict()
    for n, p in checkpoint['model_state_dict'].items():
        new_key = n[7:]
        # new_key = n
        # print(new_key)
        if new_key in state_dict.keys():
            state_dict[new_key].copy_(p)
        else:
            raise KeyError(new_key)
    return model,growth_rate, num_blocks, num_layers

def train(opt,model,train_datasets,train_dataloader,eval_dataloader,wandb=None):
    if opt.wandb:
        log_table_output = LogEdgesOutputs()
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    for epoch in range(opt.num_epochs):
        opt.lr_G = adjust_learning_rate(model.opt_G, epoch,opt.lr_G)
        opt.lr_D = adjust_learning_rate(model.opt_D, epoch,opt.lr_D)
    
        '''setting model in train mode'''
        model.train()

        '''train one epoch and evaluate the model'''
        epoch_losses = create_loss_meters_gan()
      
        outputs = train_epoch_patch_gan_edges(opt=opt,model= model,train_dataset=train_datasets,train_dataloader=train_dataloader,epoch=epoch,epoch_losses=epoch_losses)
        eval_loss, eval_l1,eval_psnr, eval_ssim,eval_hfen = validate_patch_gan_edges(opt,model, eval_dataloader)


        if opt.wandb:
            wandb.log({"val/val_loss" : eval_loss,
            "val/val_l1_error":eval_l1,
            "val/val_psnr": eval_psnr,
            "val/val_ssim":eval_ssim,
            "val/val_hfen":eval_hfen,
            "epoch": epoch
            })
            for key in epoch_losses.keys():
                wandb.log({"train/{}".format(key) : epoch_losses[key].avg,
                })
            wandb.log({"epoch": epoch})
            wandb.log({
                "other/learning_G": opt.lr_G,
                "other/learning_D": opt.lr_D,
            })
            if epoch % opt.n_freq == 0:
                log_table_output.append_list(outputs)  #create a class with list and function to loop through list and add to log table

        # apply_model(model.net_G,epoch,opt,addition=opt.addition)
        print('eval psnr: {:.4f}'.format(eval_psnr))

        if eval_psnr > best_psnr:
            best_epoch = epoch
            best_psnr = eval_psnr
            best_weights = copy.deepcopy(model.state_dict())

        '''adding to the dictionary'''
        metric_dict.update_dict([eval_loss,eval_l1,eval_psnr,eval_ssim,eval_hfen],training=False)

        metric_dict.update_dict([epoch_losses['loss_D_fake'].avg,
        epoch_losses['loss_D_real'].avg,
        epoch_losses['loss_D'].avg,
        epoch_losses['loss_G_GAN'].avg,
        epoch_losses['loss_G_L1'].avg,
        epoch_losses['loss_G'].avg])

    if opt.wandb:
        print("logging output table")
        if opt.apply_mask:
            log_table_output.log_images_and_mask(wandb=wandb)
        else:
            log_table_output.log_images(wandb=wandb)
        log_table_output.log_images(columns = ["epoch","image", "pred", "label"],wandb=wandb) 

    opt.generator = None  #remove model from yaml file before saving configuration
    opt.ssim = None
    opt.psnr = None
    path = metric_dict.save_dict(opt)
    _ = save_configuration_yaml(opt)

    path="best_weights_factor_{}_epoch_{}.pth".format(opt.factor,best_epoch)
    path = os.path.join(opt.checkpoints_dir, path)
    model.save(model,best_weights,opt,path,best_epoch)
    print('model saved')

if __name__ == "__main__":
    '''get the configuration file'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='yaml/patch_gan_edges/patch_gan_edges.yaml')
    sys.argv = ['-f']
    opt   = parser.parse_known_args()[0]

    '''load the configuration file and append to current parser arguments'''
    ydict = yaml.load(open(opt.config), Loader=yaml.FullLoader)
    for k,v in ydict.items():
        parser.add_argument('--'+str(k), required=False, default=v)
    opt   = parser.parse_args()

    '''adding seed for reproducibility'''
    torch.manual_seed(opt.seed)


    '''set device'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device


    '''Set the addition argument value for saving epoch images'''
    check=True
    for arg in vars(opt):
     if arg in ['addition']:
         check=False
    if check: opt.addition=False


    set_val_dir(opt)  #setting the training dataset dir
    set_train_dir(opt)  #setting the validation set dir

# '''load dataset (loading dataset based on dataset name and factor on arguments)'''
    if opt.edge_type in ['canny']:
        train_dataloader,eval_dataloader,train_datasets,val_datasets = load_dataset_edges(opt)
    elif opt.edge_type in ['downsample']:
        set_downsample_train_val_dir(opt)
        train_dataloader,eval_dataloader,train_datasets,val_datasets = load_dataset_downsample_edges(opt)

    print(opt.downsample_train_dir)

#    '''get the epoch image path to save the image output of every epoch for given single image'''
    if opt.patch:
        opt.epoch_image_path = '{}/z_axis/factor_{}/train/lr_f1_160_{}_z_46.png'.format(opt.dataset_size,opt.factor,opt.factor)
    else:
        if opt.edge_type in ['downsample']:
            print('save epoch image not implemented for downsample edges so using canny edges instead')
        opt.epoch_image_path = '{}/{}/factor_{}/train/lr_f1_160_{}_z_46.png'.format(opt.dataset_size,opt.dataset_name, opt.factor,opt.factor)


    '''load model'''
    if opt.pretrained_generator:
        opt.generator,growth_rate, num_blocks, num_layers = load_pretrained(checkpoint= opt.pretrained_checkpoint, device = opt.device)
        #set these parameters to save while saving the checkpoints
        opt.growth_rate = growth_rate
        opt.num_blocks = num_blocks
        opt.num_layers = num_layers
        print("Pre Trained Model Loaded")
    model = load_model(opt)
    

    # print(opt)

    '''setup the outputs and logging metric dirs on '''
    set_outputs_dir(opt) 
    set_training_metric_dir(opt) 
    set_plots_dir(opt)


    #   setting metric for evaluation
    psnr = PSNR()
    ssim = SSIM()
    opt.psnr = psnr.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)
    opt.ssim = ssim.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)


    '''wrap model for data parallelism'''
    num_of_gpus = torch.cuda.device_count()
    if num_of_gpus>1:
        model = nn.DataParallel(model,device_ids=[*range(num_of_gpus)])
        model = model.module
        opt.data_parallel = True


    print('training for factor ',opt.factor)

    metric_dict = LogMetric({'loss_D_fake': [],'loss_D_real': [],'loss_D': [],'loss_G_GAN': [],
    'loss_G_L1': [],'loss_G': [],'epoch':[]})

    '''wandb visualization'''
    if opt.wandb:
        wandb.init(
        project=opt.project_name,
                name = opt.exp_name,
                config = opt )

        wandb.watch(model.net_G,log="all",log_freq=len(train_datasets)/opt.train_batch_size)
        # wandb.watch(model.net_D,log="all",log_freq=50)
    else:
        wandb=None

    # print(model.net_G)
    train(opt,model,train_datasets,train_dataloader,eval_dataloader,wandb = wandb)

    if opt.wandb:
        wandb.unwatch(model)
        wandb.finish()
