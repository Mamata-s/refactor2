# process_yaml.py file
#imports
import yaml
import argparse
import sys
from utils.train_utils import load_model_3d,get_criterion,get_optimizer
import torch
import torch.nn as nn
import copy
from utils.logging_metric import LogMetric,create_loss_meters_srdense
from utils.train_utils import adjust_learning_rate
from utils.train_epoch import train_epoch_srdense,validate_srdense
from utils.preprocess import apply_model_using_cv
from utils.general import save_configuration_yaml,LogOutputs
from utils.config import set_outputs_dir,set_training_metric_dir,set_plots_dir
from dataset.dataset_3d import MRIDataset3DPatch
import os
import wandb

os.environ["CUDA_VISIBLE_DEVICES"]='0,1'

def train(opt,model,criterion,optimizer,train_datasets,train_dataloader,eval_dataloader,wandb=None):

    if opt.wandb:
        log_table_output = LogOutputs()

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    
    for epoch in range(opt.num_epochs):
        '''reduce learning rate by factor 0.5 on every 150 or 225 epoch'''
        opt.lr = adjust_learning_rate(optimizer, epoch,opt.lr)

        '''setting model in train mode'''
        model.train()

        '''train one epoch and evaluate the model'''
    
        epoch_losses = create_loss_meters_srdense()  #create a dictionary
        images,labels,preds = train_epoch_srdense(opt,model,criterion,optimizer,train_datasets,train_dataloader,epoch,epoch_losses)
        eval_loss, eval_l1,eval_psnr, eval_ssim,eval_hfen = validate_srdense(opt,model, eval_dataloader,criterion,addition=opt.addition)
        

        if opt.wandb:
            wandb.log({"val/val_loss" : eval_loss,
            "val/val_l1_error":eval_l1,
            "val/val_psnr": eval_psnr,
            "val/val_ssim":eval_ssim,
            "val/val_hfen":eval_hfen,
            "epoch": epoch,
            })
            for key in epoch_losses.keys():
                wandb.log({"train/{}".format(key) : epoch_losses[key].avg,
                })
            wandb.log({"other/learning_rate": opt.lr})
            
            # log_output_images(images, preds, labels) #overwrite on same table on every epoch
            print(images[0,0,0,:,:].squeeze().shape)
            if epoch % opt.n_freq == 0:
                log_table_output.append_list_3d(epoch,images[0,0,0,:,:].squeeze(),labels[0,0,0,:,:].squeeze(),preds[0,0,0,:,:].squeeze())  #create a class with list and function to loop through list and add to log table
        print('eval psnr: {:.4f}'.format(eval_psnr))

        if eval_psnr > best_psnr:
            best_epoch = epoch
            best_psnr = eval_psnr
            best_weights = copy.deepcopy(model.state_dict())

        '''adding to the dictionary'''
        metric_dict.update_dict([eval_loss,eval_l1,eval_psnr,eval_ssim,eval_hfen],training=False)

        
        metric_dict.update_dict([epoch_losses['train_loss'].avg])  

    if opt.wandb:
        log_table_output.log_images(columns = ["epoch","image", "pred", "label"],wandb=wandb)

    path = metric_dict.save_dict(opt)
    _ = save_configuration_yaml(opt)

    # path="best_weights_factor_{}_epoch_{}".format(opt.factor,best_epoch)
    # torch.save(best_weights, os.path.join(opt.checkpoints_dir, path))
    path="best_weights_factor_{}_epoch_{}.pth".format(opt.factor,best_epoch)
    path = os.path.join(opt.checkpoints_dir, path)
    model.module.save(best_weights,opt,path,optimizer.state_dict(),best_epoch)

    print('model saved')

def load_train_set(opt):
    train_datasets = MRIDataset3DPatch(opt.train_image_dir, opt.train_label_dir)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size = opt.train_batch_size,shuffle=True,
    num_workers=1,pin_memory=False,drop_last=False)
    return train_datasets,train_dataloader

def load_val_set(opt):
    val_datasets = MRIDataset3DPatch(opt.val_image_dir, opt.val_label_dir)
    val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size = opt.val_batch_size,shuffle=True,
    num_workers=1,pin_memory=False,drop_last=False)
    return val_datasets,val_dataloader


if __name__ == "__main__":
    '''get the configuration file'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='yaml/train_3d_rrdbnet.yaml')
    sys.argv = ['-f']
    opt   = parser.parse_known_args()[0]

    '''load the configuration file and append to current parser arguments'''
    ydict = yaml.load(open(opt.config), Loader=yaml.FullLoader)
    for k,v in ydict.items():
        if k=='config':
            pass
        else:
            parser.add_argument('--'+str(k), required=False, default=v)
    opt  = parser.parse_args()

    '''adding seed for reproducibility'''
    torch.manual_seed(opt.seed)

    '''set device'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device

    opt.train_image_dir = '3d_dataset50/factor_{}/train/'.format(opt.factor)
    opt.train_label_dir ='3d_dataset50/label/train/'
    opt.val_image_dir = '3d_dataset50/factor_{}/val/'.format(opt.factor)
    opt.val_label_dir ='3d_dataset50/label/val/'
    '''load dataset (loading dataset based on dataset name and factor on arguments)'''
    train_datasets,train_dataloader = load_train_set(opt)
    val_datasets,val_dataloader = load_val_set(opt)


    '''load model'''
    model = load_model_3d(opt)

    '''print model'''
    print(model)
    
    opt.patch=True
    opt.training_type = 'original_no_addition'

    '''setup the outputs and logging metric dirs on '''
    set_outputs_dir(opt) 
    set_training_metric_dir(opt) 
    set_plots_dir(opt)


    '''wrap model for data parallelism'''
    num_of_gpus = torch.cuda.device_count()
    if num_of_gpus>1:
        model = nn.DataParallel(model,device_ids=[*range(num_of_gpus)])
        opt.data_parallel = True

    '''setup loss and optimizer '''
    criterion = get_criterion(opt)
    optimizer = get_optimizer(opt,model)

    print('training for factor ',opt.factor)
    print(model)


    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    '''initialize the logging dictionary'''
    metric_dict = LogMetric( { 'train_loss' : [],'epoch':[]})


    if opt.wandb:
        wandb.init(
        project=opt.project_name,
                name = opt.exp_name,
                config = opt )

        wandb.watch(model,log="all",log_freq=1)

    else:
        wandb=None

    '''training the model'''
    train(opt,model,criterion,optimizer,train_datasets,train_dataloader,val_dataloader,wandb = wandb)

    if opt.wandb:
        wandb.unwatch(model)
        wandb.finish()