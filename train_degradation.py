# process_yaml.py file
#imports
import yaml
import argparse
import sys
from utils.train_utils import load_dataset,load_model,get_criterion,get_optimizer
import torch
import torch.nn as nn
from utils.image_quality_assessment import PSNR,SSIM
import copy
from utils.logging_metric import LogMetric,create_loss_meters_srdense
from utils.train_utils import adjust_learning_rate, read_dictionary
from utils.train_epoch import train_epoch_srdense,validate_srdense
from utils.preprocess import apply_model,apply_model_using_cv
from utils.general import save_configuration,save_configuration_yaml,LogOutputs
from utils.config import set_outputs_dir,set_training_metric_dir,set_plots_dir,set_val_dir,set_train_dir
import os
import wandb
from dataset.dataset_cv import MRIDataset
import json
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'

def train_with_no_eval(opt,model,criterion,optimizer,train_datasets,train_dataloader,wandb=None):

    if opt.wandb:
        log_table_output = LogOutputs()  
    
    for epoch in range(opt.num_epochs):
        '''reduce learning rate by factor 0.5 on every 150 or 225 epoch'''
        opt.lr = adjust_learning_rate(optimizer, epoch,opt.lr,opt.lr_decay_factor)

        '''setting model in train mode'''
        model.train()

        '''train one epoch and evaluate the model'''
    
        epoch_losses = create_loss_meters_srdense()  #create a dictionary
        images,labels,preds = train_epoch_srdense(opt,model,criterion,optimizer,train_datasets,train_dataloader,epoch,epoch_losses)
        

        if opt.wandb:
            for key in epoch_losses.keys():
                wandb.log({"train/{}".format(key) : epoch_losses[key].avg,
                })
            wandb.log({"other/learning_rate": opt.lr})
            
            if epoch % opt.n_freq == 0:
                print('images shape',images.shape)
                log_table_output.append_list(epoch,images,labels,preds)  #create a class with list and function to loop through list and add to log table
        
        metric_dict.update_dict([epoch_losses['train_loss'].avg])  

    if opt.wandb:
        log_table_output.log_images(columns = ["epoch","image", "pred", "label"],wandb=wandb)

    path = metric_dict.save_dict(opt)
    _ = save_configuration_yaml(opt)

    print('model saved')


if __name__ == "__main__":
    '''get the configuration file'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, 
    default='yaml/srdense_degradation.yaml')
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
    # torch.manual_seed(opt.seed)

    '''set device'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device
   


    '''Set the addition argument value for saving epoch images'''
    check=True
    for arg in vars(opt):
     if arg in ['addition']:
         check=False
    if check: opt.addition=False

    '''Set the addition argument value for edges training'''
    opt.edges_training=False


    opt.train_label_dir ='upsample/f1_160/upsample_image'
    opt.train_image_dir = 'upsample/f1_160/hr_25_h1/z_axis/'
    opt.dir_dict = 'upsample/f1_160/annotation_hr1_dict.pkl'
    print(opt.train_image_dir)
    print(opt.train_label_dir)
    # quit();


    '''load dataset (loading dataset based on dataset name and factor on arguments)'''
    train_datasets = MRIDataset(image_dir= opt.train_image_dir, label_dir = opt.train_label_dir,size=opt.size, dir_dict=opt.dir_dict if opt.dir_dict else None)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size = opt.train_batch_size,shuffle=False,
            num_workers=8,pin_memory=False,drop_last=False)


    '''load model'''
    model = load_model(opt)

    '''print model'''
    print(model)

    '''setup the outputs and logging metric dirs on '''
    opt.checkpoints_dir = 'outputs/degradation/checkpoints/'
    opt.loss_dir = 'outputs/degradation/loss/'
    opt.grad_norm_dir ='outputs/degradation/grad_norm/'
    opt.plot_dir = 'outputs/degradation/plots/'

    
    '''wrap model for data parallelism'''
    num_of_gpus = torch.cuda.device_count()
    print("Number of GPU available", num_of_gpus)
    if num_of_gpus>1:
        model = nn.DataParallel(model,device_ids=[*range(num_of_gpus)])
        opt.data_parallel = True
        print("Multiple GPU Training")

    '''setup loss and optimizer '''
    criterion = get_criterion(opt)
    optimizer = get_optimizer(opt,model)


    '''setup a output strategy based on training type'''
    if opt.training_type in ['addition','error_map']:
        opt.addition=True
    else:
        opt.addition=False


    print('training for factor ',opt.factor)
    # print(model)

    '''initialize the logging dictionary'''
    metric_dict = LogMetric( { 'train_loss' : [],'epoch':[]})

    #setting metric for evaluation
    psnr = PSNR()
    ssim = SSIM()
    opt.psnr = psnr.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)
    opt.ssim = ssim.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)


    if opt.wandb:
        wandb.init(
        project=opt.project_name,
                name = opt.exp_name,
                config = opt )

        wandb.watch(model,log="all",log_freq=1)

    else:
        wandb=None

    '''training the model'''
    train_with_no_eval(opt,model,criterion,optimizer,train_datasets,train_dataloader,wandb = wandb)

    if opt.wandb:
        wandb.unwatch(model)
        wandb.finish()