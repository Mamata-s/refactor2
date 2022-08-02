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
from utils.train_utils import adjust_learning_rate,load_dataset_edges,load_dataset_downsample_edges
from utils.train_epoch import train_epoch_edges,  validate_edges
from utils.preprocess import apply_model_edges, apply_model_using_cv
from utils.general import save_configuration,save_configuration_yaml,LogEdgesOutputs
from utils.config import set_outputs_dir,set_training_metric_dir,set_plots_dir,set_train_dir,set_val_dir,set_downsample_train_val_dir
import os
import wandb
os.environ["CUDA_VISIBLE_DEVICES"]='0'

def train(opt,model,criterion,optimizer,train_datasets,train_dataloader,eval_dataloader,wandb=None):

    if opt.wandb:
        log_table_output = LogEdgesOutputs()

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    
    for epoch in range(opt.num_epochs):
        '''reduce learning rate by factor 0.5 on every 150 or 225 epoch'''
        opt.lr = adjust_learning_rate(optimizer, epoch,opt.lr,opt.lr_decay_factor)

        '''setting model in train mode'''
        model.train()

        '''train one epoch and evaluate the model'''
    
        epoch_losses = create_loss_meters_srdense()  #create a dictionary
        output_dict = train_epoch_edges(opt,model,criterion,optimizer,train_datasets,train_dataloader,epoch,epoch_losses,loss_type=opt.loss_type)
        eval_loss, eval_l1,eval_psnr, eval_ssim,eval_hfen = validate_edges(opt,model, eval_dataloader,criterion)
    
        # apply_model_edges(model,epoch,opt)

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
            wandb.log({"other/learning_rate": opt.lr})
            
            # log_output_images(images, preds, labels) #overwrite on same table on every epoch
            if epoch % opt.n_freq == 0:
                log_table_output.append_list(output_dict)  #create a class with list and function to loop through list and add to log table
        print('eval psnr: {:.4f}'.format(eval_psnr))

        if eval_psnr > best_psnr:
            best_epoch = epoch
            best_psnr = eval_psnr
            best_weights = copy.deepcopy(model.state_dict())

        '''adding to the dictionary'''
        metric_dict.update_dict([eval_loss,eval_l1,eval_psnr,eval_ssim,eval_hfen],training=False)

        
        metric_dict.update_dict([epoch_losses['train_loss'].avg])  

    if opt.wandb:
        if opt.apply_mask:
            log_table_output.log_images_and_mask(wandb=wandb)
        else:
            log_table_output.log_images(wandb=wandb)

    path = metric_dict.save_dict(opt)
    _ = save_configuration_yaml(opt)

    path="best_weights_factor_{}_epoch_{}.pth".format(opt.factor,best_epoch)
    path = os.path.join(opt.checkpoints_dir, path)
    # model.save(best_weights,opt,path,optimizer.state_dict(),best_epoch)
    model.module.save(best_weights,opt,path,optimizer.state_dict(),best_epoch)
    print('model saved')

if __name__ == "__main__":
    '''get the configuration file'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='yaml/factor_4/dense_z_axis25_mask_training_original_f4.yaml')
    # default='yaml/mask_training/canny_edges_original_f4_zaxis25.yaml'
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

    # '''Set the addition argument value for saving epoch images'''
    # check=True
    # for arg in vars(opt):
    #  if arg in ['addition']:
    #      check=False
    # if check: opt.addition=False

    # '''Set the addition argument value for edges training'''
    # check=True
    # for arg in vars(opt):
    #  if arg in ['edges_training']:
    #      check=False
    # if check: opt.edges_training=False


    set_val_dir(opt)  #setting the training dataset dir
    set_train_dir(opt)  #setting the validation set dir

    '''load dataset (loading dataset based on dataset name and factor on arguments)'''
    if opt.edge_type in ['canny']:
        train_dataloader,eval_dataloader,train_datasets,val_datasets = load_dataset_edges(opt)
    elif opt.edge_type in ['downsample']:
        set_downsample_train_val_dir(opt)
        train_dataloader,eval_dataloader,train_datasets,val_datasets = load_dataset_downsample_edges(opt)

    print(opt.downsample_train_dir)
    # quit();

    '''get the epoch image path to save the image output of every epoch for given single image'''
    if opt.patch:
        opt.epoch_image_path = '{}/z_axis/factor_{}/train/lr_f1_160_{}_z_46.png'.format(opt.dataset_size,opt.factor,opt.factor)
    else:
        if opt.edge_type in ['downsample']:
            print('save epoch image not implemented for downsample edges so using canny edges instead')
        opt.epoch_image_path = '{}/{}/factor_{}/train/lr_f1_160_{}_z_46.png'.format(opt.dataset_size,opt.dataset_name, opt.factor,opt.factor)

    '''load model'''
    model = load_model(opt)

    '''print model'''
    print(model)

    print(opt)

    '''setup the outputs and logging metric dirs on '''
    set_outputs_dir(opt) 
    set_training_metric_dir(opt) 
    set_plots_dir(opt)

    # print(opt)
    # quit();

    #setting metric for evaluation
    psnr = PSNR()
    ssim = SSIM()
    opt.psnr = psnr.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)
    opt.ssim = ssim.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)


    '''wrap model for data parallelism'''
    num_of_gpus = torch.cuda.device_count()
    print('Num of GPU available', num_of_gpus)
    opt.data_parallel = False
    if num_of_gpus>1:
        model = nn.DataParallel(model,device_ids=[*range(num_of_gpus)])
        opt.data_parallel = True

    '''setup loss and optimizer '''
    criterion = get_criterion(opt)
    optimizer = get_optimizer(opt,model)


    '''setup a output strategy based on training type'''
    if opt.training_type in ['addition','error_map']:
        opt.addition= True
    else:
        opt.addition=False


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

    # print(opt)
    # quit();

    '''training the model'''
    train(opt,model,criterion,optimizer,train_datasets,train_dataloader,eval_dataloader,wandb = wandb)

    if opt.wandb:
        wandb.unwatch(model)
        wandb.finish()