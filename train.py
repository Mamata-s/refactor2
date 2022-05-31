# process_yaml.py file
#imports
import yaml
import argparse
import sys
from utils.train_utils import load_dataset,load_model,get_criterion,get_optimizer
import torch
import torch.nn as nn
import copy
from utils.logging_metric import LogMetric,create_loss_meters_srdense
from utils.train_utils import adjust_learning_rate
from utils.train_epoch import train_epoch_srdense,validate_srdense
from utils.preprocess import apply_model
from utils.general import save_configuration,LogOutputs
from utils.config import set_outputs_dir,set_training_metric_dir,set_plots_dir
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
        '''reduce learning rate by factor 0.1 on every 150 or 225 epoch'''
        opt.lr = adjust_learning_rate(optimizer, epoch,opt.lr)

        '''setting model in train mode'''
        model.train()

        '''train one epoch and evaluate the model'''
    
        epoch_losses = create_loss_meters_srdense()  #create a dictionary
        images,labels,preds = train_epoch_srdense(opt,model,criterion,optimizer,train_datasets,train_dataloader,epoch,epoch_losses)
        eval_loss, eval_l1,eval_psnr, eval_ssim,eval_hfen = validate_srdense(opt,model, eval_dataloader,criterion,addition=opt.addition)
        
        apply_model(model,epoch,opt,addition= opt.addition)

        if opt.wandb:
            wandb.log({"val/val_loss" : eval_loss,
            "val/val_l1_error":eval_l1,
            "val/val_psnr": eval_psnr,
            "val/val_ssim":eval_ssim,
            "val/val_hfen":eval_hfen,
            })
            for key in epoch_losses.keys():
                wandb.log({"train/{}".format(key) : epoch_losses[key].avg,
                })
            wandb.log({"other/learning_rate": opt.lr})
            
            # log_output_images(images, preds, labels) #overwrite on same table on every epoch
            if epoch % opt.n_freq == 0:
                log_table_output.append_list(epoch,images,labels,preds)  #create a class with list and function to loop through list and add to log table

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
    _ = save_configuration(opt)

    path="best_weights_factor_{}_epoch_{}".format(opt.factor,best_epoch)
    torch.save(best_weights, os.path.join(opt.checkpoints_dir, path))

    print('model saved')

    # if opt.wandb:
    #     if opt.data_parallel:
    #         torch.onnx.export(model.module,images,"model.onnx")
    #     else:
    #         torch.onnx.export(model,images,"model.onnx")  
    #     wandb.save("model.onnx")


if __name__ == "__main__":
    '''get the configuration file'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='yaml/resunet.yaml')
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

    '''load dataset (loading dataset based on dataset name and factor on arguments)'''
    train_dataloader,eval_dataloader,train_datasets,val_datasets = load_dataset(opt)

    '''get the epoch image path to save the image output of every epoch for given single image'''
    opt.epoch_image_path = '{}/{}/factor_{}/train/lr_f1_160_{}_z_46.png'.format(opt.dataset_size,opt.dataset_name, opt.factor,opt.factor)

    '''load model'''
    model = load_model(opt)

    '''print model'''
    print(model)

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


    '''setup a output strategy based on training type'''
    if opt.training_type in ['addition','error_map']:
        opt.addition=True
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

    '''training the model'''
    train(opt,model,criterion,optimizer,train_datasets,train_dataloader,eval_dataloader,wandb = wandb)

    if opt.wandb:
        wandb.unwatch(model)
        wandb.finish()