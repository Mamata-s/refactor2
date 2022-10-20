# process_yaml.py file

#imports
import yaml
import argparse
import sys
from utils.train_utils import get_criterion,get_optimizer
import torch
import torch.nn as nn
from utils.image_quality_assessment import PSNR,SSIM
import copy
from utils.logging_metric import LogMetric,create_loss_meters_srdense
from utils.train_utils import adjust_learning_rate
from utils.general import save_configuration_yaml,LogEdgesOutputs
from utils.config import set_outputs_dir,set_training_metric_dir,set_plots_dir,set_train_dir,set_val_dir
from dataset.dataset_cv import MRIDatasetCannyEdges,MRIDatasetHighFreqEdges
from models.densenet_edges import SRDenseNet
from utils.logging_metric import update_epoch_losses
import os
from utils.preprocess import hfen_error

from tqdm import tqdm

import wandb
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'



def train_epoch_edges_combined(opt,model,criterion,optimizer,train_dataset,train_dataloader,epoch,epoch_losses): 
    with tqdm(total=(len(train_dataset) - len(train_dataset) % opt.train_batch_size), ncols=80) as t:
        t.set_description('epoch: {}/{}'.format(epoch, opt.num_epochs - 1))

        for idx, (data) in enumerate(train_dataloader):
            images = data['image'].to(opt.device)
            labels = data['label'].to(opt.device)
            edges_lr = data['lr_edges'].to(opt.device)
            mask = data['mask'].to(opt.device)

            label_edges = labels-images

            pred_edg , pred_img = model(edges_lr, images)

        
        
            loss_edg = criterion(pred_edg, label_edges)
            loss_img = criterion(pred_img,labels)
            total_loss = loss_edg + loss_img

        
            update_epoch_losses(epoch_losses, count=len(images),values=[total_loss.item()])
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        t.set_postfix(loss='{:.6f}'.format(epoch_losses['train_loss'].avg))
        t.update(len(images))

        if epoch % opt.n_freq==0:
            if not os.path.exists(opt.checkpoints_dir):
                os.makedirs(opt.checkpoints_dir)
            path = os.path.join(opt.checkpoints_dir, 'epoch_{}_f_{}.pth'.format(epoch,opt.factor))
            if opt.data_parallel:
                model.module.save(model.state_dict(),opt,path,optimizer.state_dict(),epoch)
            else:
                model.save(model.state_dict(),opt,path,optimizer.state_dict(),epoch)
            
    return {'epoch':epoch,
            'hr': labels,
            'final_output':pred_img,
            'lr':images,
            'label_edges':label_edges,
            'pred_edges': pred_edg,
            'input_edges':edges_lr,
            'mask':mask
    }

def validate_edges_combined(opt,model, dataloader,criterion=nn.MSELoss()):
    model.eval()
    l1_loss = nn.L1Loss()
    count,psnr,ssim,loss,l1,hfen = 0,0,0,0,0,0
    with torch.no_grad():
        for data in dataloader:  #batch size is always 1 to calculate psnr and ssim
            image = data['image'].to(opt.device)
            label = data['label'].to(opt.device)
            edges_lr = data['lr_edges'].to(opt.device)
            
        
            output_edg, output_img = model(edges_lr, image)
            
            output = output_img.clamp(0.,1.)

            # print('output shape',output.shape)
            # print('label shape',label.shape)
            # print("*********************************")

            loss += criterion(output,label) 
            l1 += l1_loss(output,label)
            count += len(label)

            #psnr and ssim using tensor
            psnr += opt.psnr(output, label)
            ssim += opt.ssim(output,label)

            # old
            output = output.squeeze().detach().to('cpu').numpy()
            # image = image.squeeze().to('cpu').numpy()
            label = label.squeeze().detach().to('cpu').numpy()
            # psnr += peak_signal_noise_ratio(output, label)
            # ssim += structural_similarity(output, label)
            hfen += hfen_error(output, label)
    return loss.item()/count, l1.item()/count,psnr.item()/count, ssim.item()/count,hfen/count



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
        output_dict = train_epoch_edges_combined(opt,model,criterion,optimizer,train_datasets,train_dataloader,epoch,epoch_losses)
        eval_loss, eval_l1,eval_psnr, eval_ssim,eval_hfen = validate_edges_combined(opt,model, eval_dataloader,criterion)
    
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
    parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='yaml/combined_model_new/srdense.yaml')
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


    set_val_dir(opt)  #setting the training dataset dir
    set_train_dir(opt)  #setting the validation set dir

    # print(opt.train_image_dir)
    # print(opt.train_label_dir)
    # print(opt.val_image_dir)
    # print(opt.val_label_dir)
    # quit();

    train_datasets = MRIDatasetHighFreqEdges(opt.train_image_dir, opt.train_label_dir,threshold=opt.mask_threshold,apply_mask=opt.apply_mask, dir_dict=opt.dir_dict)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size = opt.train_batch_size,shuffle=True,
            num_workers=1,pin_memory=False,drop_last=False)


    val_datasets = MRIDatasetHighFreqEdges(opt.val_image_dir, opt.val_label_dir,threshold=opt.mask_threshold,apply_mask=opt.apply_mask, dir_dict=opt.val_dir_dict)
    eval_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size = 1,shuffle=True,
            num_workers=1,pin_memory=False,drop_last=False)



    '''load model'''
    model = SRDenseNet(num_channels=1, growth_rate = opt.growth_rate, num_blocks = opt.num_blocks, num_layers=opt.num_layers).to(opt.device)

    '''print model'''
    # print(model)

    # print(opt)


    '''setup the outputs and logging metric dirs on '''
    set_outputs_dir(opt) 
    set_training_metric_dir(opt) 
    set_plots_dir(opt)


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


    print('training for factor ',opt.factor)
    print('Model for training is given by')
    print(model)


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