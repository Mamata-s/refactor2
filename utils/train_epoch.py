
from tqdm import tqdm
from tensorboard_logger import configure, log_value
import torch
import os
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from utils.general import min_max_normalize
from utils.logging_metric import update_epoch_losses, update_losses, log_results
from utils.preprocess import hfen_error



def train_epoch_edges(opt,model,criterion,optimizer,train_dataset,train_dataloader,epoch,epoch_losses,loss_type='addition'): 
    with tqdm(total=(len(train_dataset) - len(train_dataset) % opt.train_batch_size), ncols=80) as t:
        t.set_description('epoch: {}/{}'.format(epoch, opt.num_epochs - 1))

        for idx, (data) in enumerate(train_dataloader):
            images = data['image'].to(opt.device)
            labels = data['label'].to(opt.device)
            edges_lr = data['lr_edges'].to(opt.device)
            mask = data['mask'].to(opt.device)

            label_error_map = labels-images
            preds = model(edges_lr)

            if opt.apply_mask:
                label_error_map_mask = mask * label_error_map
                preds_mask = mask * preds

                outs_m = preds_mask + images
                output = outs_m * mask
                
                if loss_type in ['addition']:
                    loss = criterion(output, labels)
                else:
                    loss = criterion(preds_mask, label_error_map_mask)

            else:
                output = preds+images
                if loss_type in ['addition']:
                    loss = criterion(output, labels)
                else:
                    loss = criterion(preds, label_error_map)

        
            update_epoch_losses(epoch_losses, count=len(images),values=[loss.item()])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t.set_postfix(loss='{:.6f}'.format(epoch_losses['train_loss'].avg))
        t.update(len(images))

        if epoch % opt.n_freq==0:
            if not os.path.exists(opt.checkpoints_dir):
                os.makedirs(opt.checkpoints_dir)
            path = os.path.join(opt.checkpoints_dir, 'epoch_{}_f_{}.pth'.format(epoch,opt.factor))
            model.module.save(model.state_dict(),opt,path,optimizer.state_dict(),epoch)
            # torch.save(model.state_dict(), os.path.join(config['outputs_dir'], 'epoch_{}_f_{}.pth'.format(epoch,args.factor)))
    return {'epoch':epoch,
            'hr': labels,
            'final_output':output,
            'lr':images,
            'label_edges':label_error_map,
            'pred_edges': preds,
            'input_edges':edges_lr,
            'mask':mask
    }


def validate_edges(opt,model, dataloader,criterion=nn.MSELoss()):
    model.eval()
    l1_loss = nn.L1Loss()
    count,psnr,ssim,loss,l1,hfen = 0,0,0,0,0,0
    with torch.no_grad():
        for data in dataloader:  #batch size is always 1 to calculate psnr and ssim
            image = data['image'].to(opt.device)
            label = data['label'].to(opt.device)
            edges_lr = data['lr_edges'].to(opt.device)
            label_edges = label-image
            output = model(edges_lr)

            output = output+image

            loss += criterion(output,label) 
            l1 += l1_loss(output,label)
            count += len(label)
            output = output.clamp(0.0,1.0)

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



def train_epoch_srdense(opt,model,criterion,optimizer,train_dataset,train_dataloader,epoch,epoch_losses): 
    with tqdm(total=(len(train_dataset) - len(train_dataset) % opt.train_batch_size), ncols=80) as t:
        t.set_description('epoch: {}/{}'.format(epoch, opt.num_epochs - 1))

        for idx, (images, labels) in enumerate(train_dataloader):
            images = images.to(opt.device)
            labels = labels.to(opt.device)
            preds = model(images)

            if opt.training_type == "error_map":
                label_error_map = labels-images
                loss = criterion(preds, label_error_map)
            elif opt.training_type=="addition":
                preds = preds+images
                loss = criterion(preds, labels)
            else:
                loss = criterion(preds, labels)

            # epoch_losses.update(loss.item(), len(images))
            update_epoch_losses(epoch_losses, count=len(images),values=[loss.item()])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t.set_postfix(loss='{:.6f}'.format(epoch_losses['train_loss'].avg))
        t.update(len(images))

        if epoch % opt.n_freq==0:
            if not os.path.exists(opt.checkpoints_dir):
                os.makedirs(opt.checkpoints_dir)
            path = os.path.join(opt.checkpoints_dir, 'epoch_{}_f_{}.pth'.format(epoch,opt.factor))
            model.module.save(model.state_dict(),opt,path,optimizer.state_dict(),epoch)
    return images,labels,preds

def validate_srdense_3d(opt,model, dataloader,criterion=nn.MSELoss(),addition=False):
    model.eval()
    l1_loss = nn.L1Loss()
    count,psnr,ssim,loss,l1,hfen = 0,0,0,0,0,0
    with torch.no_grad():
        for image,label in dataloader:  #batch size is always 1 to calculate psnr and ssim
            image = image.to(opt.device)
            label = label.to(opt.device)
            output = model(image)

            if addition:
                output = output+image


            for i in range(output.shape[2]):
                loss += criterion(output[:,:,i,:,:],label[:,:,i,:,:]) 
                l1 += l1_loss(output[:,:,i,:,:],label[:,:,i,:,:])
                count += 1
                output = output.clamp(0.0,1.0)

                #psnr ssim using tensor
                psnr += opt.psnr(output[:,:,i,:,:], label[:,:,i,:,:]).item()
                ssim += opt.ssim(output[:,:,i,:,:],label[:,:,i,:,:]).item()

                # old
                output = output[:,:,i,:,:].squeeze().detach().to('cpu').numpy()
                # image = image[:,:,i,:,:].squeeze().to('cpu').numpy()
                label = label[:,:,i,:,:].squeeze().detach().to('cpu').numpy()
                # psnr += peak_signal_noise_ratio(output, label)
                # ssim += structural_similarity(output, label)
                hfen += hfen_error(output, label)

    return loss.item()/count, l1.item()/count,psnr.item()/count, ssim.item()/count,hfen.item()/count




def validate_srdense(opt,model, dataloader,criterion=nn.MSELoss(),addition=False):
    model.eval()
    l1_loss = nn.L1Loss()
    count,psnr,ssim,loss,l1,hfen = 0,0,0,0,0,0
    with torch.no_grad():
        for image,label in dataloader:  #batch size is always 1 to calculate psnr and ssim
            image = image.to(opt.device)
            label = label.to(opt.device)
            output = model(image)

            if addition:
                output = output+image

            loss += criterion(output,label) 
            l1 += l1_loss(output,label)
            count += len(label)
            output = output.clamp(0.0,1.0)

            #psnr and ssim using tensor
            psnr += opt.psnr(output, label).item()
            ssim += opt.ssim(output,label).item()


            output = output.squeeze().detach().to('cpu').numpy()
            # image = image.squeeze().to('cpu').numpy()
            label = label.squeeze().detach().to('cpu').numpy()
            # psnr += peak_signal_noise_ratio(output, label)
            # ssim += structural_similarity(output, label)
            hfen += hfen_error(output, label)
    return loss.item()/count, l1.item()/count,psnr.item()/count, ssim.item()/count,hfen.item()/count


def validate_patch_gan(opt,model, dataloader,criterion=nn.MSELoss()):
    model.eval()
    l1_loss = nn.L1Loss()
    count,psnr,ssim,loss,l1,hfen = 0,0,0,0,0,0
    with torch.no_grad():
        for image,label in dataloader:
            image = image.to(opt.device)
            label = label.to(opt.device)
    
            output = model.net_G(image)
            loss += criterion(output,label) 
            l1 += l1_loss(output,label)

            count += len(label)
            output = output.clamp(0.0,1.0)

            #psnr and ssim using tensor
            psnr += opt.psnr(output, label).item()
            ssim += opt.ssim(output,label).item()

            output = output.squeeze().detach().to('cpu').numpy()
            # image = image.squeeze().to('cpu').numpy()
            label = label.squeeze().detach().to('cpu').numpy()
            # psnr += peak_signal_noise_ratio(output, label)
            # ssim += structural_similarity(output, label)
            hfen += hfen_error(output, label)
    return loss.item()/count, l1.item()/count,psnr.item()/count, ssim.item()/count,hfen.item()/count
    
def train_epoch_patch_gan(opt,model,train_dl, epoch, epoch_losses):
    # model = model.module
    for images,labels in tqdm(train_dl):
        model.setup_input(images,labels) 
        model.optimize()

        update_losses(model, epoch_losses, count=images.size(0)) # not implemented   
    log_results(epoch_losses) # function to print out the losses
        
    if epoch %  opt.n_freq==0:
        if not os.path.exists(opt.checkpoints_dir):
                os.makedirs(opt.checkpoints_dir)
        path = os.path.join(opt.checkpoints_dir, 'epoch_{}_f_{}.pth'.format(epoch,opt.factor))
        model.save(model.state_dict(),opt,path,epoch)
    with torch.no_grad():
        preds = model.net_G(images.to(opt.device)).to(opt.device)
    return images, preds, labels


