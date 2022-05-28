
from tqdm import tqdm
from tensorboard_logger import configure, log_value
import torch
import os
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from utils.logging_metric import update_epoch_losses, update_losses, log_results
from utils.preprocess import hfen_error


# class TrainModel():
#     def __init__(self,model,opt):
#         self.model=model
#         self.opt=opt

#     def train(self):
#         if self.opt.model_name in ['srdense','unet','residual']:
#             self.train_srdense()

#     def train_epoch_srdense(self): 
#         pass

#     def train_srgan(self):
#         pass



# def train_and_eval(opt,model,criterion,optimizer,train_dataset,train_dataloader,eval_dataloader,epoch,epoch_losses):
#     if opt.model_name in ['srdense','unet','residual']:
#         train_epoch_srdense(opt,model,criterion,optimizer,train_dataset,train_dataloader,epoch,epoch_losses)
#         eval_loss, eval_l1,eval_psnr, eval_ssim,eval_hfen = validate_srdense(opt,model, eval_dataloader,criterion)
#         return eval_loss, eval_l1,eval_psnr, eval_ssim,eval_hfen

#     elif opt.model_name in ['gan','patch_gan']:
#         train_epoch_patch_gan(opt,model,optimizer,train_dataloader,epoch,epoch_losses)
#         eval_loss, eval_l1,eval_psnr, eval_ssim,eval_hfen = validate_patch_gan(opt,model, eval_dataloader,criterion)
#         return eval_loss, eval_l1,eval_psnr, eval_ssim,eval_hfen
#     else:
#         print('the model training not implemented')


def train_epoch_srdense(opt,model,criterion,optimizer,train_dataset,train_dataloader,epoch,epoch_losses): 
    with tqdm(total=(len(train_dataset) - len(train_dataset) % opt.batch_size), ncols=80) as t:
        t.set_description('epoch: {}/{}'.format(epoch, opt.num_epochs - 1))

        for idx, (images, labels) in enumerate(train_dataloader):
            images = images.to(opt.device)
            labels = labels.to(opt.device)
            preds = model(images)

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
            torch.save({
                    'epoch': epoch,
                    'growth_rate': opt.growth_rate,
                    'num_blocks':opt.num_blocks,
                    'num_layers':opt.num_layers,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, path)
            # torch.save(model.state_dict(), os.path.join(config['outputs_dir'], 'epoch_{}_f_{}.pth'.format(epoch,args.factor)))
    return images


def validate_srdense(opt,model, dataloader,criterion=nn.MSELoss()):
    model.eval()
    l1_loss = nn.L1Loss()
    count,psnr,ssim,loss,l1,hfen = 0,0,0,0,0,0
    with torch.no_grad():
        for image,label in dataloader:  #batch size is always 1 to calculate psnr and ssim
            image = image.to(opt.device)
            label = label.to(opt.device)
            output = model(image)
            loss += criterion(output,label) 
            l1 += l1_loss(output,label)
            count += len(label)
            output = output.clamp(0.0,1.0)
            output = output.squeeze().detach().to('cpu').numpy()
            # outputs = min_max_normalize(outputs)
            image = image.squeeze().to('cpu').numpy()
            label = label.squeeze().detach().to('cpu').numpy()

            psnr += peak_signal_noise_ratio(output, label)
            ssim += structural_similarity(output, label)
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

            output = output.squeeze().detach().to('cpu').numpy()
            # outputs = min_max_normalize(outputs)
            image = image.squeeze().to('cpu').numpy()
            label = label.squeeze().detach().to('cpu').numpy()

            psnr += peak_signal_noise_ratio(output, label)
            ssim += structural_similarity(output, label)
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
        torch.save({
                'epoch': epoch,
                'n_blocks': opt.n_blocks,
                'start_filters': opt.start_filters,
                'activation':opt.activation,
                'normalization':opt.normalization,
                'model_state_dict': model.state_dict(),
                'conv_mode':opt.conv_mode,
                'dim':opt.dim,
                'up_mode':opt.up_mode,
                'init':opt.init,
                'gan_mode': opt.gan_mode,
                'n_down':opt.n_down,
                'num_filers':opt.num_filters,
                'g_optimizer_state_dict': model.opt_G.state_dict(),
                'd_optimizer_state_dict': model.opt_D.state_dict(),
                }, path)
        # path = os.path.join(opt.checkpoints_dir, 'full_model_epoch_{}_f_{}.pth'.format(epoch,opt.factor))
        # torch.save(model.state_dict(), path)
    return images

