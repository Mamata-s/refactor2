'''
opt.datset_size=25/50
opt.datset_name=full/z_axis/patch/mix
opt.factor=2/4/6
opt.model_name=original/unet/srdensenet
opt.patch_size=64/96
opt.patch=True

'''

#datset path for corresponding dataset_name and dataset_size
def set_train_dir(opt):
    if opt.patch:
        set_patch_train_dir(opt)
    else:
        opt.train_image_dir = '{}/{}/factor_{}/train'.format(opt.dataset_size,opt.dataset_name, opt.factor)
        opt.train_label_dir = '{}/{}/label/train'.format(opt.dataset_size,opt.dataset_name)
    return 1

def set_val_dir(opt=None,dataset_name=None,factor=None,dataset_size=None):
    if opt:
        if opt.patch:
            set_patch_val_dir(opt)
        else:
            opt.val_image_dir = '{}/{}/factor_{}/val'.format(opt.dataset_size,opt.dataset_name, opt.factor)
            opt.val_label_dir = '{}/{}/label/val'.format(opt.dataset_size,opt.dataset_name)
        return 1
    else:
        val_image_dir = '{}/{}/factor_{}/val'.format(dataset_size,dataset_name,factor)
        val_label_dir = '{}/{}/label/val'.format(dataset_size,dataset_name) 
        return val_image_dir,val_label_dir


def set_patch_train_dir(opt):
    opt.train_image_dir = '{}/patch/patch-{}/factor_{}/train'.format(opt.dataset_size,opt.patch_size, opt.factor)
    opt.train_label_dir ='{}/patch/patch-{}/factor_{}/label/train'.format(opt.dataset_size,opt.patch_size, opt.factor)

def set_patch_val_dir(opt):
    opt.val_image_dir= '{}/patch/patch-{}/factor_{}/val'.format(opt.dataset_size,opt.patch_size,opt.factor)
    opt.val_label_dir='{}/patch/patch-{}/factor_{}/label/val'.format(opt.dataset_size,opt.patch_size,opt.factor)



# outputs paths(checkpoints, epoch images,input_batch_images and output batch images) for corresponding datset_name_size_factor_model_name
def set_outputs_dir(opt):
    if opt.patch:
        opt.checkpoints_dir = 'outputs/{}/{}/{}/checkpoints/patch/patch-{}/factor_{}/'.format(opt.dataset_size,opt.model_name,opt.exp_name, opt.patch_size,opt.factor)
        opt.epoch_images_dir ='outputs/{}/{}/{}/epoch_images/patch/patch-{}/factor_{}/'.format(opt.dataset_size,opt.model_name,opt.exp_name, opt.patch_size,opt.factor)
        opt.input_images_dir ='outputs/{}/{}/{}/input_images/patch/patch-{}/factor_{}/'.format(opt.dataset_size,opt.model_name, opt.exp_name, opt.patch_size,opt.factor)
        opt.output_images_dir ='outputs/{}/{}/{}/output_images/patch/patch-{}/factor_{}/'.format(opt.dataset_size,opt.model_name, opt.exp_name, opt.patch_size,opt.factor) 
    else:
        opt.checkpoints_dir = 'outputs/{}/{}/{}/checkpoints/{}/factor_{}/'.format(opt.dataset_size,opt.model_name,opt.exp_name, opt.dataset_name,opt.factor)
        opt.epoch_images_dir ='outputs/{}/{}/{}/epoch_images/{}/factor_{}/'.format(opt.dataset_size,opt.model_name,opt.exp_name, opt.dataset_name,opt.factor)
        opt.input_images_dir ='outputs/{}/{}/{}/input_images/{}/factor_{}/'.format(opt.dataset_size,opt.model_name, opt.exp_name,opt.dataset_name,opt.factor)
        opt.output_images_dir ='outputs/{}/{}/{}/output_images/{}/factor_{}/'.format(opt.dataset_size,opt.model_name,opt.exp_name, opt.dataset_name,opt.factor)

# training metric paths
def set_training_metric_dir(opt):
    if opt.patch:
        opt.loss_dir = 'outputs/{}/{}/{}/losses/patch/patch-{}/factor_{}/'.format(opt.dataset_size,opt.model_name, opt.exp_name, opt.patch_size,opt.factor)
        opt.grad_norm_dir ='outputs/{}/{}/{}/grad_norm/patch/patch-{}/factor_{}/'.format(opt.dataset_size,opt.model_name,opt.exp_name, opt.patch_size,opt.factor)
    else:
        opt.loss_dir = 'outputs/{}/{}/{}/losses/{}/factor_{}/'.format(opt.dataset_size,opt.model_name,opt.exp_name, opt.dataset_name,opt.factor)
        opt.grad_norm_dir ='outputs/{}/{}/{}/grad_norm/{}/factor_{}/'.format(opt.dataset_size,opt.model_name,opt.exp_name, opt.dataset_name,opt.factor)


#plots path
def set_plots_dir(opt):
    if opt.patch:
        opt.plot_dir = 'outputs/{}/{}/{}/plots/patch/patch-{}/factor_{}/'.format(opt.dataset_size,opt.model_name, opt.exp_name,opt.patch_size,opt.factor)
    else:
        opt.plot_dir = 'outputs/{}/{}/{}/plots/{}/factor_{}/'.format(opt.dataset_size,opt.model_name, opt.exp_name,opt.dataset_name,opt.factor)




