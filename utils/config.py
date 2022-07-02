'''
opt.datset_size=25/50
opt.datset_name=full/z_axis/patch/mix
opt.factor=2/4/6
opt.model_name=original/unet/srdensenet
opt.patch_size=64/96
opt.patch=True

'''

#datset path for corresponding dataset_name and dataset_size
def get_train_dir(dataset_size,dataset_name,factor):
    train_image_dir = '{}/{}/factor_{}/train'.format(dataset_size,dataset_name, factor)
    train_label_dir = '{}/{}/label/train'.format(dataset_size,dataset_name)
    return train_image_dir,train_label_dir

def get_patch_train_dir(dataset_size,patch_size,factor):
    train_image_dir = '{}/patch/patch-{}/factor_{}/train'.format(dataset_size,patch_size, factor)
    train_label_dir ='{}/patch/patch-{}/label/train'.format(dataset_size,patch_size, factor)
    return train_image_dir, train_label_dir

def set_train_dir(opt):
    if opt.patch:
        opt.train_image_dir,opt.train_label_dir = get_patch_train_dir(opt.dataset_size,opt.patch_size,opt.factor)
    else:
        opt.train_image_dir,opt.train_label_dir = get_train_dir(opt.dataset_size,opt.dataset_name,opt.factor)
    return 1


def set_downsample_train_val_dir(opt):
    set_downsample_train_dir(opt)
    set_downsample_val_dir(opt)


def set_downsample_train_dir(opt):
    if opt.patch:
        opt.downsample_train_dir,_ = get_patch_train_dir(opt.dataset_size,opt.patch_size,opt.factor+2)
    else:
        opt.downsample_train_dir,_ = get_train_dir(opt.patch,opt.dataset_size,opt.dataset_name,opt.factor+2)
    return 1

def set_downsample_val_dir(opt):
    if opt.patch:
        opt.downsample_val_dir,_ = get_patch_val_dir(dataset_size=opt.dataset_size,patch_size=opt.patch_size,factor=opt.factor+2)       
    else:
        opt.downsample_val_dir,_ = get_val_dir(dataset_name=opt.dataset_name,factor=opt.factor+2,dataset_size=opt.dataset_size)
    return 1  

def set_val_dir(opt=None,dataset_name=None,factor=None,dataset_size=None):
    if opt:
        if opt.patch:
            opt.val_image_dir,opt.val_label_dir = get_patch_val_dir(opt.dataset_size,opt.factor)
        else:
            opt.val_image_dir,opt.val_label_dir = get_val_dir(opt.dataset_name,opt.factor,opt.dataset_size)
        return 1
    else:
        val_image_dir,val_label_dir = get_val_dir(dataset_name,factor,dataset_size)
        return val_image_dir,val_label_dir

def get_val_dir(dataset_name=None,factor=None,dataset_size=None):
    val_image_dir = '{}/{}/factor_{}/val'.format(dataset_size,dataset_name,factor)
    val_label_dir = '{}/{}/label/val'.format(dataset_size,dataset_name) 
    return val_image_dir,val_label_dir

def get_patch_val_dir(dataset_size,factor=2):
    # val_image_dir= '{}/patch/patch-{}/factor_{}/val'.format(dataset_size,patch_size,factor)
    # val_label_dir='{}/patch/patch-{}/label/val'.format(dataset_size,patch_size,factor)
    val_image_dir = '{}/z_axis/factor_{}/val'.format(dataset_size,factor)
    val_label_dir = '{}/z_axis/label/val'.format(dataset_size) 
    return val_image_dir,val_label_dir


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

