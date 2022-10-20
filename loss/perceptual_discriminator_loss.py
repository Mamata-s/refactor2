import torch.nn as nn
import torch

class DiscriminatorPerceptualLoss(nn.Module):
    ''' Measure a feature loss of a real and fake images
    this loss mesaures the feature loss from three different layers of patch discriminator
    we measure the mse loss between the features
    input format: [batch, channel, x,y]
    output: avg loss (either measure features loss from each layer for all image in a batch at once or calculate seperately and average at last)
    '''
    def __init__(self, avg_batch=True):
        super(DiscriminatorPerceptualLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.avg_batch = avg_batch

    def forward(self, discriminator,x, y):
        # Get batch_size
        batch_size = x.shape[0]
        
        # Initialize loss vector
        loss = torch.zeros([batch_size]).to(x.device)

        # Choose discriminator
        x_feat = discriminator(x,return_feature=True)
        y_feat = discriminator(y,return_feature=True)

        if self.avg_batch:
            loss = torch.zeros([batch_size]).to(x.device) # calculate summation of all level features loss for each image in a batch seperately and avg at last
            for batch_item_idx in range(batch_size):
                for scale_idx in range(len(x_feat)):
                    loss[batch_item_idx] += self.mse(x_feat[scale_idx][batch_item_idx],y_feat[scale_idx][batch_item_idx]) #measure feature loss for each image in batch seperately and average at last

        else:
            loss = torch.zeros([len(x_feat)]).to(x.device) # calculate loss for each feature level and average at last
            for scale_idx in range(len(x_feat)):
                loss[scale_idx] += self.mse(x_feat[scale_idx],y_feat[scale_idx])

        return torch.mean(loss)