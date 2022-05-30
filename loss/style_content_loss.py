# https://github.com/ceshine/fast-neural-style/blob/201707/style-transfer.ipynb

from collections import namedtuple
import torchvision.models.vgg as vgg
import torch
import torch.nn as nn

LossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram 


class FinalStyleContentLoss(nn.Module):
  def __init__(self, style_weight =10000.0,content_weight=1.0,feature_model=None,opt=None):
    super().__init__()
    self.style_weight = style_weight
    self.content_weight = content_weight
    # self.register_buffer('style_weight', style_weight)
    # self.register_buffer('content_weight',content_weight)
    if feature_model:
      self.feature_model = feature_model
    else:
      self.feature_model = vgg.vgg16(pretrained=True)
    self.feature_model.eval()
    self.apply_feature = StyleContentLoss(model=feature_model)
    self.mse = nn.MSELoss()

  def content_loss(self,images_features,labels_features):
    content_loss = 0.
    for img_ft,label_ft in zip(images_features,labels_features):
      difference = mse(img_ft,label_ft)
      content_loss +=difference
    return self.content_weight* content_loss

  def style_loss (self,outputs_features,labels_features):
    gram_images = [gram_matrix(y).data for y in outputs_features]
    gram_labels = [gram_matrix(y).data for y in labels_features]
    style_loss =0.
    for i in range(len(labels_features)):
      gram_s = gram_labels[i]
      gram_y = gram_images[i]
      style_loss += self.mse(gram_y,gram_s.expand_as(gram_y))
      print(style_loss)
    return self.style_weight*style_loss

  def __call__(self, outputs,labels):
    outputs_features =  self.apply_feature(outputs)
    labels_features =  self.apply_feature(labels)
    content_loss = self.content_loss(outputs_features,labels_features)
    style_loss = self.style_loss(outputs_features,labels_features)
    print("style loss is",style_loss)
    return content_loss+style_loss


class StyleContentLoss(nn.Module):
    def __init__(self, model= None, opt=None):
        super(StyleContentLoss, self).__init__()
        if model:
         self. vgg_model = model
        else:
          # self.vgg_model = vgg.vgg16(pretrained=True).to(opt.device)
          self.vgg_model = vgg.vgg16(pretrained=True)
        self.vgg_layers = self.vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
    
    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)