# -*- coding: utf-8 -*-
"""hw05_validation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xUMu2eElo3eaYrIfEmHlbhAdONUSO-l4
"""

!pip install reports
import PIL.Image as Image, requests, urllib, random
import argparse, json, PIL.Image, reports, os
from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL
import numpy, torch, cv2, skimage
import skimage.io as io
from torch import nn
import torch.nn.functional as F
from pycocotools.coco import COCO
import glob
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as tvt
import matplotlib.pyplot as plt
from torchsummary import summary
import pandas as pd
## Mount google drive to run on Colab
#from google.colab import drive
#drive.mount('/content/drive')
#%cd "/content/drive/My Drive/Colab Notebooks/DeepLearning/hw05/"
#!pwd
#!ls

class your_dataset_class(Dataset):        
    def __init__(self, path, class_list, coco):
        self.class_list = class_list
        self.folder = path
        self.coco = coco
        self.catIds = coco.getCatIds(catNms = class_list)
        self.imgIds = coco.getImgIds(catIds = self.catIds)
        self.categories = coco.loadCats(self.catIds)
        labeldict = {}
        for idx, in_class in enumerate(self.class_list):
          for c in self.categories:
            if c["name"] == in_class:
              labeldict[c['id']] = idx
        self.coco_labeldict = labeldict
        #print(self.categories)

    def __len__(self):
        g = glob.glob(self.folder + '*.jpg')  # ,'*.jpg')
        return (len(g))
    
    def get_dominantfeature(self, anns): # returns the dominant label and bounding box label, x-coord, y-coord,width,height
      area_prev = 0
      label = 0 
      [x,y,w,h] = [0,0,0,0]
      for ann in anns:
        [x_temp, y_temp, w_temp, h_temp] = ann['bbox']
        area_temp = w_temp*h_temp
        #print(ann['category_id'], area_temp)
        if area_temp>area_prev:
          [x,y,w,h] = [x_temp, y_temp, w_temp, h_temp]
          area_prev = area_temp
          label = self.coco_labeldict[ann['category_id']]    
      return label, [x,y,w,h]

    def get_imagelabel(self, img_path, sc): #img_path = file location, sc = scale [0]: width, [1]: height
      saved_filename = os.path.basename(img_path)
      filename = saved_filename.split('.jpg')[0]
      image_id = int(filename)#.split('_')[-1])
      annIds = self.coco.getAnnIds(imgIds=image_id, catIds= self.catIds, iscrowd=False)
      anns = self.coco.loadAnns(annIds)
      #print(anns)
      main_feature, [x,y,w,h] = self.get_dominantfeature(anns)
      bbox = [sc[1]*y, x*sc[0], sc[1]*(y+h), sc[0]*(x+w)]
      bbox = [int(x) for x in bbox]   #inverted from x-y to rows-cols and scaled by sc
      return [main_feature, bbox]

    def __getitem__(self, item):
        g = glob.glob(self.folder + '*.jpg') #'**/*.jpg')  # , '*.jpg')
        im = PIL.Image.open(g[item])
        im, scale_fac = rescale_factor(im, 128) #overwrite old image with new resized image of size 256
        
        W, H = im.size
        transformer = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        im_array = torch.randint(0, 256, (3, H, W)).type(torch.uint8)
        for i in range(H):
            for j in range(W):
                im_array[:, j, i] = torch.tensor(im.getpixel((i, j)))

        im_scaled = im_array / im_array.max()  # scaled from 0-1
        im_tf = transformer(numpy.transpose(im_scaled.numpy()))
        num_classes = len(self.class_list)
        
        targets = self.get_imagelabel(g[item], scale_fac)
        class_label = torch.tensor(targets[0])
        bbox_label = torch.tensor(targets[1])
        sample = {'image': im_tf,
                  'bbox' : bbox_label,
                  'label': class_label}
        return sample

def rescale_factor(im_original, std_size):
  raw_width, raw_height = im_original.size
  im = im_original.resize((std_size, std_size), Image.BOX)
  w_factor = std_size/raw_width
  h_factor = std_size/raw_height
  return (im, [w_factor, h_factor])

root_path = "/content/drive/My Drive/Colab Notebooks/DeepLearning/hw05/"
coco_json_path = "annotations/instances_val2017.json"
class_list = ["bicycle","car","motorcycle","person", "truck"]
coco_object = COCO(coco_json_path)

val_path = os.path.join(root_path, "Val/")

batch_size = 1

#train_dataset = your_dataset_class(train_path, class_list, coco_object)

#train_data_loader = torch.utils.data.DataLoader(dataset = train_dataset,
#                                                batch_size = batch_size,
#                                                shuffle = True,
#                                                num_workers= 2,
#                                                drop_last=True)

val_dataset = your_dataset_class(val_path, class_list, coco_object)

val_data_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                batch_size = batch_size,
                                                shuffle = True,
                                                num_workers = 4,
                                                drop_last=True)

## Model with resnet
class SkipBlock(nn.Module):
  def __init__(self,in_ch, out_ch, downsample = False):
    super().__init__()
    self.in_ch = in_ch
    self.out_ch = out_ch
    self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride = 1, padding = 1)
    self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding = 1)
    self.bnorm1 = nn.BatchNorm2d(out_ch)
    self.bnorm2 = nn.BatchNorm2d(out_ch)
    self.downsample_tf = downsample
    self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride= 2)
  def forward(self, x):
    identity = x
    out = self.conv1(x)
    out = self.bnorm1(out)
    out = F.relu(out)
    if self.downsample_tf == True:
      identity = self.downsampler(identity)
      out = self.downsampler(out)
      out += identity
    else:
      out = self.conv2(out)
      out = self.bnorm2(out)
      out = F.relu(out)
      out += identity  
    return out

class MechEnet(nn.Module):
  def __init__(self, num_classes, depth):
    super().__init__()
    self.depth = depth // 8
    self.conv_initial = nn.Conv2d( 3, 64, 3, padding = 1)
    self.pool = nn.MaxPool2d(2,2)
    ## assume all layers are 64 channels deep
    self.skipblock64_1 = nn.ModuleList()
    for i in range(self.depth):
      #print("adding layer", i)
      self.skipblock64_1.append( SkipBlock(64,64, downsample = False) ) #append a 64 in/out ch layer - depth*2/4 convolutions
    self.skip_downsample = SkipBlock(64,64, downsample= True)
    self.skipblock64_2 = nn.ModuleList()
    for i in range(self.depth):
      #print("adding layer", i + self.depth)
      self.skipblock64_2.append( SkipBlock(64,64, downsample = False) ) #append a 64 in/out layer - depth*2/4 convolutions
    self.fc1 = nn.Linear(4*4*64, 256) # fc layer 1
    self.fc2 = nn.Linear(256, num_classes) # 5 classes - person, bike, motorcycle, car, truck
    # for regression
    self.conv_seqn = nn.Sequential(
        #nn.MaxPool2d(2,2),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace = True),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 3, 3, padding=1),
        nn.ReLU(inplace=True)
    )
    self.fc_seqn = nn.Sequential(
        nn.Linear(64*64*3, 512),
        nn.ReLU(inplace =True),
        nn.Linear(512,512),
        nn.ReLU(inplace =True),
        nn.Linear(512,512),
        nn.ReLU(inplace =True),
        nn.Linear(512,512),
        nn.ReLU(inplace =True),
        nn.Linear(512,4),
    )
  def forward(self, x):
    # x1 is the output of classification
    x = self.pool(F.relu(self.conv_initial(x)))
    x1 = x.clone()
    x1 = self.skip_downsample(x)
    for i, skips in enumerate(self.skipblock64_1[self.depth//4 :]):
      x1 = skips(x1)
    x1  = self.skip_downsample(x1)
    for i, skips in enumerate(self.skipblock64_1[:self.depth//4]):
      x1 = skips(x1)
    x1  = self.skip_downsample(x1)
    for i, skips in enumerate(self.skipblock64_2[self.depth//4:]):
      x1 = skips(x1)
    x1  = self.skip_downsample(x1)
    for i, skips in enumerate(self.skipblock64_2[:self.depth//4]):
      x1 = skips(x1)
    #x1 = self.skip_downsample(x)
    x1 = x1.view(x1.size(0),-1)
    x1 = F.relu(self.fc1(x1))
    x1 = self.fc2(x1)
    #x2 is for reqression
    x2 = self.conv_seqn(x)
    x2 = x2.view(x2.size(0), -1)
    x2 = self.fc_seqn(x2)
    return x1, x2

def run_code_for_validation(model, net_path, num_cat):
  device = torch.device("cuda:0")
  model.load_state_dict(torch.load(net_path))
  model.to(device)
  predictions = []
  targets = []
  for i, data in enumerate(val_data_loader):
    device = torch.device('cuda:0')
    inputs = data["image"]
    target_reg = data["bbox"].type(torch.FloatTensor)
    target_clf = data["label"].type(torch.LongTensor)
    inputs = inputs.to(device)
    target_reg = target_reg.to(device)
    target_clf = target_clf.to(device)
      
    outputs_clf, outputs_reg = model(inputs)
    outputs_clf = torch.argmax(outputs_clf, axis = 1)
    pred_intermediate = outputs_clf.cpu().detach().numpy()
    target_intermediate = target_clf.cpu().detach().numpy()
    predictions = numpy.append(predictions, pred_intermediate)
    targets = numpy.append(targets, target_intermediate)
  predictions = numpy.reshape(predictions, [-1])
  targets = numpy.reshape(targets, [-1])
  print(len(predictions),len(targets))
  confusion_matrix = numpy.zeros([num_cat, num_cat])   #create an nxn matrix of categories
  for i in range(len(predictions)):
    j = int(predictions[i])
    k = int(targets[i])
    confusion_matrix[j,k] += 1
  return confusion_matrix

import seaborn as sns
num_classes = len(class_list)
model_resnet1 = MechEnet(num_classes, depth = 64)
savepath = "MechEnet_wres.pth"
confusion_matrix3 = run_code_for_validation(model_resnet1, savepath, num_classes)

fig, ax1 = plt.subplots(dpi = 120)         # Sample figsize in inches
sns.heatmap(confusion_matrix3, annot=True, xticklabels = class_list , yticklabels = class_list, cmap='Blues', ax = ax1)
accuracy  = numpy.trace(confusion_matrix3) / float(numpy.sum(confusion_matrix3))
ax1.set_xlabel("Targets")
ax1.set_ylabel("Predictions")  
ax1.set_title("Confusion Matrix- Accuracy: {:.2f}%".format(100*accuracy));
plt.savefig("confusion_matrix_resnet.jpg")

## Model with no resnet
class SkipBlock(nn.Module):
  def __init__(self,in_ch, out_ch, downsample = False):
    super().__init__()
    self.in_ch = in_ch
    self.out_ch = out_ch
    self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride = 1, padding = 1)
    self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding = 1)
    self.bnorm1 = nn.BatchNorm2d(out_ch)
    self.bnorm2 = nn.BatchNorm2d(out_ch)
    self.downsample_tf = downsample
    self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride= 2)
  def forward(self, x):
    identity = x
    out = self.conv1(x)
    out = self.bnorm1(out)
    out = F.relu(out)
    if self.downsample_tf == True:
      identity = self.downsampler(identity)
      out = self.downsampler(out)
      #out += identity
    else:
      out = self.conv2(out)
      out = self.bnorm2(out)
      out = F.relu(out)
      out += identity  
    return out

class MechEnet(nn.Module):
  def __init__(self, num_classes, depth):
    super().__init__()
    self.depth = depth // 8
    self.conv_initial = nn.Conv2d( 3, 64, 3, padding = 1)
    self.pool = nn.MaxPool2d(2,2)
    ## assume all layers are 64 channels deep
    self.skipblock64_1 = nn.ModuleList()
    for i in range(self.depth):
      #print("adding layer", i)
      self.skipblock64_1.append( SkipBlock(64,64, downsample = False) ) #append a 64 in/out ch layer - depth*2/4 convolutions
    self.skip_downsample = SkipBlock(64,64, downsample= True)
    self.skipblock64_2 = nn.ModuleList()
    for i in range(self.depth):
      #print("adding layer", i + self.depth)
      self.skipblock64_2.append( SkipBlock(64,64, downsample = False) ) #append a 64 in/out layer - depth*2/4 convolutions
    self.fc1 = nn.Linear(4*4*64, 256) # fc layer 1
    self.fc2 = nn.Linear(256, num_classes) # 5 classes - person, bike, motorcycle, car, truck
    # for regression
    self.conv_seqn = nn.Sequential(
        #nn.MaxPool2d(2,2),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace = True),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 3, 3, padding=1),
        nn.ReLU(inplace=True)
    )
    self.fc_seqn = nn.Sequential(
        nn.Linear(64*64*3, 512),
        nn.ReLU(inplace =True),
        nn.Linear(512,512),
        nn.ReLU(inplace =True),
        nn.Linear(512,512),
        nn.ReLU(inplace =True),
        nn.Linear(512,512),
        nn.ReLU(inplace =True),
        nn.Linear(512,4),
    )
  def forward(self, x):
    # x1 is the output of classification
    x = self.pool(F.relu(self.conv_initial(x)))
    x1 = x.clone()
    x1 = self.skip_downsample(x)
    for i, skips in enumerate(self.skipblock64_1[self.depth//4 :]):
      x1 = skips(x1)
    x1  = self.skip_downsample(x1)
    for i, skips in enumerate(self.skipblock64_1[:self.depth//4]):
      x1 = skips(x1)
    x1  = self.skip_downsample(x1)
    for i, skips in enumerate(self.skipblock64_2[self.depth//4:]):
      x1 = skips(x1)
    x1  = self.skip_downsample(x1)
    for i, skips in enumerate(self.skipblock64_2[:self.depth//4]):
      x1 = skips(x1)
    #x1 = self.skip_downsample(x)
    x1 = x1.view(x1.size(0),-1)
    x1 = F.relu(self.fc1(x1))
    x1 = self.fc2(x1)
    #x2 is for reqression
    x2 = self.conv_seqn(x)
    x2 = x2.view(x2.size(0), -1)
    x2 = self.fc_seqn(x2)
    return x1, x2

num_classes = len(class_list)
model_resnet1 = MechEnet(num_classes, depth = 64)
savepath = "MechEnet_nores.pth"
confusion_matrix3 = run_code_for_validation(model_resnet1, savepath, num_classes)

fig, ax1 = plt.subplots(dpi = 120)         # Sample figsize in inches
sns.heatmap(confusion_matrix3, annot=True, xticklabels = class_list , yticklabels = class_list, cmap='Blues', ax = ax1)
accuracy  = numpy.trace(confusion_matrix3) / float(numpy.sum(confusion_matrix3))
ax1.set_xlabel("Targets")
ax1.set_ylabel("Predictions")  
ax1.set_title("Confusion Matrix- Accuracy: {:.2f}%".format(100*accuracy));
plt.savefig("confusion_matrix_noresnet.jpg")

no_res =  pd.read_csv("reg_loss_nores.csv")
res = pd.read_csv("reg_loss_res.csv")

fig, ax = plt.subplots()
ax.plot(no_res['0'], label = "Regular network")
ax.plot(res['0'], label = "With Skip Connections")
ax.legend();
ax.set_xlabel("Training iterations");
ax.set_ylabel("MSE Loss");
ax.set_title("Regression Loss comparison");
plt.savefig("reg_loss.jpg")

no_res =  pd.read_csv("clf_loss_nores.csv")
res = pd.read_csv("clf_loss_res.csv")

fig, ax = plt.subplots()
ax.plot(no_res['0'], label = "Regular network")
ax.plot(res['0'], label = "With Skip Connections")
ax.legend();
ax.set_xlabel("Training iterations");
ax.set_ylabel("CrossEntropy Loss");
ax.set_title("Classification Loss comparison");
plt.savefig("clf_loss.jpg")