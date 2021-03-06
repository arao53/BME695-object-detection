# -*- coding: utf-8 -*-
"""hw06_validation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tJoJRXUG7LqVR3sQWOVMVUYYD2X01KOB
"""

!pip install reports
import PIL.Image as Image, requests, urllib, random
import argparse, json, PIL.Image, reports, os, pickle
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
# Mount google drive to run on Colab
#from google.colab import drive
#drive.mount('/content/drive')
#%cd "/content/drive/My Drive/Colab Notebooks/DeepLearning/hw06/"
#!pwd
#!ls

root_path = "/content/drive/My Drive/Colab Notebooks/DeepLearning/hw06/"
coco_json_path = "annotations/instances_val2017.json"
class_list = ["person", "dog", "hot dog"]
coco = COCO(coco_json_path)

class build_annotations:
  # Structure of the all_annotations file:
  # indexed by the image filepath, removing the '.jpg' or the string version (with zeros) of the imageID
  # For each image: 
  # 'imageID': corresponds to the integer image ID assigned within COCO.
  # 'num_objects': integer number of objects in the image (at most 5)
  # 'bbox': a dictionary of the bounding box array for each instance within the image. The dictionary key is the string 0-5 of each instance in order of decreasing area
  # 'labels': a dictionary of the labels of each instance within the image. The key is the same as bbox but the value is the integer category ID assigned within COCO. 

  def __init__(self, root_path, class_list, max_instances = 5):
    self.root_path = root_path
    self.image_dir = root_path + '*.jpg'
    self.cat_IDs = coco.getCatIds(catNms=class_list)
    self.max_instances = max_instances
  def __call__(self):
    all_annotations = {}
    g = glob.glob(self.image_dir)
    for i, filename in enumerate(g):
      filename = filename.split('/')[-1]
      img_ID = int(filename.split('.')[0])
      ann_Ids = coco.getAnnIds(imgIds=img_ID, catIds = self.cat_IDs, iscrowd = False)
      num_objects = min(len(ann_Ids), self.max_instances)    # cap at a max of 5 images
      anns = coco.loadAnns(ann_Ids)
      indices = sort_by_area(anns, self.max_instances)
      bbox = {}
      label = {}
      i = 0
      for n in indices:
        instance = anns[n]
        bbox[str(i)] = instance['bbox']
        label[str(i)] = instance['category_id']
        i+=1
      annotation= {"imageID":img_ID, "num_objects":i, 'bbox': bbox, 'labels':label}
      all_annotations[filename.split('.')[0]] = annotation
    ann_path = self.root_path +  "image_annotations.p"
    pickle.dump( all_annotations, open(ann_path, "wb" ) )
    print('Annotations saved in:', ann_path)


def sort_by_area(anns, num):
  areas = numpy.zeros(len(anns))
  for i, instance in enumerate(anns):
    areas[i] = instance['area']
  indices = numpy.argsort(areas)[-num:]
  return indices[::-1]

class your_dataset_class(Dataset):
    def __init__(self, path, class_list, coco):
        self.class_list = class_list
        self.folder = path
        self.coco = coco
        self.catIds = coco.getCatIds(catNms = class_list)
        self.imgIds = coco.getImgIds(catIds = self.catIds)
        self.categories = coco.loadCats(self.catIds)
        #create label dictionary
        labeldict = {}
        for idx, in_class in enumerate(self.class_list):
          for c in self.categories:
            if c["name"] == in_class:
              labeldict[c['id']] = idx
        self.coco_labeldict = labeldict

        #if first time running, index the image dataset to make annotation .p file
        annotation_path = path + 'image_annotations.p'
        if os.path.exists(annotation_path) ==False: 
          print("Indexing dataset to compile annotations...")
          dataset_annotations = build_annotations(path, class_list)
          dataset_annotations()
        
        self.data_anns = pickle.load(open(annotation_path, "rb" ))
        
    def __len__(self):
        g = glob.glob(self.folder + '*.jpg')  # ,'*.jpg')
        return (len(g))

    def get_imagelabel(self, img_path, sc, max_objects = 5): #img_path = file location, sc = scale [0]: width, [1]: height
      saved_filename = os.path.basename(img_path)
      filename = saved_filename.split('.jpg')[0]
      image_id = int(filename)#.split('_')[-1])

      bbox_tensor = torch.zeros(max_objects, 4, dtype=torch.uint8)
      label_tensor = torch.zeros(max_objects, dtype=torch.uint8) + 13
      
      target_obj = self.data_anns[filename]
      num_objects = target_obj['num_objects']
      for n in range(num_objects):
        [x,y,w,h] = target_obj['bbox'][str(n)]
        bbox = [sc[1]*y, x*sc[0], sc[1]*(y+h), sc[0]*(x+w)]
        bbox_tensor[n,:] = torch.tensor(numpy.array(bbox))

        cat_label = target_obj['labels'][str(n)]
        data_label = self.coco_labeldict[cat_label]
        label_tensor[n] = torch.tensor(data_label)
      return bbox_tensor, label_tensor

    def __getitem__(self, item):
      g = glob.glob(self.folder + '*.jpg') #'**/*.jpg')  # , '*.jpg')
      im = PIL.Image.open(g[item])
      im, scale_fac = rescale_factor(im, 128) #overwrite old image with new resized image of size 256
      im_ori = im
      W, H = im.size
      transformer = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
      im_array = torch.randint(0, 256, (3, H, W)).type(torch.uint8)
      for i in range(H):
         for j in range(W):
            im_array[:, j, i] = torch.tensor(im.getpixel((i, j)))

      im_scaled = im_array / im_array.max()  # scaled from 0-1
      im_tf = transformer(numpy.transpose(im_scaled.numpy()))
      num_classes = len(self.class_list)
        
      bbox_tensor = torch.zeros(5,4, dtype=torch.uint8)
      bbox_label_tensor = torch.zeros(5, dtype=torch.uint8) + len(self.class_list) +1   #predict no object

      bbox, label = self.get_imagelabel(g[item], scale_fac)

      sample = {'im_ID': g[item], 
                'scale':scale_fac,
                #'im_ori': im_ori,
                'image': im_tf,
                'bbox' : bbox,
                'label': label}
      return sample

def rescale_factor(im_original, std_size):
  raw_width, raw_height = im_original.size
  im = im_original.resize((std_size, std_size), Image.BOX)
  w_factor = std_size/raw_width
  h_factor = std_size/raw_height
  return (im, [w_factor, h_factor])


val_path = os.path.join(root_path, "Val/")

batch_size = 64

val_dataset = your_dataset_class(val_path, class_list, coco)

val_data_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                batch_size = batch_size,
                                                shuffle = True,
                                                num_workers = 2,
                                                drop_last=True)

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
    
    self.fc_seqn = nn.Sequential(
        nn.Linear(64*4*4, 3000),
        nn.ReLU(inplace =True),
        nn.Linear(3000,3000),
        nn.ReLU(inplace =True),
        nn.Linear(3000,8*8*(5*(4+3+1)))   #5 anchor boxes*( bbox + classes + no class)
    )    

  def forward(self, x):
    # x1 is the output of classification
    x = self.pool(F.relu(self.conv_initial(x)))
    
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
    x1 = self.fc_seqn(x1)

    return x1

def run_code_for_validation(net, net_path, num_classes, im_size = 128, max_objects = 5, yolo_interval = 16):
  device = torch.device("cuda:0")
  model.load_state_dict(torch.load(net_path))
  model.to(device)
  criterion = torch.nn.MSELoss() 
  loss_tracker = []
  num_cells_image_height = im_size//yolo_interval
  num_cells_image_width = im_size//yolo_interval
  num_yolo_cells = num_cells_image_height*num_cells_image_width
  print_iteration = 3
  num_anchor_boxes = 5
  predictions = []
  targets = []

  yolo_tensor = torch.zeros(batch_size, num_yolo_cells, num_anchor_boxes, 1*5+3)   #batch size, 8*8, 1*5+3 classes
  
  class Abox:
    def __init__(self, AR, topleft, abox_h, abox_w, abox_idx):
      self.AR = AR
      self.topleft = topleft
      self.abox_h = abox_h
      self.abox_w = abox_w
      self.abox_idx= abox_idx

  device = torch.device("cuda:0")
  print('Beginning validation...')

  for i, data in enumerate(val_data_loader):
    sample_batch = data['im_ID']
    im_tensor = data["image"]
    target_reg = data["bbox"].type(torch.FloatTensor)
    target_clf = data["label"].type(torch.LongTensor)
            
    im_tensor = im_tensor.to(device)
    target_reg = target_reg.to(device)
    target_clf = target_clf.to(device)
    yolo_tensor = yolo_tensor.to(device)
    obj_centers = {ibx : 
                        {idx : None for idx in range(max_objects)} 
                    for ibx in range(im_tensor.shape[0])
                    }      

    anchor_boxes_1_1 = [[Abox(1/1, (i*yolo_interval,j*yolo_interval), yolo_interval, yolo_interval,  0) 
                                                                      for i in range(num_cells_image_height)]
                                                                         for j in range(num_cells_image_width)]
    anchor_boxes_1_3 = [[Abox(1/3, (i*yolo_interval,j*yolo_interval), yolo_interval, 3*yolo_interval,  1) 
                                                                      for i in range(num_cells_image_height)]
                                                                         for j in range(num_cells_image_width)]
    anchor_boxes_3_1 = [[Abox(3/1, (i*yolo_interval,j*yolo_interval), 3*yolo_interval, yolo_interval,  2) 
                                                                      for i in range(num_cells_image_height)]
                                                                         for j in range(num_cells_image_width)]
    anchor_boxes_1_5 = [[Abox(1/5, (i*yolo_interval,j*yolo_interval), yolo_interval, 5*yolo_interval,  3) 
                                                                      for i in range(num_cells_image_height)]
                                                                         for j in range(num_cells_image_width)]
    anchor_boxes_5_1 = [[Abox(5/1, (i*yolo_interval,j*yolo_interval), 5*yolo_interval, yolo_interval,  4) 
                                                                      for i in range(num_cells_image_height)]
                                                                         for j in range(num_cells_image_width)]

    #Build the yolo tensor based on the bounding box and label tensors from the target/dataset
    for b in range(im_tensor.shape[0]):   # Loop through batch index
      for idx in range(max_objects):      # Loop through each object in the target tensor
          height_center_bb = (target_reg[b][idx][1].item() + target_reg[b][idx][3].item()) // 2   
          width_center_bb = (target_reg[b][idx][0].item() + target_reg[b][idx][2].item()) // 2
          obj_bb_height = target_reg[b][idx][3].item() - target_reg[b][idx][1].item()
          obj_bb_width = target_reg[b][idx][2].item() - target_reg[b][idx][0].item()
          obj_label = target_clf[b][idx].item()
          if obj_label == 13: 
            obj_label = 4
          
          eps = 1e-8
          AR = float(obj_bb_height + eps) / float(obj_bb_width + eps)
          cell_row_idx = int(height_center_bb // yolo_interval) ## for the i coordinate
          cell_col_idx = int(width_center_bb // yolo_interval) ## for the j coordinates

          if AR <= 0.2: ## 
            anchbox = anchor_boxes_1_5[cell_row_idx][cell_col_idx]
          elif AR <= 0.5:
            anchbox = anchor_boxes_1_3[cell_row_idx][cell_col_idx]
          elif AR <= 1.5:
            anchbox = anchor_boxes_1_1[cell_row_idx][cell_col_idx]
          elif AR <= 4:
            anchbox = anchor_boxes_3_1[cell_row_idx][cell_col_idx]
          elif AR > 4:
            anchbox = anchor_boxes_5_1[cell_row_idx][cell_col_idx]
          

          bh = float(obj_bb_height) / float(yolo_interval) ## (G)
          bw = float(obj_bb_width) / float(yolo_interval)
          obj_center_x = float(target_reg[b][idx][2].item() + target_reg[b][idx][0].item()) / 2.0
          obj_center_y = float(target_reg[b][idx][3].item() + target_reg[b][idx][1].item()) / 2.0

          yolocell_center_i = cell_row_idx*yolo_interval + float(yolo_interval) / 2.0
          yolocell_center_j = cell_col_idx*yolo_interval + float(yolo_interval) / 2.0
          del_x = float(obj_center_x - yolocell_center_j) / yolo_interval
          del_y = float(obj_center_y - yolocell_center_i) / yolo_interval
          yolo_vector = [0, del_x, del_y, bh, bw, 0, 0, 0]
          if obj_label<4:
            yolo_vector[4 + obj_label] = 1
            yolo_vector[0] = 1
          yolo_cell_index = cell_row_idx * num_cells_image_width + cell_col_idx
          yolo_tensor[b, yolo_cell_index, anchbox.abox_idx] = torch.FloatTensor( yolo_vector ) 

    yolo_tensor_flattened = yolo_tensor.view(im_tensor.shape[0], -1)

    ## Foward Pass
    pred_yolo = net(im_tensor)
    loss = criterion(pred_yolo, yolo_tensor_flattened)
    
    loss_tracker = numpy.append(loss_tracker, loss.cpu().detach().numpy())


    pred_unscrm = pred_yolo.view(im_tensor.shape[0], 8**2, 5, -1)
    
    sample_yolo_tensor = pred_unscrm

    for i in range(yolo_tensor.shape[0]):
      for j in range(yolo_tensor.shape[1]):
        for k in range(yolo_tensor.shape[2]):
          #if pred_unscrm[i,j,k,0] > 0.5:           
            #pred_label = torch.argmax(pred_unscrm[i,j,k,5:]).cpu().detach().numpy()
          #else:
          #  pred_label = num_classes
          if yolo_tensor[i,j,k,0] ==1:
            true_label = torch.argmax(yolo_tensor[i,j,k,5:]).cpu().detach().numpy()
            pred_label = torch.argmax(pred_unscrm[i,j,k,5:]).cpu().detach().numpy()
          #else:
          #  true_label = num_classes
            predictions = numpy.append(predictions, pred_label)
            targets = numpy.append(targets, true_label)

  print('Building confusion matrix...')
  confusion_matrix = numpy.zeros([num_classes,num_classes])   #3 classes + 1 no object
  for i in range(len(predictions)):
    j = int(predictions[i])
    k = int(targets[i])
    confusion_matrix[j,k] += 1
  return confusion_matrix, loss_tracker, sample_yolo_tensor, sample_batch

import seaborn as sns
class_list = ['person', 'dog', 'hot dog']
num_classes = len(class_list)
model = MechEnet(num_classes, depth = 64)
savepath = "MechEnet.pth"
confusion_matrix, val_loss, yolo_sample, batches = run_code_for_validation(model, savepath, num_classes)

print(val_loss)

fig, ax1 = plt.subplots(dpi = 120)         # Sample figsize in inches
class_list.append('No object')
sns.heatmap(confusion_matrix, annot=True, xticklabels = class_list , yticklabels = class_list, cmap='Blues', ax = ax1)
accuracy  = numpy.trace(confusion_matrix) / float(numpy.sum(confusion_matrix))
ax1.set_xlabel("Targets")
ax1.set_ylabel("Predictions")  
ax1.set_title("Confusion Matrix- Accuracy: {:.2f}%".format(100*accuracy));
plt.savefig("confusion_matrix.jpg")

def show_image(image_anns):
  img = coco.loadImgs(rand_img['imageID'])[0]
  I = io.imread(img['coco_url'])
  if len(I.shape) == 2:
    I = skimage.color.gray2rgb(I)
  catIds = coco.getCatIds(catNms= class_list)

  annIds = coco.getAnnIds(imgIds=rand_img['imageID'], catIds= catIds, iscrowd=False) 
  anns = coco.loadAnns(annIds)
  image = numpy.uint8(I)
  for i in range(rand_img['num_objects']):
    [x,y,w,h] = rand_img['bbox'][str(i)]
    print([x,y,w,h])
    label = rand_img['labels'][str(i)]
    image = cv2.rectangle(image, (int(x), int(y)), (int(x +w), int(y + h)), (36,255,12), 2)
    class_label = coco_labels_inverse[label]
    image = cv2.putText(image, class_list[class_label], (int(x), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)
  return image

coco_labels_inverse = {}
catIds = coco.getCatIds(catNms = class_list)
categories = coco.loadCats(catIds)
categories.sort(key= lambda x:x['id'])
for idx, in_class in enumerate(class_list):
  for c in categories:
    if c["name"] == in_class:
      coco_labels_inverse[c['id']] = idx

## the image i want in its yolo sample form
import math
im_considered = yolo_sample[31,:,:,:]
im_pred_anch = torch.zeros(64,8)
cell_pred = []
num_cell_width = 8
yolo_interval = 16

coco_labels_inverse = {}
catIds = coco.getCatIds(catNms = class_list)
categories = coco.loadCats(catIds)
categories.sort(key= lambda x:x['id'])
for idx, in_class in enumerate(class_list):
  for c in categories:
    if c["name"] == in_class:
      coco_labels_inverse[c['id']] = idx

def show_image(image_anns):
  img = coco.loadImgs(rand_img['imageID'])[0]
  I = io.imread(img['coco_url'])
  if len(I.shape) == 2:
    I = skimage.color.gray2rgb(I)
  catIds = coco.getCatIds(catNms= class_list)

  annIds = coco.getAnnIds(imgIds=rand_img['imageID'], catIds= catIds, iscrowd=False) 
  anns = coco.loadAnns(annIds)
  image = numpy.uint8(I)
  for i in range(rand_img['num_objects']):
    [x,y,w,h] = rand_img['bbox'][str(i)]
    label = rand_img['labels'][str(i)]
    image = cv2.rectangle(image, (int(x), int(y)), (int(x +w), int(y + h)), (36,255,12), 2)
    class_label = coco_labels_inverse[label]
    image = cv2.putText(image, 'True ' + class_list[class_label], (int(x), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)
  return image


annotation_path = root_path + 'Val/'+ 'image_annotations.p'
data_anns = pickle.load(open(annotation_path, "rb" ))

g = glob.glob(root_path + 'Val/'+ '*.jpg')
idx = numpy.random.randint(0, len(g))
#idx = 100
img_loc = batches[31].split('/')[-1].split('.')[0]
rand_img = data_anns[img_loc]
image = show_image(rand_img)


scale = val_dataset.__getitem__(idx)['scale']
#loop through the tensor, filter out non predictions, and transform true predictions
for i in range(im_considered.shape[0]):
  AR = torch.argmax(im_considered[i,:,0])
  im_pred_anch[i,:] = im_considered[i,AR,:]
  if im_pred_anch[i,0] > 1.11:
    if AR == 0:
      w,h = 1,1
    elif AR == 1:
      w,h = 1,3
    elif AR == 2:
      w,h = 3,1
    elif AR == 3:
      w,h = 1,5
    elif AR == 4:
      w,h = 5,1
    row_idx = math.floor(i/num_cell_width)
    col_idx = i%num_cell_width
    yolo_box = im_pred_anch[i,1:5].cpu().detach().numpy()
    x1 = ((row_idx + 0.5)*yolo_interval)/scale[0]
    x2 = x1 + (w*yolo_interval)/scale[0]
    y1 = (col_idx + 0.5)*yolo_interval/scale[1]
    y2 = y1+ (h*yolo_interval)/scale[1]
    label = torch.argmax(im_pred_anch[i,5:]).cpu().detach().numpy()
    pred_label = str('Predicted ' + class_list[label])
    temp = [pred_label, x1,y1, x2,y2]
    cell_pred = numpy.append(cell_pred, temp)
    image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
    image = cv2.putText(image, pred_label, (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

fig, ax = plt.subplots(1,1, dpi = 150)
ax.imshow(image)
ax.set_axis_off()
plt.axis('tight')
plt.show()

data_loaded_image = val_dataset.__getitem__(idx)