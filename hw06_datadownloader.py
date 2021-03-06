# -*- coding: utf-8 -*-
"""hw06_downloader.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MukLV3uy9QpF953MeR4YYAK0DrGosL_q
"""

!pip install reports

import PIL.Image as Image, requests, urllib, random
import argparse, json, PIL.Image, os, pickle
from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL
import numpy, torch
from torch import nn
import torch.nn.functional as F
from pycocotools.coco import COCO
from torch.utils.data import DataLoader,Dataset
import glob 
import seaborn as sns
seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)

## Mount google drive to run on Colab
#from google.colab import drive
#drive.mount('/content/drive')
#%cd "/content/drive/My Drive/Colab Notebooks/DeepLearning/hw06/"
#!pwd
#!ls

class build_dataset:
    def __init__(self, class_list, num_images, directory):
        self.class_list = class_list
        self.cat_list = coco.getCatIds(class_list)
        #self.category = category
        self.directory = directory
        #self.cat_ID = coco.getCatIds(catNms=self.category)
        #self.img_ID = coco.getImgIds(catIds=self.cat_ID)
        self.num_images = num_images
    def __call__(self):     
        counter = 0
        for category in self.class_list:
          cat_ID = coco.getCatIds(catNms = category)
          img_ID = coco.getImgIds(catIds = cat_ID)
          self.num_images = min(self.num_images, len(img_ID))
          for idx in range(self.num_images):
              #i = numpy.random.randint(0, len(self.img_ID))
              i = img_ID[idx]
              #print("size of img_ID:", len(self.img_ID))
              url, name = get_imageinfo(img_ID[idx])
              #print(name)
              filepath = os.path.join(self.directory, name)
              check = check_image(img_ID[idx], self.cat_list, filepath)
              if check == True:
                download_image(filepath, url, 0)
                counter +=1 
        print(counter, "Images downloaded in dataset")

        image_dir = self.directory + '*.jpg'
        g = glob.glob(image_dir)
        print(len(g)- len(set(g)), "Duplicate images\n")


def get_imageinfo(img_ID):
    img = coco.loadImgs(img_ID)[0]
    img_url = img['coco_url']
    img_name = img['file_name']
    return (img_url, img_name)

def check_image(img_ID, cat_list, filepath):
    # function checks whether an image should be downloaded 
    # Returns False if the image should be passed, True if the image should be downloaded
    if os.path.exists(filepath)==True:
      return False

    img = coco.loadImgs(img_ID)[0]
    ann_IDs = coco.getAnnIds(imgIds= img_ID, catIds= cat_list, iscrowd = False)
    anns = coco.loadAnns(ann_IDs)
    
    if len(anns) < 2:
      return False
    else:
      return True


def download_image(filepath, img_url, iter):
    if len(img_url) <= 1:
        return
    try:
        img_resp = requests.get(img_url, timeout=1)
    except ConnectionError:
        iter += 1
        if iter < 5:
            download_image(filepath, img_url, iter)
            return
        else:
            return
    except ReadTimeout:
        return "next image"
    except TooManyRedirects:
        return "next image"
    except MissingSchema:
        return "next image"
    except InvalidURL:
        return "next image"
    if not 'content-type' in img_resp.headers:
        return "next image"
    if not 'image' in img_resp.headers['content-type']:
        return "next image"
    if (len(img_resp.content) < 1000):
        return "next image"
    #os.mkdir(filepath)
    with open(filepath, 'wb') as img_f:
        img_f.write(img_resp.content)
    im = Image.open(filepath)
    if im.mode != "RGB":
        im = im.convert(mode="RGB")
    #im_resized = im.resize((64, 64), Image.BOX)
    # Overwrite original image with downsampled image
    #im_resized.save(filepath)
    im.save(filepath)
    return "save successful"


def make_directories(path, c_list):
    try:
        os.mkdir(path)
    except OSError as error:
        print("Directory already exists:", path)
    dirs = c_list
    for i in range(len(c_list)):
        dirs[i] = os.path.join(path,c_list[i])
        try:
            os.mkdir(dirs[i])
        except OSError as error:
            print("Directory already exists:", dirs[i])
    return dirs

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
      #print('\nbbox:', bbox, 'label:', label, '\n')
      #print(img_ID, i)
      #create_annotation(filename, i, bbox, label)
      #build annotation filename
      #filename_ann = "image_ann/" + filename.split('.')[0] + '.p'

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

root_path = "/content/drive/My Drive/Colab Notebooks/DeepLearning/hw06/Val/"
coco_json_path = "annotations/instances_val2017.json"
coco = COCO(coco_json_path)
class_list = ["person", "dog", "hot dog"]

catIds = coco.getCatIds(catNms=class_list)
imgIds = coco.getImgIds(catIds=catIds)
#directories = make_directories(root_path, class_list)


try:
  directory = os.mkdir(root_path)
except OSError as error:
  print("")

#training_dataset = build_dataset(class_list, 1000, root_path)
#training_dataset()

validation_dataset = build_dataset(class_list, 100, root_path)
validation_dataset()

dataset_annotations = build_annotations(root_path, class_list)
dataset_annotations()

areas = numpy.zeros(len(anns))
for i, instance in enumerate(anns):
  areas[i] = instance['area']
  #print(instance['bbox'])
indices = numpy.argsort(areas)[-5:]
idx_r = indices[::-1] #reversing using list slicing
#print(areas, idx_r)
mydict = {}
info = {'a': 1, 'b':2}
name = 'fun'
mydict['info'] = info
info = {'a': 12, 'b':2}
name = 'notfun'
mydict['info'] = (info)
mydict['info']['a']

# Create all annotation file 
image_dir = root_path + '*.jpg'
cat_IDs = coco.getCatIds(catNms=class_list)
max_instances = 5
all_annotations = {}
g = glob.glob(image_dir)
for i, filename in enumerate(g):
  filename = filename.split('/')[-1]
  img_ID = int(filename.split('.')[0])
  ann_Ids = coco.getAnnIds(imgIds=img_ID, catIds = cat_IDs, iscrowd = False)
  num_objects = min(len(ann_Ids), max_instances)    # cap at a max of 5 images
  anns = coco.loadAnns(ann_Ids)
  indices = sort_by_area(anns, max_instances)
  bbox = {}
  label = {}
  i = 0
  for n in indices:
    instance = anns[n]
    bbox[str(i)] = instance['bbox']
    label[str(i)] = instance['category_id']
    i+=1
  #print('\nbbox:', bbox, 'label:', label, '\n')
  #print(img_ID, i)
  #create_annotation(filename, i, bbox, label)
  #build annotation filename
  #filename_ann = "image_ann/" + filename.split('.')[0] + '.p'

  annotation= {"imageID":img_ID, "num_objects":i, 'bbox': bbox, 'labels':label}
  all_annotations[filename.split('.')[0]] = annotation

ann_path = root_path +  "image_annotations.p"
pickle.dump( all_annotations, open(ann_path, "wb" ) )

# Structure of the all_annotations file:
# indexed by the image filepath, removing the '.jpg' or the string version (with zeros) of the imageID
# For each image: 
# 'imageID': corresponds to the integer image ID assigned within COCO.
# 'num_objects': integer number of objects in the image (at most 5)
# 'bbox': a dictionary of the bounding box array for each instance within the image. The dictionary key is the string 0-5 of each instance in order of decreasing area
# 'labels': a dictionary of the labels of each instance within the image. The key is the same as bbox but the value is the integer category ID assigned within COCO.