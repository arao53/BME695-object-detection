# -*- coding: utf-8 -*-
"""coco_scrapper.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10H4o1qFvgK9WRgVt7bDeqFE5Rh9QnbA1
"""

!pip install reports

import PIL.Image as Image, requests, urllib, random
import argparse, json, PIL.Image, os
from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL
import numpy, torch
from torch import nn
import torch.nn.functional as F
from pycocotools.coco import COCO
from torch.utils.data import DataLoader,Dataset
import glob

seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)

# Commented out IPython magic to ensure Python compatibility.
## Mount google drive to run on Colab
from google.colab import drive
drive.mount('/content/drive')
# %cd "/content/drive/My Drive/Colab Notebooks/DeepLearning/hw05/"
!pwd
!ls

#root_path = "/content/drive/My Drive/Colab Notebooks/DeepLearning/hw05/Train/"
#coco_json_path = "annotations/instances_train2017.json"
#class_list = ["bus"]#, "car"]
#images_per_class = 2000
#subcategory_names = class_list

class build_category_class:
    def __init__(self, category, num_images, directory):
        self.category = category
        self.directory = directory
        self.cat_ID = coco.getCatIds(catNms=self.category)
        self.img_ID = coco.getImgIds(catIds=self.cat_ID)
        self.num_images = min(num_images, len(self.img_ID))
    def __call__(self):     
        print(len(self.img_ID), "Images in the", self.category, "class dataset \n Downloading...")
        for idx in range(self.num_images):
            #i = numpy.random.randint(0, len(self.img_ID))
            i = self.img_ID[idx]
            #print("size of img_ID:", len(self.img_ID))
            url, name = get_imageinfo(self.img_ID[idx])
            #print(name)
            filepath = os.path.join(self.directory, name)
            download_image(filepath, url, 0)
        
        g = glob.glob(self.directory + '*/.jpg')
        print(len(g), "Images of class - ", self.category, " - downloaded \n")


def get_imageinfo(img_ID):
    img = coco.loadImgs(img_ID)[0]
    img_url = img['coco_url']
    img_name = img['file_name']
    return (img_url, img_name)


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

root_path = "/content/drive/My Drive/Colab Notebooks/DeepLearning/hw05/Val/"
coco_json_path = "annotations/instances_val2017.json"
coco = COCO(coco_json_path)
class_list = ["bicycle","car","motorcycle","person", "truck"]

catIds = coco.getCatIds(catNms=class_list)
imgIds = coco.getImgIds(catIds=catIds)
#directories = make_directories(root_path, class_list)


try:
  directory = os.mkdir(root_path)
except OSError as error:
  print("")


for i in range(len(class_list)):
  #print(class_list[i])
  subclass = build_category_class(class_list[i], 100, root_path)
  subclass()
  print(i+1, "Category downloaded: ", class_list[i])

root_path = "/content/drive/My Drive/Colab Notebooks/DeepLearning/hw05/Val/"

coco_json_path = "annotations/instances_val2017.json"
#directories = make_directories(root_path, class_list)
try:
  directory = os.mkdir(root_path)
except OSError as error:
  print("")

coco = COCO(coco_json_path)
for i in range(len(class_list)):
  #print(class_list[i])
  subclass = build_category_class(class_list[i], 100, root_path)
  subclass()
  print(i+1, "Category downloaded: ", class_list[i])