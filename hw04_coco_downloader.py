import PIL.Image as Image, requests, urllib
import argparse, json, PIL.Image, reports, os
from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL
import numpy as np
from pycocotools.coco import COCO

parser = argparse.ArgumentParser(description ='HW04 Coco Downloader')
parser.add_argument('--root_path', type =str, required = True)
parser.add_argument('--coco_json_path', type =str , required = True)
parser.add_argument('--class_list', nargs = '*', type =str , required = True)
parser.add_argument('--images_per_class' ,type =int , required = True)
args, args_other = parser.parse_known_args()

root_path = args.root_path
coco_json_path = args.coco_json_path
class_list = args.class_list
images_per_class = args.images_per_class
subcategory_names = class_list

class build_category_class:
    def __init__(self, category, num_images, directory):
        self.category = category
        self.directory = directory
        self.cat_ID = coco.getCatIds(catNms=self.category)
        self.img_ID = coco.getImgIds(catIds=self.cat_ID)
        self.num_images = num_images
    def __call__(self):
        length = min(len(self.img_ID), images_per_class)
        for i in range(length):
            url, name = get_imageinfo(self.img_ID, i)
            filepath = os.path.join(self.directory, name)
            download_image(filepath, url, 0)


def get_imageinfo(img_ID, imagenumber):
    img = coco.loadImgs(img_ID[imagenumber])[0]
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
    im_resized = im.resize((64, 64), Image.BOX)
    # Overwrite original image with downsampled image
    im_resized.save(filepath)
    return "save successful"


def make_directories(path, c_list):
    try:
        os.mkdir(path)
    except OSError as error:
        print("Directory already exists:", path)
    dirs = c_list
    for i in range(len(c_list)):
        dirs[i] = path + "\\" + c_list[i]
        try:
            os.mkdir(dirs[i])
        except OSError as error:
            print("Directory already exists:", dirs[i])
    return dirs


coco = COCO(coco_json_path)
catIds = coco.getCatIds(catNms=class_list)
imgIds = coco.getImgIds(catIds=catIds)
directories = make_directories(root_path, class_list)

for i in range(len(class_list)):
    print(class_list[i])
    subclass = build_category_class(class_list[i], images_per_class, directories[i])
    subclass()
    # print(i+1, "Category downloaded: ", class_list[i])

#:
#    img = coco.loadImgs(imgIds[i])[0]
#    img_url = img['coco_url']
#    img_name = img['file_name']
#    img_resp = requests.get(img_url, timeout=1)
#

# with open(img_file_path, 'wb') as img_f:
#    img_f.write(img_resp.content)
# Resize image to 64x64

# im = Image.open(img_file_path)
# if im.mode != "RGB":
#    im = im.convert(mode="RGB")
# im_resized = im.resize((64, 64), Image.BOX)
# Overwrite original image with downsampled image
# im_resized.save(img_file_path)

# get all images containing given categories, select one at random
# catIds = coco.getCatIds(catNms=['boat'])
# imgIds = coco.getImgIds(catIds=catIds)
# print(imgIds)
# imgIds = coco.getImgIds(imgIds = [324158])
# img = coco.loadImgs(imgIds[1])[0]
# print(img['coco_url'])

# requests.get("http://images.cocodataset.org/zips/train2017.zip")


# dataDir= "C:\\Users\\aksha\\PycharmProjects\\DL695\\COCOapi\\cocoapi-master\\PythonAPI\\pycocotools\\coco.py"
# dataType='val2017'
# annFile='{}/annotations1/instances_{}.json'.format(dataDir,dataType)
# coco=COCO(annFile)
# cats = coco.loadCats(coco.getCatIds())
# nms=[cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))

# nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))
# catIds = a.getCatIds(catNms=['person','dog', 'car']) # calling the method from the class
