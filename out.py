from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import glob
import os
import torch
import cv2
from PIL import Image
import create_anchor
from dataset import show,visualization
import dataset
import train



img_list = sorted(glob.glob(os.path.join("/content/coco128/coco128/images/train2017","*")))
img_size = 416

model_path = 'model.pth'
train.model.load_state_dict(torch.load(model_path))
model = train.model.cuda()
for path in img_list:
  img = cv2.imread(path)
  img = cv2.resize(img , (img_size , img_size))
  img = Image.fromarray(img)

  img = dataset.transform(img).unsqueeze(0).cuda()
  with torch.no_grad():
    preds  = list(model(img))
  img = cv2.imread(path)[:,:,::-1]
  img = cv2.resize(img , (img_size , img_size))
  img = torch.tensor(img.transpose(2,0,1))
  for color,pred in zip(["red","green","blue"],preds):
    bbox_list = visualization(pred,create_anchor.anchor,img_size,conf = 0.9)
  bbox_list_tensor = torch.tensor(bbox_list)  
  if bbox_list_tensor.shape[0] != 0:
      img = draw_bounding_boxes(img, bbox_list_tensor, colors=color, width=1)
  show(img)