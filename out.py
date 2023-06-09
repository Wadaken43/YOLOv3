from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import glob
import numpy as np
import torchvision.transforms.functional as FF
import os
import torch
import cv2
from PIL import Image
from dataset import visualization
import dataset
import torchvision.transforms as T
from train import model
from create_anchor import anchor



img_list = sorted(glob.glob(os.path.join("/home/wada_docker/Documents/YOLOv3/coco128/images/train2017","*")))
img_size = 416

model_path = 'model_300_1.pth'
model.load_state_dict(torch.load(model_path)) ###
model = model.cuda()


def show2(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = FF.to_pil_image(img) #tensor型からPILイメージに変換
        img = np.asarray(img)
        axs[0, i].imshow(img)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        cv2.imwrite('result/out_'+str(a)+'.jpg',img)

for a,path in enumerate(img_list):
  print(a)
  img = cv2.imread(path)
  img = cv2.resize(img , (img_size , img_size))
  img = Image.fromarray(img)

  img = dataset.transform(img).unsqueeze(0).cuda()  #####
  with torch.no_grad():
    preds  = list(model(img))
  img = cv2.imread(path)[:,:,::-1]
  img = cv2.resize(img , (img_size , img_size))
  img = torch.tensor(img.transpose(2,0,1))
  for color,pred in zip(["red","green","blue"],preds):
    bbox_list = visualization(pred,anchor,img_size,conf = 0.9)
    bbox_list_tensor = torch.tensor(bbox_list)
    print(bbox_list_tensor.shape)  
    if bbox_list_tensor.shape[0] != 0:
      img = draw_bounding_boxes(img, bbox_list_tensor, colors=color, width=1)
  
  show2(img)
  if a==20:  
    break
