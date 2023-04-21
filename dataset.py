from six import b
from torch.utils.data import Dataset,DataLoader
import cv2
from PIL import Image
import math
import numpy as np
from torchvision.ops.boxes import box_iou
import glob
import os
import torch
import torch.nn as nn
import create_anchor 
import pandas as pd



class YOLOv3_Dataset(Dataset):
  
  def __init__(self,img_list , label_list,class_n,img_size,anchor_dict,transform):
    super().__init__()
    self.img_list = img_list
    self.label_list = label_list
    self.anchor_dict = anchor_dict
    self.class_n = class_n
    self.img_size = img_size
    self.transform = transform
    self.anchor_iou = torch.cat([torch.zeros(9,2) , torch.tensor(self.anchor_dict[["width","height"]].values)] ,dim = 1)

#ラベルを読み出す
  def get_label(self , path): 
    bbox_list = []
    with open(path , 'r',newline='\n') as f:  
      for s_line in f:
        bbox = [float(x) for x in s_line.rstrip('\n').split(' ')]
        bbox_list.append(bbox)
    return bbox_list

  #幅と高さをTw,Thに変換する関数  
  def wh2twth(self, wh):
    twth = []
    for i in range(9):
      anchor = self.anchor_dict.iloc[i]
      aw = anchor["width"]
      ah = anchor["height"]
      twth.append([math.log(wh[0]/aw) , math.log(wh[1]/ah)])
    return twth

  # 中心座標をCx,Cy,tx,tyに変換する関数
  def cxcy2txty(self,cxcy):
    map_size = [int(self.img_size/32) , int(self.img_size/16) , int(self.img_size/8)]
    txty = []
    for size in map_size:
      grid_x = int(cxcy[0]*size)
      grid_y = int(cxcy[1]*size)
      
      tx = math.log((cxcy[0]*size - grid_x + 1e-10) / (1 - cxcy[0]*size +grid_x+ 1e-10))
      ty = math.log((cxcy[1]*size - grid_y+ 1e-10) / (1 - cxcy[1]*size + grid_y+ 1e-10))
      txty.append([grid_x , tx , grid_y ,ty])
    return txty

  #ラベルをテンソルに変換する関数
  def label2tensor(self , bbox_list):
    map_size = [int(self.img_size/32) , int(self.img_size/16) , int(self.img_size/8)]
    tensor_list = []

    for size in map_size:
      for x in range(3):
        tensor_list.append(torch.zeros((4 + 1 + self.class_n,size,size)))
    
    for bbox in bbox_list:
      cls_n = int(bbox[0])
      txty_list = self.cxcy2txty(bbox[1:3])
      twth_list = self.wh2twth(bbox[3:])
      
      #バウンディングボックスと最も形状が近いAnchorBoxはIoUを用いて計算する。
      label_iou = torch.cat([torch.zeros((1,2))  , torch.tensor(bbox[3:]).unsqueeze(0)],dim=1)
      iou = box_iou(label_iou, self.anchor_iou)[0]
      obj_idx = torch.argmax(iou).item()

      for i , twth in enumerate(twth_list):
        tensor = tensor_list[i]
        txty = txty_list[int(i/3)]
       
        if i == obj_idx:
          
          tensor[0,txty[2],txty[0]] = txty[1]
          tensor[1,txty[2],txty[0]] = txty[3]
          tensor[2,txty[2],txty[0]] = twth[0]
          tensor[3,txty[2],txty[0]] = twth[1]
          tensor[4,txty[2],txty[0]] = 1
          tensor[5 + cls_n,txty[2],txty[0]] = 1
    

    scale3_label = torch.cat(tensor_list[0:3] , dim = 0)
    scale2_label = torch.cat(tensor_list[3:6] , dim = 0)
    scale1_label = torch.cat(tensor_list[6:] , dim = 0)

    return scale3_label , scale2_label , scale1_label  
     
  #__getitem__メソッド：指定されたインデックスに対応するデータとラベルを返すためのメソッド
  def __getitem__(self , idx):
    img_path = self.img_list[idx]
    
    label_path = self.label_list[idx]
    
    bbox_list = self.get_label(label_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img , (self.img_size , self.img_size))
    img = Image.fromarray(img)
    img = self.transform(img)
    scale3_label , scale2_label , scale1_label = self.label2tensor(bbox_list)
    
    return img , scale3_label , scale2_label , scale1_label
    
  def __len__(self):
    return len(self.img_list)


###データセットの確認
import numpy as np

def visualization(y_pred,create_anchor.anchor,img_size,conf = 0.5,is_label = False):
  size = y_pred.shape[2]
  anchor_size = create_anchor.anchor[create_anchor.anchor["type"] == size]
  bbox_list = []
  for i in range(3):
    a = anchor_size.iloc[i]
    grid = img_size/size
    y_pred_cut = y_pred[0,i*(4 + 1 + class_n) :(i+1)*(4 + 1 + class_n) ].cpu()
    if is_label:
      y_pred_conf = y_pred_cut[4,:,:].cpu().numpy()
    else:
      y_pred_conf = torch.sigmoid(y_pred_cut[4,:,:]).cpu().numpy()         
    index = np.where(y_pred_conf > conf)
    
    for y,x in zip(index[0],index[1]):
      cx = x*grid + torch.sigmoid(y_pred_cut[0,y,x]).numpy()*grid
      cy = y*grid + torch.sigmoid(y_pred_cut[1,y,x]).numpy()*grid
      width = a["width"]*torch.exp(y_pred_cut[2,y,x]).numpy()*img_size
      height = a["height"]*torch.exp(y_pred_cut[3,y,x]).numpy()*img_size
      xmin,ymin,xmax,ymax = cx - width/2 , cy - height/2 ,cx + width/2 , cy + height/2
      bbox_list.append([xmin,ymin,xmax,ymax])
  return bbox_list
 
import torchvision.transforms.functional as FF
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = FF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
img_list = glob.glob(os.path.join("/home/wada/YOLOv3/coco128/images/train2017","*"))
img_size = 416


#print(anchor)
import torchvision.transforms as T
class_n = 80
img_list = sorted(glob.glob(os.path.join("/home/wada/YOLOv3/coco128/images/train2017","*")))
label_list = sorted(glob.glob(os.path.join("/home/wada/YOLOv3/coco128/labels/train2017","*")))
transform = T.Compose([T.ToTensor()])
train_data = YOLOv3_Dataset(img_list,label_list,80,img_size , create_anchor.anchor,transform)
train_loader = DataLoader(train_data , batch_size = 1)

import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 200

for n , (img , scale3_label , scale2_label ,scale1_label) in enumerate(train_loader):
  path = img_list[n]
  img = cv2.imread(path)[:,:,::-1]
  img = cv2.resize(img , (img_size , img_size))
  img = torch.tensor(img.transpose(2,0,1))
  preds = [scale3_label , scale2_label , scale1_label]
  for color,pred in zip(["red","green","blue"],preds):
    bbox_list = visualization(pred,create_anchor.anchor,img_size,conf = 0.9,is_label = True)
    img = draw_bounding_boxes(img, torch.tensor(bbox_list), colors=color, width=1)
  show(img)