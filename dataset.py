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
from create_anchor import anchor_dict,anchor 


class YOLOv3_Dataset(Dataset):
  
  def __init__(self,img_list , label_list,class_n,img_size,anchor_dict,transform):
    super().__init__()
    self.img_list = img_list
    self.label_list = label_list
    self.anchor_dict = anchor_dict #create_anchorで出力したanchorのデータ
    self.class_n = class_n
    self.img_size = img_size
    self.transform = transform
    #anchor_boxを2次元tensorに落とし込んだもの//values 辞書から値のみ出す。
    self.anchor_iou = torch.cat([torch.zeros(9,2) , torch.tensor(self.anchor_dict[["width","height"]].values)] ,dim = 1)


#coco128からラベルをリスト型で読み出す
  def get_label(self , path): 
    bbox_list = []
    with open(path , 'r',newline='\n') as f:  
      for s_line in f:
        bbox = [float(x) for x in s_line.rstrip('\n').split(' ')]
        bbox_list.append(bbox)
    return bbox_list

###YOLOv3の予測方法
  #(YOLOv3の予測tw,th 微調整用)
  #Anchorboxをtw,thを用いて微調整することで物体を予測する。
  #直接物体の幅や高さを予測するのではなく、AnchorBoxの微調整を学習する。
  def wh2twth(self, wh): #whはデータセットの物体の幅と高さ
    twth = []
    for i in range(9):
      anchor = self.anchor_dict.iloc[i]
      aw = anchor["width"] #aw,ahはアンカーボックスのサイズ
      ah = anchor["height"]
      twth.append([math.log(wh[0]/aw) , math.log(wh[1]/ah)])
      #print(type(twth))
      #print(twth)
    return twth

  # 中心座標をCx,Cy,tx,tyに変換する関数(YOLOv3の予測tx,ty)
  def cxcy2txty(self,cxcy):
    #全ての出力サイズに対して計算している。特定のサイズのみで十分
    map_size = [int(self.img_size/32) , int(self.img_size/16) , int(self.img_size/8)] 
    txty = []
    for size in map_size:
      #grid_x,grid_y:bboxの中心座標(0~1)*それぞれの出力サイズ=それぞれの出力サイズにおける1ピクセルの幅Cx,Cy
      grid_x = int(cxcy[0]*size) ###int
      grid_y = int(cxcy[1]*size)
      
      tx = math.log((cxcy[0]*size - grid_x + 1e-10) / (1 - cxcy[0]*size +grid_x+ 1e-10))
      ty = math.log((cxcy[1]*size - grid_y+ 1e-10) / (1 - cxcy[1]*size + grid_y+ 1e-10))
      txty.append([grid_x , tx , grid_y ,ty])

    return txty

  #ラベルをテンソルに変換する関数
  def label2tensor(self , bbox_list):  
    map_size = [int(self.img_size/32) , int(self.img_size/16) , int(self.img_size/8)]
    tensor_list = []
  #学習時において、学習データ(画像)ごとに異なるスケールに対応する3つの3Dtensorを予測する。
  #各grid cellに3サイズのbbox、各boxに4つの座標と、検出対象かを表すobjectness scoreと、ラベルの確信が格納される。
    for size in map_size:
      for x in range(3):
        tensor_list.append(torch.zeros((4 + 1 + self.class_n,size,size))) 
        #print(tensor_list)
        #print(tensor_list[0].shape)
    
    ##yolov4の予測結果をリストに格納
    for bbox in bbox_list: #bbox_listはcoco128データセットの中のlabelsのリスト
      cls_n = int(bbox[0])
      txty_list = self.cxcy2txty(bbox[1:3]) #bboxの中心座標 一回転で3つのリスト
      twth_list = self.wh2twth(bbox[3:]) #bboxの幅と高さ 一回転で9つのリスト
      #if (bbox ==[0.0, 0.0884609, 0.379012, 0.0463594, 0.112612] ):
        #print(txty_list)
      #print(bbox)
      
      #バウンディングボックスと最も形状が近いAnchorBoxはIoUを用いて計算する。
      label_iou = torch.cat([torch.zeros((1,2))  , torch.tensor(bbox[3:]).unsqueeze(0)],dim=1) 
      iou = box_iou(label_iou, self.anchor_iou)[0]
      obj_idx = torch.argmax(iou).item() #最もiouが大きかったindex番号 1回転で1つだけ出力

      for i , twth in enumerate(twth_list): #iに入るのは0から8まで twthに入るのはyolov4の出力(リスト)
        tensor = tensor_list[i] 
        #print(tensor_list[3].size())
        txty = txty_list[int(i/3)] ##why box3つずつの中心点は変わらないから
  
        if i == obj_idx:  #iouが最も大きかった場合,yolov4の予測をtensorに追加する。
          tensor[0,txty[2],txty[0]] = txty[1] #tx
          tensor[1,txty[2],txty[0]] = txty[3] #ty
          tensor[2,txty[2],txty[0]] = twth[0] #th
          tensor[3,txty[2],txty[0]] = twth[1] #tw
          tensor[4,txty[2],txty[0]] = 1 #objectness score
          tensor[5 + cls_n,txty[2],txty[0]] = 1 #クラス  

    scale3_label = torch.cat(tensor_list[0:3] , dim = 0)
    scale2_label = torch.cat(tensor_list[3:6] , dim = 0)
    scale1_label = torch.cat(tensor_list[6:] , dim = 0)
    #print("scale1_labelの次元数\n",scale1_label.dim())
    #print("scale1_labelのサイズ\n",scale1_label.size())
    return scale3_label , scale2_label , scale1_label  

  #__getitem__メソッド：指定されたインデックスに対応するデータとラベルを返すためのメソッド
  def __getitem__(self , idx):
    img_path = self.img_list[idx]
    label_path = self.label_list[idx]
    #print(label_list[1])
    bbox_list = self.get_label(label_path) #coco128データセットの中のlabelsのリスト
    #画像を読み込んで、
    img = cv2.imread(img_path)
    img = cv2.resize(img , (self.img_size , self.img_size))
    #print(img.shape)
    img = Image.fromarray(img)
    #print(type(img)) PIL.Image.image
    img = self.transform(img)
    #print(type(img)) torch.Tensor
    #print(img.shape)
    scale3_label , scale2_label , scale1_label = self.label2tensor(bbox_list)
    
    return img , scale3_label , scale2_label , scale1_label
    
  def __len__(self):
    return len(self.img_list)


###データセットの確認
import numpy as np

def visualization(y_pred,anchor,img_size,conf = 0.5,is_label = False):
  size = y_pred.shape[2]
  anchor_size = anchor[anchor["type"] == size]
  #print(anchor_size)
  bbox_list = []
  for i in range(3):
    a = anchor_size.iloc[i] ##指定されたインデックス番号の値を抽出
    #print(a)
    grid = img_size/size ##
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
        img = FF.to_pil_image(img) #tensor型からPILイメージに変換
        img = np.asarray(img)
        axs[0, i].imshow(img)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        cv2.imwrite('anchor_dataset/out_'+str(n+1)+'.jpg',img)

#メイン
#from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
img_size = 416

#print(anchor)
import torchvision.transforms as T
class_n = 80
img_list = sorted(glob.glob(os.path.join("/home/wada_docker/Documents/YOLOv3/coco128/images/train2017","*")))

label_list = sorted(glob.glob(os.path.join("/home/wada_docker/Documents/YOLOv3/coco128/labels/train2017","*")))
transform = T.Compose([T.ToTensor()]) #tensorに変更、T.composeは複数のtransformを同時に行うときに使う。
"""
Datasetにおいて「def __len__()」と「def __getitem__()」は必須。
Transform(dataの前処理)によるerrorを確認するため。
transformによるerrorは参照して初めて出力される。
Transformの処理に不安があるならば必ず事前に参照出力させておくとそこでエラー確認ができる
"""
img_data = ("/home/wada_docker/Documents/YOLOv3/coco128/images/train2017/000000000009.jpg")
label_data = ("/home/wada_docker/Documents/YOLOv3/coco128/labels/train2017/000000000009.txt")
train_data = YOLOv3_Dataset(img_list,label_list,80,img_size , anchor,transform) 

#第一引数：Dataset
train_loader = DataLoader(train_data , batch_size = 1)

plt.rcParams['figure.max_open_warning'] = 200

for n , (img , scale3_label , scale2_label ,scale1_label) in enumerate(train_loader):
  path = img_list[n]
  #print(n)
  img = cv2.imread(path)[:,:,::-1]
  img = cv2.resize(img , (img_size , img_size))
  img = torch.tensor(img.transpose(2,0,1))
  
  preds = [scale3_label , scale2_label , scale1_label]
  #スケールごとに色を変える
  for color,pred in zip(["red","green","blue"],preds):
    bbox_list = visualization(pred,anchor,img_size,conf = 0.9,is_label = True)
    bbox_list_tensor = torch.tensor(bbox_list)
    #print(bbox_list)
    #print(bbox_list_tensor.shape)
    if bbox_list_tensor.shape[0] != 0:
      img = draw_bounding_boxes(img, bbox_list_tensor ,colors=color, width=1)
  
  #show(img)
  if n==20:
    break
