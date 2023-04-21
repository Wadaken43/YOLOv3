import cv2
import numpy
import os
import glob
from sklearn.cluster import KMeans
import pandas as pd


  #データセットを読み込む
cocodata_path = "/home/wada/YOLOv3/coco128/labels/train2017"
label_list = glob.glob(os.path.join(cocodata_path, '*txt'))
#print(label_list)

#データを扱いやすい形に変える
bbox_dict = {'width':[] , 'height':[]} #辞書型
bbox_list = []
for path in label_list:
  with open(path , mode='r',newline='\n') as f: #
      #print(f)
      for s_line in f:
        #print(s_line)
        bbox = [float(x) for x in s_line.rstrip('\n').split(' ')] #改行文字を削除して、空白文字に基づいて行を分割する。
        print(bbox)
        bbox_dict['width'].append(bbox[3])
        bbox_dict['height'].append(bbox[4])
        bbox_list.append(bbox[3:5]) #3番目以上5番目未満
df = pd.DataFrame(bbox_dict)
#print(bbox_list)
#print(df)
#print(df.head())  ###

#K-Means法によるクラスタ分析
y_km = KMeans(n_clusters=9,            
            init='random',  #クラスタセンサの初期化方法
            n_init=10,      #セントロイドのシードを変えて施行する回数        
            max_iter=300,   #1回施行あたりの最大反復回数         
            tol=1e-04,      #収束の許容範囲         
            random_state=0).fit_predict(bbox_list) #特徴量Xをクラスタリングし、結果を返す。   

df['cluster'] = y_km #新しい列としてDataFrame"df"に追加します。//実行結果は何番のグループに属しているのかを示しています。
#print(df.head())  ###
#print(df)

#9つのクラスタそれぞれについて、「幅」「高さ」「面積」特徴の平均値を計算する。
anchor_dict = {"width":[],"height":[],"area":[]}
for i in range(9):
  anchor_dict["width"].append(df[df["cluster"] == i].mean()["width"])
  anchor_dict["height"].append(df[df["cluster"] == i].mean()["height"])
  anchor_dict["area"].append(df[df["cluster"] == i].mean()["width"]*df[df["cluster"] == i].mean()["height"])
print(anchor_dict)

#
anchor = pd.DataFrame(anchor_dict).sort_values('area', ascending=False)
anchor["type"] = [13 ,13 ,13 ,26 ,26 ,26 ,52, 52,52 ]
print(anchor)



