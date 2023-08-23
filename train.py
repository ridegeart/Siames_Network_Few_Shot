# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.getcwd())
import pickle
import numpy as np
import cv2
from dataloader import Siamese_Loader
import pandas as pd
import tensorflow as tf
# %matplotlib inline
import matplotlib.pyplot as plt

from loss import mixed_loss,focal_loss,contrastive_loss,WeightedContrastiveLoss,AdaptiveCrossEntropyLoss
from keras.models import load_model
import csv
from sklearn.model_selection import train_test_split
from siamese_model import SiameseNetwork
from tensorflow.keras.optimizers import Adam,RMSprop,SGD
from keras.callbacks import EarlyStopping
import seaborn as sns

"""
load train/test data set
""" 

data_path = './data/'

with open("./data/LFWTrain.pkl","rb") as f:
  train = pickle.load(f)

b = np.vsplit(train, 5749)
b = np.array(b)
x_train_List, x_test_List= b[0:4599], b[4599:]

x_train_List = x_train_List.reshape(4599, 2, 105, 105, 1)/255.0
x_test_List = x_test_List.reshape(1150, 2, 105, 105, 1)/255.0

print('train shape：')
print(x_train_List.shape)
print('test shape：')
print(x_test_List.shape)

#siamese_net = load_model('/content/drive/MyDrive/lfwa/lfw2/best_c_76.0.h5', custom_objects={"W_init": W_init ,"b_init": b_init})

loss_every = 10
N = 10
batch_size = 32
best = 54.0
best_epoch = 0
cls_epoch_loss = []
cls_epoch_acc = []
dis_epoch_loss = []
dis_epoch_acc = []

test_epoch_loss = []
test_epoch_acc = []

for i in range(1,700):
  loader = Siamese_Loader(x_train_List)
  x_train_dis, y_train_dis = loader.get_traingPairs(batch_size=200)
  x_train_cls, y_train_cls = loader.make_oneshot_task(N=10)

  x_train_0, x_val_0, y_train_0, y_val_0 = train_test_split(x_train_dis[0], y_train_dis,
                                          test_size=0.2,
                                          random_state=0)
  x_train_1, x_val_1, y_train_1, y_val_1 = train_test_split(x_train_dis[1], y_train_dis,
                                          test_size=0.2,
                                          random_state=0)
  x_train_0 = np.array(x_train_0, dtype='float64')
  x_val_0 = np.array(x_val_0, dtype='float64')
  x_train_1 = np.array(x_train_1, dtype='float64')
  x_val_1 = np.array(x_val_1, dtype='float64')
  x_train = [x_train_0, x_train_1]
  x_val = [x_val_0, x_val_1]
  #print(x_train_0.shape,x_train_1.shape,x_val_0.shape,x_val_1.shape,y_train_0.shape, y_train_1.shape,y_val_0.shape,y_val_1.shape)
  if y_train_0[0] != y_train_1[0] and y_val_0[0] != y_val_1[0]:
      raise Exception("y train lists or y validation list do not equal")

  callback = []
  es = EarlyStopping(monitor='loss', min_delta=0.01, patience=5, mode='auto', verbose=1)
  callback.append(es)
  #distance
  siamese = SiameseNetwork(seed=0, width=105, height=105, cells=1, loss=contrastive_loss, metrics=['accuracy'],
              optimizer= Adam(0.00006), dropout_rate=0.4)
  #siamese.siamese_net.load_weights("./weight/best_LFW_floss.h5")
  history = siamese.siamese_net.fit(x_train, y_train_0, validation_data= (x_val,y_val_0), batch_size=batch_size, callbacks=callback, verbose=1)
  siamese.siamese_net.save('./weight/LFW_closs.h5')
  dis_epoch_loss.append(history.history['loss'][0])
  dis_epoch_acc.append(history.history['accuracy'][0])

  #classify
  siamese = SiameseNetwork(seed=0, width=105, height=105, cells=1, loss=focal_loss, metrics=['accuracy'],
              optimizer= Adam(0.00006), dropout_rate=0.4)
  history = siamese.siamese_net.fit(x_train_cls, y_train_cls, batch_size=batch_size, callbacks=callback, verbose=1)
  siamese.siamese_net.load_weights('./weight/LFW_closs.h5')
  cls_epoch_loss.append(history.history['loss'][0])
  cls_epoch_acc.append(history.history['accuracy'][0])

  #-------------Test------------------------------------

  loader2 = Siamese_Loader(x_test_List)
  val_acc = loader2.test_oneshot(siamese.siamese_net,N,k=550,verbose=True)
  print("iteration {}, validation acc : {:.7f}".format(i, val_acc))

  test_epoch_acc.append(val_acc)
  
  if val_acc >= best:
    best = val_acc
    best_epoch = i
    print("saving")
    siamese.siamese_net.save('./weight/best_LFW.h5')

  #num_epoch = int(i / loss_every), 

  print('starte picture for train')
  path='./graph_folder/'
  num_epoch=i, 
  epochs = [x for x in range(num_epoch[0])]
  print(len(epochs),len(dis_epoch_acc))

  dis_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":dis_epoch_acc, "Mode":['train']*(num_epoch[0])})
  dis_loss_df = pd.DataFrame({"Epochs":epochs, "Loss":dis_epoch_loss, "Mode":['train']*(num_epoch[0])})
  cls_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":cls_epoch_acc, "Mode":['train']*(num_epoch[0])})
  cls_loss_df = pd.DataFrame({"Epochs":epochs, "Loss":cls_epoch_loss, "Mode":['train']*(num_epoch[0])})
  train_accuracy_df = pd.concat([dis_accuracy_df,cls_accuracy_df])
  train_loss_df = pd.concat([dis_loss_df,cls_loss_df])
  test_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":test_epoch_acc, "Mode":['test']*(num_epoch[0])})
  #test_loss_df = pd.DataFrame({"Epochs":epochs, "Loss":test_epoch_loss, "Mode":['test']*(num_epoch[0])})

  sns.lineplot(data=train_accuracy_df.reset_index(), x='Epochs', y='Accuracy', hue='Mode')
  plt.title('Accuracy Graph')
  plt.savefig(path+f'train_accuracy_epoch.png')
  plt.clf()

  sns.lineplot(data=train_loss_df.reset_index(), x='Epochs', y='Loss', hue='Mode')
  plt.title('Loss Graph')
  plt.savefig(path+f'train_loss_epoch.png')
  plt.clf()

  sns.lineplot(data=test_accuracy_df.reset_index(), x='Epochs', y='Accuracy', hue='Mode')
  plt.title('Accuracy Graph')
  plt.savefig(path+f'test_accuracy_epoch.png')
  plt.clf()
  """
  sns.lineplot(data=test_loss_df.reset_index(), x='Epochs', y='Loss', hue='Mode')
  plt.title('Loss Graph')
  plt.savefig(path+f'test_loss_epoch.png')
  plt.clf()
  """
  
  print('picture save done  start next epoch')
  print(f'Best accuracy at {best_epoch}')

