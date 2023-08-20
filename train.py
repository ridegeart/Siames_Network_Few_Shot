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

from loss import mixed_loss,focal_loss,l_softmax_loss,contrastive_loss,AdaptiveContrastiveLoss
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
train_path = os.path.join(data_path, "peopleDevTrain_color.pkl")  # A path for the train file
test_path = os.path.join(data_path, "peopleDevTest_color.pkl")
with open(train_path,"rb") as f:
  x_train_List = pickle.load(f)
with open(test_path,"rb") as f:
  x_test_List = pickle.load(f)

print('train shape：')
print(x_train_List.shape)
print('test shape：')
print(x_test_List.shape)

siamese = SiameseNetwork(seed=0, width=105, height=105, cells=3, loss=AdaptiveContrastiveLoss(), metrics=['accuracy'],
              optimizer= Adam(0.00006), dropout_rate=0.4)
#siamese_net = load_model('/content/drive/MyDrive/lfwa/lfw2/best_c_76.0.h5', custom_objects={"W_init": W_init ,"b_init": b_init})
#siamese.siamese_net.load_weights("./weight/best_c_2.h5")

loss_every = 10
batch_size = 32
N = 5
best = 35.9
best_epoch = 0
train_epoch_loss = []
train_epoch_acc = []

test_epoch_loss = []
test_epoch_acc = []

for i in range(1,300):
  loader = Siamese_Loader(x_train_List)
  x_train, y_train = loader.get_traingPairs(batch_size=200)
  #x_train, y_train = loader.make_oneshot_task(N=5)
  x_train_0, x_val_0, y_train_0, y_val_0 = train_test_split(x_train[0], y_train,
                                          test_size=0.2,
                                          random_state=0)
  x_train_1, x_val_1, y_train_1, y_val_1 = train_test_split(x_train[1], y_train,
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

  history = siamese.siamese_net.fit(x_train, y_train_0, validation_data= (x_val,y_val_0), callbacks=callback, verbose=1)
  #history = siamese.siamese_net.fit(x_train, y_train, batch_size=batch_size, callbacks=callback, verbose=1)
    
  train_epoch_loss.append(history.history['loss'][0])
  train_epoch_acc.append(history.history['accuracy'][0])

  #-------------Test------------------------------------
  loader2 = Siamese_Loader(x_test_List)
  val_loss, val_acc = siamese.siamese_net.test_on_batch(x_val, y_val_0)
  print("iteration {}, validation loss : {:.7f},validation acc : {:.7f}".format(i,val_loss, val_acc))

  test_epoch_loss.append(val_loss)
  test_epoch_acc.append(val_acc)
  """
  loader2 = Siamese_Loader(x_test_List)
  val_acc = loader2.test_oneshot(siamese.siamese_net,N,k=550,verbose=True)

  print("iteration {}, validation acc : {:.7f}".format(i, val_acc))

  test_epoch_acc.append(val_acc)
  """
  if val_acc >= best:
    best = val_acc
    best_epoch = i
    print("saving")
    siamese.siamese_net.save('./weight/best_cbam_closs.h5')

  #num_epoch = int(i / loss_every), 

  print('starte picture for train')
  path='./graph_folder/'
  num_epoch=i, 
  epochs = [x for x in range(num_epoch[0])]
  print(len(epochs),len(train_epoch_acc))

  train_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":train_epoch_acc, "Mode":['train']*(num_epoch[0])})
  train_loss_df = pd.DataFrame({"Epochs":epochs, "Loss":train_epoch_loss, "Mode":['train']*(num_epoch[0])})

  test_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":test_epoch_acc, "Mode":['test']*(num_epoch[0])})

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

  sns.lineplot(data=test_loss_df.reset_index(), x='Epochs', y='Loss', hue='Mode')
  plt.title('Loss Graph')
  plt.savefig(path+f'test_loss_epoch.png')
  plt.clf()
  
  print('picture save done  start next epoch')
  print(f'Best accuracy at {best_epoch}')

