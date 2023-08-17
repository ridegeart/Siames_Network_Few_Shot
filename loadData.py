# -*- coding: utf-8 -*-

import os
import pickle
from PIL import Image
import numpy as np
from skimage.transform import rotate, AffineTransform, warp, rescale
import cv2
import random
# %matplotlib inline
import matplotlib.pyplot as plt

"""
    Define transform types
"""
def affinetransform(image):
      transform = AffineTransform(translation=(-30,0))
      warp_image = warp(image,transform, mode="wrap")
      return warp_image
        
def anticlockwise_rotation(image):
      angle= random.randint(0,45)
      return rotate(image, angle)

def clockwise_rotation(image):
      angle= random.randint(0,45)
      return rotate(image, -angle)

class DataLoader(object):
    """
    Class for loading data from image files
    """

    def __init__(self, width, height, cells, data_path, output_path):

        self.width = width
        self.height = height
        self.cells = cells
        self.data_path = data_path
        self.output_path = output_path

    def transform(self,image):
      if random.random() > 0.5:
        image = affinetransform(image)
      if random.random() > 0.5:
        image = anticlockwise_rotation(image)
      if random.random() > 0.5:
        image = clockwise_rotation(image)

      return image

    def _open_image(self, path):
        """
        Using the Image library we open the image in the given path. The path must lead to a .jpg file.
        We then resize it to 105x105 like in the paper (the dataset contains 250x250 images.)
        Returns the image as a numpy array.
        """
        image = Image.open(path)
        #image = image.convert('L')
        image = image.resize((self.width, self.height))
        data = np.asarray(image)
        data = np.array(data, dtype='float64')
        return data

    def convert_image_to_array(self, person, image_num, data_path, predict=False):
        """
        Given a person, image number and datapath, returns a numpy array which represents the image.
        predict - whether this function is called during training or testing. If called when training, we must reshape
        the images since the given dataset is not in the correct dimensions.
        """
        image_path = os.path.join(data_path, person, f'{image_num}.jpg')
        image_data = self._open_image(image_path)
        image_data = image_data.reshape(self.width, self.height, self.cells)/255
        if not predict:
            image_data = image_data.reshape(self.width, self.height, self.cells)
        return image_data

    def load(self, set_name):
        """
        Writes into the given output_path the images from the data_path.
        dataset_type = train or test
        """
        print(set_name)
        print('Loading dataset...')
        x_first = []
        x_second = []

        for person_name in os.listdir(self.data_path):
          x_first = []
          person_num = len(os.listdir(os.path.join(self.data_path,person_name)))
          for i in range(2):
            if person_num > 2 :
              idx = random.randint(0,int(person_num)-1)
              while f'{str(idx)}.jpg' not in os.listdir(os.path.join(self.data_path,person_name)):
                idx = random.randint(0,int(person_num)-1)
              image = self.convert_image_to_array(person=person_name,image_num=str(idx),data_path=self.data_path)
              x_first.append(image)

            else:
              if i < person_num:
                num = os.listdir(os.path.join(self.data_path,person_name))
                image = self.convert_image_to_array(person=person_name,image_num=num[i].strip('.jpg'),data_path=self.data_path)
                x_first.append(image)
              else:
                idx = random.randint(0,int(person_num)-1)
                while f'{str(idx)}.jpg' not in os.listdir(os.path.join(self.data_path,person_name)):
                  idx = random.randint(0,int(person_num)-1)
                image_path = os.path.join(self.data_path, person_name, f'{str(idx)}.jpg')
                image_data = self._open_image(image_path)
                image_data = self.transform(image_data)
                image_data = np.asarray(image_data)
                image_data = np.array(image_data, dtype='float64')
                first_image_1 = image_data.reshape(self.width, self.height, self.cells)/255
                x_first.append(first_image_1)   
          x_first = np.stack(x_first, axis=0)
          x_second.append(x_first)
        x_second = np.stack(x_second, axis=0)

        print('Done loading dataset')
        with open(self.output_path, 'wb') as f:
            pickle.dump(x_second, f)

print("Loaded data loader")
  