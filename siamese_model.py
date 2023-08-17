import os
from keras import Input, Sequential, Model
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda, BatchNormalization, Activation,Dropout, Subtract
from keras.regularizers import l2
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from keras import backend, layers, metrics
import tensorflow as tf
import numpy as np
import random
from model.cbam import CBAM_block
from model.convnext import ConvNeXtTiny

def euclidean_distance(vects):
 x, y = vects
 return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
 shape1, shape2 = shapes
 return (shape1[0], 1)

class SiameseNetwork(object):
    def __init__(self, seed, width, height, cells, loss, metrics, optimizer, dropout_rate):
        """
        Seed - The seed used to initialize the weights
        width, height, cells - used for defining the tensors used for the input images
        loss, metrics, optimizer, dropout_rate - settings used for compiling the siamese model (e.g., 'Accuracy' and 'ADAM)
        """
        K.clear_session()
        self.load_file = None
        self.seed = seed
        self.initialize_seed()
        self.optimizer = optimizer

        # Define the matrices for the input images
        input_shape = (width, height, cells)
        left_input = Input(input_shape)
        right_input = Input(input_shape)

        # Get the CNN architecture as presented in the paper (read the readme for more information)
        model = self._get_architecture(input_shape)
        encoded_l = model(left_input)
        encoded_r = model(right_input)

        # Add a layer to combine the two CNNs
        #L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        #L1_siamese_dist = L1_layer([encoded_l, encoded_r])
        #L1_siamese_dist = Dropout(dropout_rate)(L1_siamese_dist)
        L1_siamese_dist = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([encoded_l, encoded_r])
        prediction = Dense(1, activation='sigmoid', bias_initializer=self.initialize_bias)(L1_siamese_dist)

        siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
        self.siamese_net = siamese_net
        self.siamese_net.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    def initialize_seed(self):
        """
        Initialize seed all for environment
        """
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def initialize_weights(self, shape, dtype=None):
        """
        Called when initializing the weights of the siamese model, uses the random_normal function of keras to return a
        tensor with a normal distribution of weights.
        """
        return K.random_normal(shape, mean=0.0, stddev=0.01, dtype=dtype, seed=self.seed)
    def initialize_bias(self, shape, dtype=None):
        """
        Called when initializing the biases of the siamese model, uses the random_normal function of keras to return a
        tensor with a normal distribution of weights.
        """
        return K.random_normal(shape, mean=0.5, stddev=0.01, dtype=dtype, seed=self.seed)
    def _get_architecture(self, inputshape):
        """
        resnet50
        """
        """
        resnet50 = ResNet50(include_top=False, input_shape=(105, 105, 3))
        output = layers.GlobalAveragePooling2D()(resnet50.output)
        base_model = tf.keras.models.Model(resnet50.input, output)
        """
        """
        convnext
        """
        """
        convnext = ConvNeXtTiny(model_name="convnext_tiny",
        include_top=True,
        include_preprocessing=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax")
        base_model = tf.keras.models.Model(convnext.input, output)
        """
        input_layer = Input(shape=inputshape)
        model = Conv2D(filters=64,
                   kernel_size=(10, 10),
                   kernel_initializer=self.initialize_weights,
                   kernel_regularizer=l2(2e-4),
                   name='Conv1')(input_layer)
        model = BatchNormalization()(model)
        model = Activation("relu")(model)
        model = MaxPooling2D()(model)

        model = CBAM_block(model)
        
        model = Conv2D(filters=128,
                   kernel_size=(7, 7),
                   kernel_initializer=self.initialize_weights,
                   bias_initializer=self.initialize_bias,
                   kernel_regularizer=l2(2e-4),
                   name='Conv2')(model)
        model = BatchNormalization()(model)
        model = Activation("relu")(model)
        model = MaxPooling2D()(model)

        model = Conv2D(filters=128,
                   kernel_size=(4, 4),
                   kernel_initializer=self.initialize_weights,
                   bias_initializer=self.initialize_bias,
                   kernel_regularizer=l2(2e-4),
                   name='Conv3')(model)
        model = BatchNormalization()(model)
        model = Activation("relu")(model)
        model = MaxPooling2D()(model)

        model = Conv2D(filters=256,
                   kernel_size=(4, 4),
                   kernel_initializer=self.initialize_weights,
                   bias_initializer=self.initialize_bias,
                   kernel_regularizer=l2(2e-4),
                   name='Conv4'
                   )(model)
        model = BatchNormalization()(model)
        model = Activation("relu")(model)

        model = Flatten()(model)
        model = Dense(4096,
                  activation='sigmoid',
                  kernel_initializer=self.initialize_weights,
                  kernel_regularizer=l2(2e-3),
                  bias_initializer=self.initialize_bias)(model)
        
        return Model(inputs=input_layer, outputs=model)