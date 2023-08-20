import keras
from sklearn.preprocessing import  OneHotEncoder
import tensorflow as tf
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.losses import Loss

def l_softmax_loss(y_true, y_pred, m=4, depth=10):
    depth = len(y_true)
    y_true = tf.cast(y_true, tf.int32)
    y_true = K.one_hot(y_true , depth)  # convert to one-hot labels
    cos_t = y_pred
    sin_t = tf.sqrt(1 - K.square(cos_t))
    cos_mt = tf.cos(m * cos_t - m)  # L-Softmax formula
    sin_mt = tf.sin(m * cos_t - m)
    one_hot = tf.cast(y_true, 'float32')
    loss = K.mean(one_hot * (sin_mt * (m / sin_t) + cos_mt - cos_t))
    return loss

def focal_loss(y_true, y_pred ,alpha=0.25,gamma=2.0):
        gamma_tensor = K.ones_like(y_true) * gamma
        alpha_tensor = K.ones_like(y_true) * alpha
        
        ce_loss = K.binary_crossentropy(y_true, y_pred)
        p_t = K.exp(-ce_loss)
        focal_loss = alpha_tensor * K.pow((1.0 - p_t), gamma_tensor) * ce_loss

        return focal_loss

def contrastive_loss(y_true, y_pred ,margin=1.0):
  y_true = tf.cast(y_true, tf.float32)
  #return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))#1
  return K.mean((1-y_true) * K.square(y_pred) + (y_true) * K.square(K.maximum(margin - y_pred, 0)))#2

def mixed_loss(y_true, y_pred, alpha=0.25, gamma=2.0, margin=1.0, num_classes=20):
    """
    Mixed loss function that combines L-Softmax loss, Focal loss and Contrastive loss
    :param y_true: True labels
    :param y_pred: Predicted labels
    :param alpha: Weight of Focal loss
    :param gamma: Focusing parameter of Focal loss
    :param margin: Margin for Contrastive loss
    :param num_classes: Number of classes
    :return: Total loss
    """
    #num_classes = len(y_true)
    #print(len(y_true),y_true.shape,y_pred.shape)

    # L-Softmax loss
    #ls_loss = tf.reduce_mean(l_softmax_loss(y_true, y_pred, m=margin, depth=num_classes))
    #ls_loss = tf.reduce_mean(l_softmax_loss(y_true, y_pred, m=margin))
    
    # Focal loss
    #f_loss = tf.reduce_mean(focal_loss(y_true, y_pred, alpha, gamma))

    # Contrastive loss
    c_loss = tf.reduce_mean(contrastive_loss(y_true, y_pred, margin))

    # Combine losses
    total_loss = c_loss

    return total_loss

class AdaptiveContrastiveLoss(Loss):
    def __init__(self, initial_margin=1.0, margin_update_rate=0.01):
        super(AdaptiveContrastiveLoss, self).__init__()
        self.margin = initial_margin
        self.margin_update_rate = margin_update_rate

    def update_margin(self, margin_difficulty):
        self.margin += self.margin_update_rate * margin_difficulty

    def call(self, y_true, distances):
        # Calculate contrastive loss
        loss = y_true * tf.square(distances) + (1 - y_true) * tf.square(tf.maximum(self.margin - distances, 0.0))
        
        # Estimate difficulty based on the margin difference
        margin_difficulty = tf.square(self.margin - distances) * (1 - y_true)
        self.update_margin(margin_difficulty)

        return tf.reduce_mean(loss)