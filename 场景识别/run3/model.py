import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import resnet50, vgg19

class _FlatAndClassfy(layers.Layer):
    def __init__(self, n_class):
        super().__init__()
        self.flat = layers.Flatten()
        self.f1 = layers.Dense(1024, use_bias=True, name='f1')
        self.f2 = layers.Dense(128, use_bias=True, name='f2')
        self.f_out = layers.Dense(n_class, use_bias=False, name='f_out')
    
    def __call__(self, x_in, dropout):
        x = self.flat(x_in)
        x = self.f1(x)
        x = tf.nn.relu(x)
        x = tf.nn.dropout(x, dropout)
        x = self.f2(x)
        x = tf.nn.relu(x)
        x = tf.nn.dropout(x, dropout)
        x = self.f_out(x)
        return x

class ResNet50(Model):
    def __init__(self, n_class, retrain=False, weights='imagenet'):
        super().__init__(name='resnet50')
        self.encoder = resnet50.ResNet50(include_top=False, weights=weights)
        self.encoder.trainable = retrain
        self.pool = layers.GlobalAveragePooling2D(name='pool')
        self.cls = _FlatAndClassfy(n_class)
    @tf.function
    def __call__(self, x_in, dropout=0.):
        if x_in.shape[-1] == 1:
            x = tf.concat([x_in, x_in, x_in], -1)
        x = self.encoder(x)
        x = self.pool(x)
        x = self.cls(x, dropout)
        return x

class VGG19(Model):
    def __init__(self, n_class, retrain=False, weights='imagenet'):
        super().__init__(name='vgg19')
        self.encoder = vgg19.VGG19(include_top=False, weights=weights)
        self.encoder.trainable = retrain
        self.pool = layers.GlobalAveragePooling2D(name='pool')
        self.cls = _FlatAndClassfy(n_class)

    def __call__(self, x_in, dropout=0.):
        if x_in.shape[-1] == 1:
            x = tf.concat([x_in, x_in, x_in], -1)
        x = self.encoder(x)
        x = self.pool(x)
        x = self.cls(x, dropout)
        return x