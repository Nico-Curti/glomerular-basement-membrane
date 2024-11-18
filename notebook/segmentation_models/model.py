#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Input

__author__  = ['Gianluca Carlini', 'Nico Curti']
__email__ = ['gianluca.carlini3@unibo.it', 'nico.curti2@unibo.it']

__all__ = ['efficientnetb3']


IMG_SIZE = 256

def decoder_block(inputs, skip, filters, stage):
    
  x = UpSampling2D(size=2, name=f"decoder_stage_{stage}_upsampling")(inputs)
  x = Concatenate(axis=-1, name=f"decoder_stage_{stage}concat")([x, skip])
  x = Conv2D(
    filters,
    kernel_size=(3, 3),
    kernel_initializer="he_uniform",
    padding="same",
    name=f"decoder_stage_{stage}a_conv",
    use_bias=False
  )(x)
  x = BatchNormalization(axis=-1, name=f"decoder_stage_{stage}a_bn")(x)
  x = Activation('relu', name=f"decoder_stage_{stage}a_relu")(x)
  x = Conv2D(
    filters,
    kernel_size=(3, 3),
    kernel_initializer="he_uniform",
    padding="same",
    name=f"decoder_stage_{stage}b_conv",
    use_bias=False
  )(x)
  x = BatchNormalization(axis=-1, name=f"decoder_stage_{stage}b_bn")(x)
  x = Activation('relu', name=f"decoder_stage_{stage}b_relu")(x)

  return x


def transp_conv_block(inputs, filters, stage):
    
  x = inputs
  x = UpSampling2D(size=2, name=f"decoder_stage_{stage}_upsampling")(inputs)
  x = Conv2D(
    filters,
    kernel_size=(3, 3),
    kernel_initializer="he_uniform",
    padding="same",
    name=f"decoder_stage_{stage}a_conv",
    use_bias=False
  )(x)
  x = BatchNormalization(axis=-1, name=f"decoder_stage_{stage}a_bn")(x)
  x = Activation('relu', name=f"decoder_stage_{stage}a_relu")(x)
  x = Conv2D(
    filters,
    kernel_size=(3, 3),
    kernel_initializer="he_uniform",
    padding="same",
    name=f"decoder_stage_{stage}b_conv",
    use_bias=False
  )(x)
  x = BatchNormalization(axis=-1, name=f"decoder_stage_{stage}b_bn")(x)
  x = Activation('relu', name=f"decoder_stage_{stage}b_relu")(x)

  return x


def efficientnetb3 (trainable=True):

  model = tf.keras.applications.efficientnet.EfficientNetB3(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling=None,
    classes=1,
    classifier_activation='linear'
  )
  model.trainable = trainable

  skip_connections = []

  for i in range(2, 7):
    if i == 5:
      continue
    skip_connections.append(model.get_layer(f'block{i}a_expand_activation').output)

  skip_connections.reverse()

  x = model.get_layer('top_activation').output
  filters = [256, 128, 64, 32]
  for i, skip in enumerate(skip_connections):
    x = decoder_block(x, skip, filters[i], stage=i)

  x = transp_conv_block(x, 16, stage=4)
  x = Conv2D(filters=1, kernel_size=(3, 3), padding='same', name='final_conv')(x)
  x = Activation('sigmoid', name='sigmoid')(x)


  model = tf.keras.models.Model(inputs=model.layers[3].input, outputs=x)

  return model
