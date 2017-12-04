from .base_model import *
from .layers import *

import numpy as np
import tensorflow as tf

class Unet1024(BaseModel):

  def build(self, stats):
    # image shape
    self.image_input_shape = [1024, 1024, 1]
    # mask shape
    self.mask_input_shape = [1024, 1024, 1]
    # number of classes
    self.num_classes = 1

    # Build our dataflow graph.
    self.GRAPH = tf.Graph()
    with self.GRAPH.as_default():
      self.learning_rate = tf.placeholder(tf.float32, shape=[])

      SPECTROGRAMS, MASKS = self.read_inputs([self.input_file], spectrogram_shape=[np.prod(self.image_input_shape)],
        masks_shape=[np.prod(self.mask_input_shape)],
        batch_size=self.batch_size, capacity=100, min_after_dequeue=self.batch_size,
        num_threads=self.input_pipeline_threads, is_training=(self.is_training or self.is_testing)
      )

      training_set_count, test_set_count = stats

      print('training set count: ' + str(training_set_count))
      print('test set count: ' + str(test_set_count))

      SPECTROGRAMS = tf.reshape(SPECTROGRAMS, [self.batch_size] + self.image_input_shape)
      MASKS = tf.reshape(MASKS, [self.batch_size] + self.mask_input_shape)

      # MAG_SPECS = tf.abs(SPECTROGRAMS)

      relu = tf.nn.relu

      # Build convolutional layers.
      down0b, down0b_pool = unet_down_layer_group(SPECTROGRAMS, 'conv_down_8', 1, 8, relu, is_training=self.is_training)
      down0a, down0a_pool = unet_down_layer_group(down0b_pool, 'conv_down_16', 8, 16, relu, is_training=self.is_training)
      down0, down0_pool = unet_down_layer_group(down0a_pool, 'conv_down_32', 16, 32, relu, is_training=self.is_training)
      down1, down1_pool = unet_down_layer_group(down0_pool, 'conv_down_64', 32, 64, relu, is_training=self.is_training)
      down2, down2_pool = unet_down_layer_group(down1_pool, 'conv_down_128', 64, 128, relu, is_training=self.is_training)
      down3, down3_pool = unet_down_layer_group(down2_pool, 'conv_down_256', 128, 256, relu, is_training=self.is_training)
      down4, down4_pool = unet_down_layer_group(down3_pool, 'conv_down_512', 256, 512, relu, is_training=self.is_training)

      center = unet_center_layer_group(down4_pool, 'center', 512, 1024, relu, is_training=self.is_training)

      up4 = unet_up_layer_group(center, 'conv_up_512', 1024, 512, relu, is_training=self.is_training, mirror_down=down4)
      up3 = unet_up_layer_group(up4, 'conv_up_256', 512, 256, relu, is_training=self.is_training, mirror_down=down3)
      up2 = unet_up_layer_group(up3, 'conv_up_128', 256, 128, relu, is_training=self.is_training, mirror_down=down2)
      up1 = unet_up_layer_group(up2, 'conv_up_64', 128, 64, relu, is_training=self.is_training, mirror_down=down1)
      up0 = unet_up_layer_group(up1, 'conv_up_32', 64, 32, relu, is_training=self.is_training, mirror_down=down0)
      up0a = unet_up_layer_group(up0, 'conv_up_16', 32, 16, relu, is_training=self.is_training, mirror_down=down0a)
      up0b = unet_up_layer_group(up0a, 'conv_up_8', 16, 8, relu, is_training=self.is_training, mirror_down=down0b)

      with tf.variable_scope('output') as scope:
        pred_y_pre_sigmoid = conv2d_layer(up0b, shape=[1, 1, 8, 1], scope=scope,
                             layer_count=1, act_func=None, is_training=self.is_training)
        self.Y = tf.sigmoid(pred_y_pre_sigmoid)

      # Compute the cross entropy loss.
      self.COST = tf.reduce_mean(self.bce_dice_loss(MASKS, self.Y, pred_y_pre_sigmoid))
      tf.summary.scalar("cost", self.COST)

      # Watch the Dice Loss by itself
      self.DICE_LOSS = tf.reduce_mean(self.dice_loss(MASKS, self.Y))
      tf.summary.scalar("dice loss", self.DICE_LOSS)

      # Compute gradients.
      OPTIMIZER = tf.train.RMSPropOptimizer(self.learning_rate)
      GRADIENTS = OPTIMIZER.compute_gradients(self.COST)

      # Apply gradients.
      self.APPLY_GRADIENT_OP = OPTIMIZER.apply_gradients(GRADIENTS)

      # Add histograms for gradients to our TensorBoard logs.
      for GRADIENT, VAR in GRADIENTS:
        if GRADIENT is not None:
          tf.summary.histogram('{}/gradients'.format(VAR.op.name), GRADIENT)

      # Collect the TensorBoard summaries.
      self.SUMMARIES_OP = tf.summary.merge_all()