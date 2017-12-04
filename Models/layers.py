import tensorflow as tf
import numpy as np

def variable_on_cpu(name, shape, initializer, is_training=True, dtype=tf.float32):
  '''
  Create a shareable variable.
  '''

  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005), dtype=dtype)
  return var

# Thomas's hot new activation function
def selu(x, name):
  with tf.name_scope(name) as scope:
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

def conv2d_layer(X, shape, scope, strides=(1, 1, 1, 1), layer_count=0, padding='SAME', is_training=True, act_func=tf.nn.relu):
  '''
  Create a convolution layer. Remember shape is [filter_height, filter_width, in_channels, out_channels]
  '''

  kernel = variable_on_cpu(
    'kernel' + str(layer_count), shape, tf.contrib.layers.variance_scaling_initializer(), is_training=is_training
  )
  conv = tf.nn.conv2d(X, kernel, strides, padding=padding)
  biases = variable_on_cpu('b' + str(layer_count), [shape[-1]], tf.constant_initializer(0.0), is_training=is_training)
  activation = tf.nn.bias_add(conv, biases)
  if act_func is not None:
    activation = act_func(activation, name=scope.name + str(layer_count))
    tf.summary.histogram('{}/activations'.format(scope.name + str(layer_count)), activation)
  tf.summary.scalar(
    '{}/sparsity'.format(scope.name + str(layer_count)), tf.nn.zero_fraction(activation)
  )
  return activation

def conv2d_bn_layer(X, shape, scope, strides=(1, 1, 1, 1), layer_count=0, padding='SAME', is_training=True, act_func=tf.nn.relu):
  '''
  Create a convolution layer with batch norm. Remember shape is [filter_height, filter_width, in_channels, out_channels]
  '''

  kernel = variable_on_cpu(
    'kernel' + str(layer_count), shape, tf.contrib.layers.variance_scaling_initializer(), is_training=is_training
  )
  conv = tf.nn.conv2d(X, kernel, strides, padding=padding)
  biases = variable_on_cpu('b' + str(layer_count), [shape[-1]], tf.constant_initializer(0.0), is_training=is_training)
  activation = tf.nn.bias_add(conv, biases)
  if act_func is not None:
    activation = act_func(activation, name=scope.name + str(layer_count))
    tf.summary.histogram('{}/activations'.format(scope.name + str(layer_count)), activation)
  activation = tf.contrib.layers.batch_norm(activation, fused=True, data_format="NHWC")
  tf.summary.scalar(
    '{}/sparsity'.format(scope.name + str(layer_count)), tf.nn.zero_fraction(activation)
  )
  return activation

def upsampling_2d_layer(X, size, name):
  '''
  Create an upsampling 2D layer.
  '''

  H, W, _ = X.get_shape().as_list()[1:]
  H_multi, W_multi = size
  target_H = H * H_multi
  target_W = W * W_multi
  return tf.image.resize_nearest_neighbor(X, (target_H, target_W), name="upsample_"+name)

def max_pool_layer(X, window_size, strides, scope, padding='SAME'):
  '''
  Create a max pooling layer.
  '''

  return tf.nn.max_pool(X, ksize=window_size, strides=strides,
                        padding=padding, name=scope.name)

def unet_down_layer_group(X, scope_name, features_start, features_end, act_func=tf.nn.relu, is_training=True):
  with tf.variable_scope(scope_name) as scope:
    down = conv2d_bn_layer(X, shape=[3, 3, features_start, features_end], scope=scope,
                          layer_count=1, act_func=act_func, is_training=is_training)
    down = conv2d_bn_layer(down, shape=[3, 3, features_end, features_end], scope=scope,
                          layer_count=2, act_func=act_func, is_training=is_training)
    down_pool = max_pool_layer(down, window_size=(1, 2, 2, 1), strides=(1, 2, 2, 1), scope=scope)
  return down, down_pool

def unet_center_layer_group(X, scope_name, features_start, features_end, act_func=tf.nn.relu, is_training=True):
  with tf.variable_scope(scope_name) as scope:
    center = conv2d_bn_layer(X, shape=[3, 3, features_start, features_end], scope=scope,
                          layer_count=1, act_func=act_func, is_training=is_training)
    center = conv2d_bn_layer(center, shape=[3, 3, features_end, features_end], scope=scope,
                          layer_count=2, act_func=act_func, is_training=is_training)
  return center

def unet_up_layer_group(X, scope_name, features_start, features_end, act_func=tf.nn.relu, is_training=True, mirror_down=None):
  with tf.variable_scope(scope_name) as scope:
    up = upsampling_2d_layer(X, (2, 2), name=scope.name)
    up = tf.concat([mirror_down, up], axis=-1, name="concat_" + scope.name)
    features_start = features_start + int(mirror_down.shape[-1])
    up = conv2d_bn_layer(up, shape=[3, 3, features_start, features_end], scope=scope,
                       layer_count=1, act_func=act_func, is_training=is_training)
    up = conv2d_bn_layer(up, shape=[3, 3, features_end, features_end], scope=scope,
                       layer_count=2, act_func=act_func, is_training=is_training)
    up = conv2d_bn_layer(up, shape=[3, 3, features_end, features_end], scope=scope,
                       layer_count=3, act_func=act_func, is_training=is_training)
  return up


def fc_layer(X, n_in, n_out, scope, is_training=True, act_func=None):
  '''
  Create a fully connected (multi-layer perceptron) layer.
  '''

  weights = variable_on_cpu(
    'W', [n_in, n_out], tf.contrib.layers.xavier_initializer(dtype=tf.float64), is_training=is_training, dtype=tf.float64
  )
  biases = variable_on_cpu('b', [n_out], tf.constant_initializer(0.0), is_training=is_training, dtype=tf.float64)
  if act_func is not None:
    activation = act_func(tf.matmul(X, weights) + biases, name=scope.name)
  else:
    activation = tf.add(tf.matmul(X, weights), biases, name=scope.name)
  tf.summary.histogram('{}/activations'.format(scope.name), activation)
  tf.summary.scalar(
    '{}/sparsity'.format(scope.name), tf.nn.zero_fraction(activation)
  )
  return activation

def fc_dropout_layer(X, n_in, n_out, scope, keep_prob, is_training=True, act_func=None):
  '''
  Create a fully connected (multi-layer perceptron) layer, with dropout
  '''

  weights = variable_on_cpu(
    'W', [n_in, n_out], tf.contrib.layers.xavier_initializer(dtype=tf.float64), is_training=is_training, dtype=tf.float64
  )
  biases = variable_on_cpu('b', [n_out], tf.constant_initializer(0.0), is_training=is_training, dtype=tf.float64)
  if act_func is not None:
    activation = act_func(tf.matmul(X, weights) + biases, name=scope.name)
  else:
    activation = tf.add(tf.matmul(X, weights), biases, name=scope.name)
  post_dropout = tf.nn.dropout(activation, keep_prob)
  tf.summary.histogram('{}/activations'.format(scope.name), activation)
  tf.summary.scalar(
    '{}/sparsity'.format(scope.name), tf.nn.zero_fraction(activation)
  )
  return post_dropout

def fc_bn_layer(X, n_in, n_out, scope, is_training=True, act_func=None):
  '''
  Create a fully connected layer with batch normalization
  '''

  weights = variable_on_cpu(
    'W', [n_in, n_out], tf.contrib.layers.xavier_initializer(dtype=tf.float32), is_training=is_training
  )
  biases = variable_on_cpu('b', [n_out], tf.constant_initializer(0.0), is_training=is_training)
  fc = tf.add(tf.matmul(X, weights), biases, name=scope.name)
  fc = tf.contrib.layers.batch_norm(fc, center=True, scale=True, is_training=is_training, scope=scope)
  if act_func is not None:
    activation = act_func(fc, name=scope.name)
  else:
    activation = fc
  tf.summary.histogram('{}/activations'.format(scope.name), activation)
  tf.summary.scalar(
    '{}/sparsity'.format(scope.name), tf.nn.zero_fraction(activation)
  )
  return activation