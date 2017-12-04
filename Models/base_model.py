import csv

import tensorflow as tf
import time
import shutil
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants
import boto3
from .preprocessing import *
import requests
import cv2
from tqdm import tqdm
import pandas as pd


class BaseModel(object):
  def __init__(self, args, stats, configs):
    training_count, testing_count = stats
    self.mode = args.phase
    self.delete_old = args.delete_old == 'True'
    self.model = args.model
    self.model_file = 'Results/' + self.model + '/unet.ckpt'
    self.is_training = self.mode == 'train'
    self.is_testing = self.mode == 'test'
    if self.is_training:
      self.input_file = configs['TF_RECORDS_TRAIN']
    else:
      self.input_file = configs['TF_RECORDS_TEST']
    self.num_gpus = int(args.num_gpus)
    if self.is_training or self.is_testing:
      self.batch_size = configs['BATCH_SIZE']
    else:
      self.batch_size = 1

    self.visual_progress_factor = 5
    self.num_batches = int(np.ceil(float(training_count)/float(self.batch_size)) / self.visual_progress_factor)
    self.num_epochs = configs['NUMBER_OF_EPOCHS'] * self.visual_progress_factor
    self.stagnent_epochs_threshold = 6 * self.visual_progress_factor
    self.stagnent_lr_factor = 0.1
    self.lr = configs['LEARNING_RATE']

    self.input_pipeline_threads = 1
    self.graph_config = tf.ConfigProto(allow_soft_placement=True,
                                       log_device_placement=False,
                                       inter_op_parallelism_threads=5,
                                       intra_op_parallelism_threads=2)

    print('building Model')
    self.build(stats)

  def build(self, stats):
    raise NotImplementedError

  # training the chosen model
  def train(self):
    print('training Model')
    # Start training the model.
    # this session is for multi-gpu training
    with tf.Session(graph=self.GRAPH, config=self.graph_config) as SESSION:
      # Create Coordinator
      COORDINATOR = tf.train.Coordinator()

      # Initialize all the variables.
      SESSION.run(tf.global_variables_initializer())

      if self.delete_old:
        # remove old tensorboard and models files:
        shutil.rmtree('Results/'+self.model)
        os.makedirs('Results/'+self.model)
      else:
        # restore the session
        GRAPH_WRITER = tf.train.Saver()
        GRAPH_WRITER.restore(SESSION, self.model_file)

      shutil.rmtree('Tensorboard/' + self.model)
      os.makedirs('Tensorboard/' + self.model)

      # Start Queue Runners
      THREADS = tf.train.start_queue_runners(SESSION, COORDINATOR)
      # Create a tensorflow summary writer.
      SUMMARY_WRITER = tf.summary.FileWriter('Tensorboard/'+self.model, graph=self.GRAPH)
      # Create a tensorflow graph writer.
      GRAPH_SAVER = tf.train.Saver(tf.global_variables())

      TOTAL_DURATION = 0.0
      GLOBAL_STEP = 0
      BEST_DICE_LOSS = 0.0
      BEST_COST_VALUE = np.inf
      STAGNENT_EPOCHS = 0
      LEARNING_RATE = self.lr
      for EPOCH in range(self.num_epochs):
        DURATION = 0
        ERROR = 0.0
        START_TIME = time.time()
        for MINI_BATCH in range(self.num_batches):
          _, SUMMARIES, COST_VAL, DICE_LOSS = SESSION.run([
            self.APPLY_GRADIENT_OP, self.SUMMARIES_OP, self.COST, self.DICE_LOSS
          ], feed_dict={self.learning_rate: LEARNING_RATE})
          ERROR += COST_VAL
          GLOBAL_STEP += 1

        # Write the summaries to disk.
        SUMMARY_WRITER.add_summary(SUMMARIES, EPOCH)
        DURATION += time.time() - START_TIME
        TOTAL_DURATION += DURATION
        # Update the console.
        print('Epoch %d: loss = %.4f (%.3f sec), dice loss = %.8f' % (EPOCH, ERROR, DURATION, DICE_LOSS))
        # Check for stagnent epochs
        if BEST_COST_VALUE > ERROR:
          STAGNENT_EPOCHS = 0
          BEST_COST_VALUE = ERROR
          print('Saving Session, loss: ' + str(BEST_COST_VALUE))
          GRAPH_SAVER.save(SESSION, self.model_file)
        else:
          STAGNENT_EPOCHS += 1
        # Check if there is a learning plateau and the LR should be decreased
        if STAGNENT_EPOCHS >= self.stagnent_epochs_threshold:
          LEARNING_RATE = LEARNING_RATE * self.stagnent_lr_factor
          print("Reducing learning rate to: " + str(LEARNING_RATE))
          STAGNENT_EPOCHS = 0
        # Check if loss is good enough to end early
        if EPOCH + 1 == self.num_epochs or DICE_LOSS > 0.997:
          print(
            'Done training for %d epochs. (%.3f sec) total steps %d' % (EPOCH, TOTAL_DURATION, GLOBAL_STEP)
          )
          break
      print('Training Done!')
      COORDINATOR.request_stop()
      COORDINATOR.join(THREADS)

  # get test accuracies of models
  def test(self):
    print('testing Model')
    with tf.Session(graph=self.GRAPH, config=self.graph_config) as SESSION:
      COORDINATOR = tf.train.Coordinator()
      SESSION.run(tf.global_variables_initializer())
      THREADS = tf.train.start_queue_runners(SESSION, COORDINATOR)

      # restore the session
      GRAPH_WRITER = tf.train.Saver()
      GRAPH_WRITER.restore(SESSION, self.model_file)

      for EPOCH in range(10):
        DICE_LOSS = SESSION.run(self.DICE_LOSS)
        # Update the console.
        print('Epoch %d: dice loss = %.8f' % (EPOCH, DICE_LOSS))
      COORDINATOR.request_stop()
      COORDINATOR.join(THREADS)

  # process file for validation
  def val(self, configs):
    print('validating model')
    samples, _ = librosa.load(configs['EXAMPLE_FILE'], sr=configs['SAMPLE_RATE'], mono=True)
    base_spec = stft(samples, configs)
    copied_spec = base_spec[:]
    trimmed_spec = copied_spec[1:]
    transposed_spec = trimmed_spec.T

    # this is to fill the end with zeros so it goes into the conv layers evenly
    size_of_zeros_array = configs['STACKED_FRAMES'] - (transposed_spec.shape[0] % configs['STACKED_FRAMES'])
    spacing_array = np.zeros([size_of_zeros_array, configs['STACKED_FRAMES']], dtype=np.float32)

    # sampled_spec = sample_frames(transposed_spec, configs['STACKED_FRAMES'], configs['SAMPLE_HOP'])
    transposed_spec = np.append(transposed_spec, spacing_array, axis=0)
    number_of_windows = transposed_spec.shape[0] / configs['STACKED_FRAMES']
    sampled_spec = np.split(transposed_spec, number_of_windows)

    masks = []
    with tf.Session(graph=self.GRAPH, config=self.graph_config) as SESSION:
      SESSION.run(tf.global_variables_initializer())

      # restore the session
      GRAPH_WRITER = tf.train.Saver()
      GRAPH_WRITER.restore(SESSION, self.model_file)

      for window in sampled_spec:
        mask = SESSION.run(self.Y, feed_dict={self.spectrograms: [window.flatten()]})
        masks.append(mask)

    big_mask = np.concatenate(masks)
    big_mask = np.reshape(big_mask, [np.prod([big_mask.shape[0], big_mask.shape[1]]), big_mask.shape[2]])
    big_mask = big_mask[:-(size_of_zeros_array)]
    big_mask = big_mask.T

    vocal_mask = big_mask[:]
    non_vocal_mask = np.subtract(1, big_mask)

    ones_array = np.full([vocal_mask.shape[1]], 1, dtype=np.float32)
    vocal_mask = np.insert(vocal_mask, 0, ones_array, axis=0)
    non_vocal_mask = np.insert(non_vocal_mask, 0, ones_array, axis=0)

    vocal_stft = np.multiply(vocal_mask, base_spec)
    non_vocal_stft = np.multiply(non_vocal_mask, base_spec)

    vocal_samples = istft(vocal_stft, configs)
    non_vocal_samples = istft(non_vocal_stft, configs)

    librosa.output.write_wav("vocal_test.wav", vocal_samples, sr=configs['SAMPLE_RATE'])
    librosa.output.write_wav("nonvocal_test.wav", non_vocal_samples, sr=configs['SAMPLE_RATE'])

  # custom loss functions
  def dice_loss(self, y_true, y_pred):
    smooth = 1.
    y_true_f = tf.contrib.layers.flatten(y_true)
    y_pred_f = tf.contrib.layers.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1) + smooth)

  def bce_dice_loss(self, y_true, y_pred, y_pred_pre_sigmoid):
    # return tf.contrib.keras.backend.binary_crossentropy(y_true, y_pred) + (1 - self.dice_loss(y_true, y_pred))
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred_pre_sigmoid) + (1 - self.dice_loss(y_true, y_pred))

  # read inputs from tfrecords
  def read_inputs(self, file_paths, spectrogram_shape, masks_shape, batch_size=64,
                  capacity=1000, min_after_dequeue=900, num_threads=2, is_training=True):

    with tf.name_scope('input'):
      # if training we use an input queue otherwise we use placeholders
      if is_training:
        # Create a file name queue.
        filename_queue = tf.train.string_input_producer(file_paths)
        reader = tf.TFRecordReader()
        # Read an example from the TFRecords file.
        _, example = reader.read(filename_queue)
        features = tf.parse_single_example(example, features={
          'spectrograms': tf.FixedLenFeature([], tf.string),
          'masks': tf.FixedLenFeature([], tf.string)
        })
        # Decode sample
        spectrogram = tf.decode_raw(features['spectrograms'], tf.float32)
        spectrogram.set_shape(spectrogram_shape)
        mask = tf.decode_raw(features['masks'], tf.float32)
        mask.set_shape(masks_shape)

        self.spectrograms, self.masks = tf.train.shuffle_batch(
          [spectrogram, mask], batch_size=batch_size,
          capacity=capacity, min_after_dequeue=min_after_dequeue,
          num_threads=num_threads,
        )
      else:
        spectrogram_shape = [batch_size] + spectrogram_shape
        self.spectrograms = tf.placeholder(tf.float32, shape=spectrogram_shape)
        masks_shape = [batch_size] + masks_shape
        self.masks = tf.placeholder(tf.float32, shape=masks_shape)

      return self.spectrograms, self.masks