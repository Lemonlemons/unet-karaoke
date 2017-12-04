import random
import requests

import os

import numpy as np
import tensorflow as tf

import csv
import sys
import json

import librosa
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2
from scipy import signal, misc

def bytes_feature(value):
  '''
  Creates a TensorFlow Record Feature with value as a byte array.
  '''

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
  '''
  Creates a TensorFlow Record Feature with value as a 64 bit integer.
  '''

  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def prepare_train_and_test_data(configs, args, input_size):
  train_path = configs['TF_RECORDS_TRAIN']
  test_path = configs['TF_RECORDS_TEST']
  meta_path = configs['TF_RECORDS_META']

  print('Preparing Training and Testing Data')

  dataset_base = configs['DATASET']
  dataset_files = os.listdir(dataset_base)

  dataset_train_split, dataset_test_split = train_test_split(dataset_files, test_size=0.1, random_state=42)

  # Write the training set.
  training_count = prepare_unet_tfrecord(train_path, args, dataset_train_split, input_size, dataset_base, configs)

  # Write the testing set.
  test_count = prepare_unet_tfrecord(test_path, args, dataset_test_split, input_size, dataset_base, configs)

  with open(meta_path, 'w') as OUTPUT:
    OUTPUT.write('{},{}'.format(training_count, test_count))

  print('preprocessing completed')

def get_stats(configs):
  meta_file = configs['TF_RECORDS_META']
  with open(meta_file, 'r') as INPUT:
    META_DATA = INPUT.readline()
    tuple = [
      float(DATA_POINT) for DATA_POINT in META_DATA.split(',')
    ]
  return tuple

def prepare_unet_tfrecord(set_path, args, files, input_size, dataset_base, configs):

  writer = tf.python_io.TFRecordWriter(set_path)
  count = 0

  for data_file in files:
    print(data_file)
    vocal_stems = []
    nonvocal_stems = []

    track_base_path = dataset_base + data_file
    track_files = os.listdir(track_base_path)

    for stem in track_files:
      stem_path = track_base_path + '/' + stem
      samples, _ = librosa.load(stem_path, sr=configs['SAMPLE_RATE'], mono=True)
      if stem == 'vocals.wav':
        vocal_stems.append(samples)
      else:
        nonvocal_stems.append(samples)

    vocal_stems = np.array(vocal_stems, dtype=np.float32)
    nonvocal_stems = np.array(nonvocal_stems, dtype=np.float32)
    vocal_mix = mix_stems(vocal_stems)
    nonvocal_mix = mix_stems(nonvocal_stems)

    vocal_stft = stft(vocal_mix, configs)[1:]
    nonvocal_stft = stft(nonvocal_mix, configs)[1:]

    vocal_stft = vocal_stft.T
    nonvocal_stft = nonvocal_stft.T

    true_mix_stft = np.add(vocal_stft, nonvocal_stft)

    vocal_mag = np.absolute(vocal_stft)
    nonvocal_mag = np.absolute(nonvocal_stft)

    # handles when both the vocal_mag and nonvocal_mag are zero
    with np.errstate(divide='ignore', invalid='ignore'):
      true_masks = np.true_divide(vocal_mag, np.add(vocal_mag, nonvocal_mag), dtype=np.float32)
      true_masks[true_masks == np.inf] = 0
      true_masks = np.nan_to_num(true_masks, copy=False)

    hop_sample = configs['SAMPLE_HOP']

    mask_frames = sample_frames(true_masks, configs['STACKED_FRAMES'], hop_sample)
    mix_frames = sample_frames(true_mix_stft, configs['STACKED_FRAMES'], hop_sample)

    print("after sampling")
    print(mask_frames.shape)
    print(mix_frames.shape)

    # removing any portions of the vocal, nonvocal, or mix samples that just straight up zeros
    # zeros_array = np.zeros(true_mix_stft[0].shape)
    # mix_indexs_to_remove = []
    # for index, frame in enumerate(true_mix_stft):
    #   if np.array_equal(zeros_array, frame):
    #     mix_indexs_to_remove.append(index)
    # true_mix_stft = np.delete(true_mix_stft, mix_indexs_to_remove, 0)
    # true_masks = np.delete(true_masks, mix_indexs_to_remove, 0)
    #
    # mask_indexs_to_remove = []
    # for index, frame in enumerate(true_masks):
    #   if np.array_equal(zeros_array, frame):
    #     mask_indexs_to_remove.append(index)
    # true_mix_stft = np.delete(true_mix_stft, mask_indexs_to_remove, 0)
    # true_masks = np.delete(true_masks, mask_indexs_to_remove, 0)
    #
    # print('after removing silent windows')
    # print(true_mix_stft.shape)
    # print(true_masks.shape)

    # mixo = np.concatenate(mix_frames)
    # VOCAL_SIGNALS = istft(mixo.T, configs=configs)
    # librosa.output.write_wav("vocal_test.wav", VOCAL_SIGNALS, sr=configs['SAMPLE_RATE'])
    # sys.exit()

    count += len(mix_frames)

    for innerindex, window in enumerate(mix_frames):
      # Write the final input frames and binary_mask to disk.
      example = tf.train.Example(features=tf.train.Features(feature={
        'spectrograms': bytes_feature(window.flatten().tostring()),
        'masks': bytes_feature(mask_frames[innerindex].flatten().tostring())
      }))
      writer.write(example.SerializeToString())

  writer.close()
  return count

def mix_stems(stems):
  nstems, nsamples = stems.shape
  mix = np.zeros((nsamples, ))
  for stem in stems:
    # stem = np.divide(stem, np.amax(np.absolute(stem)))
    # stem = np.divide(stem, nstems)
    mix = np.add(mix, stem)
  return mix

def stft(mix, configs):
  fft_size = configs['FFT_SIZE']
  amount_of_overlap = fft_size - configs['HOP_SIZE']
  _, _, Zxx = signal.stft(mix, window='hann', nperseg=fft_size, noverlap=amount_of_overlap, nfft=fft_size)
  return Zxx

def istft(mix, configs):
  fft_size = configs['FFT_SIZE']
  amount_of_overlap = fft_size - configs['HOP_SIZE']
  _, x = signal.istft(mix, window='hann', nperseg=fft_size, noverlap=amount_of_overlap, nfft=fft_size)
  return x

def sample_frames(X, L, H):
  n_hops = int(np.round((X.shape[0] - L) / H))
  Y = []
  for hop in range(n_hops):
    hop_start = (hop * H)
    chunk = X[hop_start:hop_start + L, :]
    Y.append(chunk)
  return np.array(Y, dtype=np.float32)