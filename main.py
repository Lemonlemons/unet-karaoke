import argparse
import os

from Models.unet1024 import *
from Models.preprocessing import *


def main(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('--phase', default='train', help='Phase: Can be preprocess, train, test, val, or submission')
  parser.add_argument('--data_location', default='Data', help='Directory to save the tfrecords file in')
  parser.add_argument('--delete_old', default='True', help='Should we keep the old results and tensorboard files')
  parser.add_argument('--num_gpus', default=1, help='How many gpus would you like to use?')
  parser.add_argument('--model', default='unet1024', help='Model to use: Can be unet128, unet256, unet512, or unet1024')

  args = parser.parse_args()

  configs = {
    'TF_RECORDS_TRAIN': os.path.join(args.data_location, 'karaoke_train.tfrecords'),
    'TF_RECORDS_TEST': os.path.join(args.data_location, 'karaoke_test.tfrecords'),
    'TF_RECORDS_META': os.path.join(args.data_location, 'karaoke.meta'),
    'INPUT_SIZE': int(args.model[4:]),
    'DATASET': 'D:/Projects/deep-karaoke/Data/DSD100/Sources/',
    'FFT_SIZE': 2048,
    'HOP_SIZE': 512,
    'SAMPLE_RATE': 44100,
    'STACKED_FRAMES': 1024,
    'SAMPLE_HOP': 1024,
    'BATCH_SIZE': 6,
    'LEARNING_RATE': 0.0001,
    'NUMBER_OF_EPOCHS': 100,
    'EXAMPLE_FILE': 'D:/Projects/deep-karaoke/Data/ophelia.mp3'
  }

  # Preprocess training data
  if args.phase == 'preprocess':
    prepare_train_and_test_data(configs, args, configs['INPUT_SIZE'])

  # Train the model
  elif args.phase == 'train':
    stats = get_stats(configs)
    model = create_model(configs, args, stats)
    model.train()

  # Test the model
  elif args.phase == 'test':
    stats = get_stats(configs)
    model = create_model(configs, args, stats)
    model.test()

  elif args.phase == 'val':
    stats = get_stats(configs)
    model = create_model(configs, args, stats)
    model.val(configs)

  # Create Submission
  elif args.phase == "submission":
    stats = get_stats(configs)
    model = create_model(configs, args, stats)
    model.create_submission(configs)

  else:
    print("No valid phase selected")

def create_model(configs, args, stats):
  # Select the model you want to use
  if args.model == "unet1024":
    model = Unet1024(args, stats, configs)

  else:
    print("no valid model selected")
    sys.exit()

  return model

if __name__=="__main__":
  main(sys.argv)
