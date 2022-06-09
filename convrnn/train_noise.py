import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from PIL import Image, ImageEnhance
import cv2
import urllib
import numpy as np
from tensorflow.keras.utils import to_categorical
import glob
from random import shuffle
import h5py
import torch
from torchvision import transforms
import math
import time
import os
import argparse


# tf.enable_v2_behavior()
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

from rcnn_sat import preprocess_image, bl_net
from load_data import load_dataset, load_dataset_h5, prep_pixels, prep_pixels_h5, load_dataset_ImageNet
from custom_transforms import all_random_noise, AllRandomNoise, AddGaussianNoise, add_gaussian_noise

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='noise-contrast-gray', type=str)
parser.add_argument('--color', default='gray', type=str)
parser.add_argument('--download-data', default=False, type=bool)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--dataset', default = 'ImageNet2012_16classes_rebalanced', type = str)
parser.add_argument('--data_dir', default = '/vast/sbp354/ImageNet', type = str)
parser.add_argument('--batch_size', default = 128, type = int)
parser.add_argument('--model_dir', default = '/scratch/sbp354/dynamic-nn/convrnn')
args = parser.parse_args()
print(args)

data_root = '../data/{}'.format(args.color)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
    if args.dataset == 'CIFAR10':
        if args.download_data == True:
            trainX, trainy, testX, testy = load_dataset()
            os.makedirs(data_root, exist_ok = True)
            prep_pixels_h5(trainX, trainy, testX, testy, data_root, args.color)
            args.download_data = False

        if args.download_data == False:
            trainX,trainy,testX,testy = load_dataset_h5(data_root)
    elif args.dataset == 'ImageNet2012_16classes_rebalanced':
        train_ds, test_ds, class_weights = load_dataset_ImageNet(train_dir = os.path.join(args.data_dir, 'train_16classes'),
                                                                val_dir = os.path.join(args.data_dir, 'val_16classes'),
                                                                batch_size = args.batch_size,
                                                                grayscale = True,
                                                                blur = False,
                                                                gauss_noise = True)
        train_ds = train_ds.take(int(len(train_ds)-1))
        #train_ds = train_ds.take(1)
        test_ds = train_ds.take(int(len(test_ds)-1))
    if args.dataset == 'CIFAR10':
        input_layer = tf.keras.layers.Input((128, 128, 3))
        model = bl_net(input_layer, classes=10, cumulative_readout=False)
    elif args.dataset == 'ImageNet2012_16classes_rebalanced':
        input_layer = tf.keras.layers.Input((224, 224, 3))
        model = bl_net(input_layer, classes=16, cumulative_readout=False)

    os.chdir(args.model_dir)
    if args.pretrained:
        model.load_weights('bl_imagenet.h5',skip_mismatch=True,by_name=True)

    ## Lets try fine tuning it
    # tf.keras.utils.plot_model(model,to_file='check.png')

    skip_layers = ['ReadoutDense','Sotfmax_Time_0','Sotfmax_Time_1',
                'Sotfmax_Time_2','Sotfmax_Time_3','Sotfmax_Time_4',
                'Sotfmax_Time_5','Sotfmax_Time_6','Sotfmax_Time_7']

    for layer in model.layers:
        if layer.name in skip_layers:
            layer.trainable = True
        else:
            layer.trainable = False

    # compile model with optimizer and loss
    """
    B, BL and parameter-matched controls (B-K, B-F and B-D) were trained for a total of 90 epochs 
    with a batch size of 100. B-U was trained using the same procedure but with a batch size of 64 
    due to its substantially larger number of parameters.

    The cross-entropy between the softmax of the network category readout and the labels 
    was used as the training loss. For networks with multiple readouts (BL and B-U), 
    we calculate the cross-entropy at each readout and average this across readouts. 
    Adam [64] was used for optimisation with a learning rate of 0.005 and epsilon parameter 0.1. 
    L2-regularisation was applied throughout training with a coefficient of 10−6.

    """
    cce = tf.keras.losses.CategoricalCrossentropy()
    opt = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

    checkpoint = ModelCheckpoint(os.path.join('model/ckpt', "pretrained_mp_noise_{}.hdf5".format(args.tag)),
                                    monitor='loss', verbose=1,
                                    save_best_only=True, mode='auto', period=1)


    train_data_augmentation = tf.keras.Sequential([layers.Rescaling(1./255),
                                            layers.RandomFlip(mode = 'horizontal'),
                                            layers.RandomCrop(224, 224),
                                            AllRandomNoise()])
    test_data_augmentation = tf.keras.Sequential([layers.Rescaling(1./255),
                                                  layers.Resizing(224, 224, crop_to_aspect_ratio = True),
                                                  AddGaussianNoise()])
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.map(lambda x, y: (train_data_augmentation(x, training = True), y))
    train_ds =train_ds.prefetch(buffer_size=AUTOTUNE)

    test_ds = test_ds.map(lambda x, y: (test_data_augmentation(x), y))

    history = model.fit(x=train_ds,
                        validation_data=test_ds,
                        steps_per_epoch=len(train_ds),
                        epochs=100,callbacks=[checkpoint],
                        class_weight = class_weights)

    model.save(os.path.join('model','{}_{}'.format(
        args.tag,
        time.strftime('%Y.%m.%d_%H.%M.%S')))
    )
