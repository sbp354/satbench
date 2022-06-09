from torchvision import transforms, datasets
import random
import torch
from collections import Counter
import pandas as pd
import os
from PIL import Image
# from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset

from custom_transforms import AllRandomBlur, AllRandomNoise, AddGaussianNoise, AddGaussianBlur

def get_dataloaders(args, train=True, human=False):
	trainset, testset = None, None

	if args.dataset == 'cifar100':
		dataloaders = get_cifar_dataloaders(args, train)

	elif args.dataset == 'cifar10':
		dataloaders = get_cifar_dataloaders(args, train)

	elif args.dataset == 'imagenet':
		if train == False and human == True:
			dataloaders = get_imagenet_human_dataset(args)
		else:
			dataloaders = get_imagenet_dataloaders(args, train)

	else:
		raise ValueError('Invalid dataset name: {}'.format(args.dataset))

	if train:
		return dataloaders[0], dataloaders[1]
	return dataloaders
	
def get_imagenet_human_dataset(args):
	class HumanImagesDataset(Dataset):
		def __init__(self, root_dir, dirs = ['0', '1', '2', '3', '4'], transform=None):
			self.root_dir = root_dir
			self.transform = transform
			self.dirs = dirs
			self.classes = ['knife', 'keyboard', 'elephant', 'bicycle', 'airplane', 'clock', 
			'oven', 'chair', 'bear', 'boat', 'cat', 'bottle', 'truck', 'car', 'bird', 'dog']
			self.names = dict([(t, os.listdir(os.path.join(root_dir, t))) for t in os.listdir(root_dir) if t in self.dirs])
			
		def __len__(self):
			return len([f for t in self.names for f in self.names[t]])

		def get_t_from_i(self, idx):
			for t in self.dirs:
				if idx < len(self.names[t]):
					return t
				else:
					idx -= len(self.names[t])

			return None

		def __getitem__(self, idx):
			t = self.get_t_from_i(idx)
			
			image_path = os.path.join(self.root_dir, t, self.names[t][idx])

			if 'color_gray' not in image_path:
				mode = self.names[t][idx].split('_')[2]
			else:
				mode = self.names[t][idx].split('_')[3]

			# image = io.imread(image_path)
			image = np.array(Image.open(image_path).convert('RGB'))
			
			label = self.classes.index(self.names[t][idx].split('_')[-1].split('.')[0])

			if self.transform:
				image = self.transform(image)

			return image, label, mode

	if args.mode == 'color' or args.mode == 'gray':
		root_dir = '../datasets/ColorGraySplit'
	elif args.mode == 'noise':
		root_dir = '../datasets/NoiseSplit_gray_contrast0.2'
	elif args.mode == 'blur':
		root_dir = '../datasets/BlurSplit_color'

	dataset = HumanImagesDataset(root_dir, transform=transforms.ToTensor(), dirs = args.dirs)
	if args.subsample == False:
		dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
	else:
		print("TAKING SUBSET")
		idxs = random.sample(range(0, len(dataset)), len(dataset)//2)
		subset = Subset(dataset, idxs)
		print(len(subset))
		dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

	return dataloader


def get_imagenet_dataloaders(args, train):
	if train == True:
		transform_train, transform_test = create_transforms(args, train, imagesize=224)
	else:
		transform_test = create_transforms(args, train, imagesize=224)

	if train == True:
		trainset = datasets.ImageNet(
			'/imagenet/',
			# '/misc/vlgscratch4/LecunGroup/katrina/data/imagenet_split/',
			split='train',
			download=False,
			transform=transform_train)
		if args.subset:
			trainset_sub = torch.utils.data.Subset(trainset, range(50000))
		else:
			trainset_sub = trainset

	testset = datasets.ImageNet(
		'/imagenet/',
		# '/misc/vlgscratch4/LecunGroup/katrina/data/imagenet_split/',
		split='val',
		download=False,
		transform=transform_test
	)
	if args.subset:
		testset_sub = torch.utils.data.Subset(testset, range(5000))
	else:
		testset_sub = testset


	if train == True:
		# Compute frequencies of each of 16 categories and therefore, determine sampling weights
		if args.subset:
			train_targets = trainset.targets[:50000]
		else:
			train_targets = trainset.targets

		target_freqs1000 = dict(Counter(train_targets))
		sixteen_class_map_master = pd.read_csv('../datasets/imagenet_mapping.csv', header=0)
		sixteen_class_mapping = pd.Series(sixteen_class_map_master['sixteen_class_id'].values,
											index=sixteen_class_map_master['thousand_class_id']).to_dict()
		train_targets16 = [sixteen_class_mapping[i] for i in train_targets if i in sixteen_class_mapping]
		target_freqs16 = dict(Counter(train_targets16))
		weights1000 = []
		for i in list(set(train_targets)):
			if i in sixteen_class_mapping:
				w = 1. / (target_freqs16[sixteen_class_mapping[i]] * target_freqs1000[i])
			else:
				w = 0.
			weights1000.append(w)

		weights = [weights1000[i] for i in train_targets]

		train_sampler = torch.utils.data.WeightedRandomSampler(weights, len(trainset_sub), replacement=True)

		# create dataloaders
		trainloader = torch.utils.data.DataLoader(
			trainset_sub,
			batch_size=args.batch_size,
			sampler=train_sampler,
			num_workers=args.workers 
		)

	testloader = torch.utils.data.DataLoader(
		testset_sub,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.workers  
	)

	if train:
		return trainloader, testloader
	return testloader


def get_cifar_dataloaders(args, train=True):
	transform_train, transform_test = create_transforms(args, train, imagesize=32)

	dataset = datasets.CIFAR10 if args.dataset == 'cifar10' else datasets.CIFAR100

	trainset = dataset(
		root='../data',
		train=True,
		download=True,
		transform=transform_train
	)
	testset = dataset(
		root='../data',
		train=False,
		download=True,
		transform=transform_test
	)

	trainloader = torch.utils.data.DataLoader(
		trainset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.workers,
		pin_memory=True
	)
	testloader = torch.utils.data.DataLoader(
		testset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.workers,
		pin_memory=True
	)

	return trainloader, testloader


def create_transforms(args, train, imagesize):
	if train == True:
		transform_train = [
			transforms.Resize((imagesize, imagesize)),
			transforms.RandomCrop(imagesize, padding=4),
			transforms.RandomHorizontalFlip(),
		]
		if args.extraaug:
			transform_train += [
				transforms.RandomVerticalFlip(),
				transforms.RandomRotation(15.0) # random rotation of (-15, 15) degrees
			]

	transform_test = [
		transforms.Resize((imagesize, imagesize)),
	]

	if args.mode == 'gray' or (args.mode in ['noise', 'blur'] and 'cifar' in args.dataset) or (args.mode == 'noise' and args.dataset == 'imagenet'):
		if train == True:
			transform_train.append(transforms.Grayscale(num_output_channels=3))
		transform_test.append(transforms.Grayscale(num_output_channels=3))
	
	if train:
		transform_train.append(transforms.ToTensor())
	transform_test.append(transforms.ToTensor())

	if args.mode == 'blur':
		if train == True:
			transform_train.append(AllRandomBlur(args.kernel_size, args.blur_std))
			transform_test.append(AllRandomBlur(args.kernel_size, args.blur_std))
		else:
			transform_test.append(AddGaussianBlur(args.kernel_size, args.blur_std))

	elif args.mode == 'noise':
		if train == True:
			transform_train.append(AllRandomNoise(0.0, args.noise_std, args.contrast))	
			transform_test.append(AllRandomNoise(0.0, args.noise_std, args.contrast))	
		else:
			transform_test.append(AddGaussianNoise(0.0, args.noise_std, args.contrast))

	if train == True:
		if args.extraaug:
			transform_train.append(transforms.Normalize(
				mean=[0.5, 0.5, 0.5],
				std=[0.5, 0.5, 0.5],
			))
			transform_test.append(transforms.Normalize(
				mean=[0.5, 0.5, 0.5],
				std=[0.5, 0.5, 0.5],
			))
		return transforms.Compose(transform_train), transforms.Compose(transform_test)

	return transforms.Compose(transform_test)
