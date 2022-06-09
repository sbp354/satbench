import torch
from torchvision.utils import save_image
import pandas as pd

import os
import shutil

from utils import get_dataloaders

class Args:
	def __init__(self):
		self.depth = 18
		self.dataset = 'imagenet'
		self.class_num = 16
		self.mode = 'gray'
		self.epoch = 200
		self.lambda_KD = 0.5
		self.lr = 0.1
		self.weight_decay = 5e-4
		self.momentum = 0.9
		self.batch_size = 64
		self.workers = 2

args = Args()

trainloader, testloader = get_dataloaders(args)

if os.path.exists('dataloading'):
	shutil.rmtree('dataloading/')
os.makedirs('dataloading')

if (args.dataset=='imagenet') & (args.class_num==16):
	sixteen_class_map_master = pd.read_csv('../imagenet_mapping.csv', header=0)
	thousand_class_ids = sixteen_class_map_master['thousand_class_id'].values
	sixteen_class_mapping = pd.Series(sixteen_class_map_master['label'].values,
									index=sixteen_class_map_master['thousand_class_id']).to_dict()

for i, data in enumerate(trainloader):
	inputs, labels = data

	if args.dataset == "imagenet" and args.class_num == 16:
		idx_keep = torch.tensor([x.item() in thousand_class_ids for x in labels])
		labels = labels[idx_keep]
		inputs = inputs[idx_keep]

		sixteen_labels = [sixteen_class_mapping[x.item()] for x in labels]

		if inputs.size(0) == 0:
			continue

	for j in range(len(inputs)):
		image = inputs[j]
		sixteen_label = sixteen_labels[j]
		thousand_label = labels[j]

		save_image(image, 'dataloading/' + str(i) + '_' + str(j) + '_' + sixteen_label + '_' + str(thousand_label.item()) + '.png')

