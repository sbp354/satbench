import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import sresnet
import torch.nn.functional as F

import time
import os
import json
from collections import Counter
import pandas as pd

from custom_transforms import AllRandomNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(100)
torch.cuda.manual_seed(100)

parser = argparse.ArgumentParser(description='SCAN Training')
parser.add_argument('--tag', default="noise-contrast-gray", type=str)
parser.add_argument('--depth', default=18, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--class_num', default=10, type=int)
parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--lambda_KD', default=0.5, type=float)
parser.add_argument('--noise-std', default=0.04, type=float)
parser.add_argument('--contrast', default=0.1, type=float)
args = parser.parse_args()
print(args)


def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs/3.0, dim=1)
    softmax_targets = F.softmax(targets/3.0, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


BATCH_SIZE = 64
LR = 0.1
WORKERS = 4

trainset, testset = None, None
if args.dataset == 'cifar100':
    print("dataset: CIFAR100")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        AllRandomNoise(0.0, args.noise_std, contrast=args.contrast),
    ])
    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        AllRandomNoise(0.0, args.noise_std, contrast=args.contrast),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root=args.data_root,
        train=True,
        download=True,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root=args.data_root,
        train=False,
        download=True,
        transform=transform_test
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS
    )

elif args.dataset == 'cifar10':
    print("dataset: CIFAR10")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        AllRandomNoise(0.0, args.noise_std, contrast=args.contrast),
    ])
    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        AllRandomNoise(0.0, args.noise_std, contrast=args.contrast),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=True,
        download=True,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=False,
        download=True,
        transform=transform_test
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS
    )

elif args.dataset == 'imagenet':
    print("dataset: IMAGENET")
    transform_train = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            AllRandomNoise(0.)
        ])
    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        AllRandomNoise(0.)
    ])

    trainset = torchvision.datasets.ImageNet(
        '/imagenet/',
        split='train',
        download=False,
        transform=transform_train)
    testset = torchvision.datasets.ImageNet(
        '/imagenet/',
        split='val',
        download=False,
        transform=transform_test
    )

    # Compute frequencies of each of 16 categories and therefore, determine sampling weights
    train_targets = trainset.targets
    target_freqs1000 = dict(Counter(train_targets))
    sixteen_class_map_master = pd.read_csv('../imagenet_mapping.csv', header=0)
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
    train_sampler = torch.utils.data.WeightedRandomSampler(weights, len(trainset), replacement=True)

    # create dataloaders
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=WORKERS 
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS  
    )

net = None
if args.depth == 9:
    net = sresnet.resnet9(num_classes=args.class_num, align="CONV")
    print("using resnet 9")
if args.depth == 18:
    net = sresnet.resnet18(num_classes=args.class_num, align="CONV")
    print("using resnet 18")
if args.depth == 34:
    net = sresnet.resnet34(num_classes=args.class_num, align="CONV")
    print("using resnet 34")

net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, weight_decay=5e-4, momentum=0.9)

if __name__ == "__main__":
    best_acc = 0
    # create experiment directory
    expt_id = "{}_{}_resnet{}_cifar{}_ep{}_lkd{}".format(
        args.tag,
        time.strftime('%Y.%m.%d_%H.%M.%S'),
        args.depth,
        args.class_num,
        args.epoch,
        args.lambda_KD
    )
    root = os.path.join('expt/train/imagenet', expt_id)
    os.makedirs(root, exist_ok=True)

    if (args.dataset=='imagenet') & (args.class_num==16):
        sixteen_class_map_master = pd.read_csv('../imagenet_mapping.csv', header=0)
        thousand_class_ids = sixteen_class_map_master['thousand_class_id'].values
        sixteen_class_mapping = pd.Series(sixteen_class_map_master['sixteen_class_id'].values,
                                        index=sixteen_class_map_master['thousand_class_id']).to_dict()
    with open(os.path.join(root, "config.json"), 'w') as f:
        json.dump(vars(args), f)

    print("Start Training")  # 定义遍历数据集的次数
    for epoch in range(args.epoch):
        correct4, correct3, correct2, correct1, correct0 = 0, 0, 0, 0, 0
        predicted4, predicted3, predicted2, predicted1, predicted0 = 0, 0, 0, 0, 0
        if epoch in [75, 130, 180]:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data

            if args.dataset == "imagenet" and args.class_num == 16:
                idx_keep = torch.tensor([x.item() in thousand_class_ids for x in labels])
                labels = labels[idx_keep]
                inputs = inputs[idx_keep]

                labels = torch.tensor([sixteen_class_mapping[x.item()] for x in labels])

                if inputs.size(0) == 0:
                    continue

            inputs, labels = inputs.to(device), labels.to(device)
            outputs, feature_loss = net(inputs)

            ensemble = sum(outputs[:-1])/len(outputs)
            ensemble.detach_()
            ensemble.requires_grad = False

            #   compute loss
            loss = torch.FloatTensor([0.]).to(device)

            #   for deepest classifier
            loss += criterion(outputs[0], labels)

            #   for soft & hard target
            teacher_output = outputs[0].detach()
            teacher_output.requires_grad = False

            for index in range(1, len(outputs)):
                loss += CrossEntropy(outputs[index], teacher_output) * args.lambda_KD * 9
                loss += criterion(outputs[index], labels) * (1 - args.lambda_KD)

            #   for faeture align loss
            if args.lambda_KD != 0:
                loss += feature_loss * 5e-7

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += float(labels.size(0))
            sum_loss += loss.item()

            _0, predicted0 = torch.max(outputs[0].data, 1)
            _1, predicted1 = torch.max(outputs[1].data, 1)
            _2, predicted2 = torch.max(outputs[2].data, 1)
            _3, predicted3 = torch.max(outputs[3].data, 1)
            _4, predicted4 = torch.max(ensemble.data, 1)

            correct0 += float(predicted0.eq(labels.data).cpu().sum())
            correct1 += float(predicted1.eq(labels.data).cpu().sum())
            correct2 += float(predicted2.eq(labels.data).cpu().sum())
            correct3 += float(predicted3.eq(labels.data).cpu().sum())
            correct4 += float(predicted4.eq(labels.data).cpu().sum())

            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: 4/4: %.2f%% 3/4: %.2f%% 2/4: %.2f%%  1/4: %.2f%%'
                    ' Ensemble: %.2f%%' % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
                                            100 * correct0 / total, 100 * correct1 / total,
                                            100 * correct2 / total, 100 * correct3 / total,
                                            100 * correct4 / total))

        print("Waiting Test!")
        with torch.no_grad():
            correct4, correct3, correct2, correct1, correct0 = 0, 0, 0, 0, 0
            predicted4, predicted3, predicted2, predicted1, predicted0 = 0, 0, 0, 0, 0
            correct = 0.0
            total = 0.0
            for data in testloader:
                net.eval()
                images, labels = data

                if args.dataset == "imagenet" and args.class_num == 16:
                    idx_keep = torch.tensor([x.item() in thousand_class_ids for x in labels])
                    labels = labels[idx_keep]
                    images = images[idx_keep]

                    labels = torch.tensor([sixteen_class_mapping[x.item()] for x in labels])

                    if images.size(0) == 0:
                        continue

                images, labels = images.to(device), labels.to(device)
                outputs, feature_loss = net(images)
                ensemble = sum(outputs) / len(outputs)
                _0, predicted0 = torch.max(outputs[0].data, 1)
                _1, predicted1 = torch.max(outputs[1].data, 1)
                _2, predicted2 = torch.max(outputs[2].data, 1)
                _3, predicted3 = torch.max(outputs[3].data, 1)
                _4, predicted4 = torch.max(ensemble.data, 1)

                correct0 += float(predicted0.eq(labels.data).cpu().sum())
                correct1 += float(predicted1.eq(labels.data).cpu().sum())
                correct2 += float(predicted2.eq(labels.data).cpu().sum())
                correct3 += float(predicted3.eq(labels.data).cpu().sum())
                correct4 += float(predicted4.eq(labels.data).cpu().sum())
                total += float(labels.size(0))

            msg = 'Epoch: %3d Acc: 4/4: %.4f%% 3/4: %.4f%% 2/4: %.4f%%  1/4: %.4f%% Ensemble: %.4f%%' % (
                epoch,
                100 * correct0 / total, 100 * correct1 / total,
                100 * correct2 / total, 100 * correct3 / total,
                100 * correct4 / total)
            print(msg)
            with open(os.path.join(root, "summary.log"), 'a') as f:
                f.write(msg+'\n')

            if correct0/total > best_acc:
                torch.save(net.state_dict(), os.path.join(root, "bestmodel.pth"))
                print("model saved")
                best_acc = correct0/total

    print("Training Finished, Total Epochs = %d" % args.epoch)
