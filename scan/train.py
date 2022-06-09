import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sresnet
import small_resnet
import torch.nn.functional as F
import pandas as pd

import time
import datetime
import os
import json

from utils import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(100)
torch.cuda.manual_seed(100)

parser = argparse.ArgumentParser(description='SCAN Training')
parser.add_argument('--tag', type=str)
parser.add_argument('--depth', default=18, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--class_num', default=10, type=int)
parser.add_argument('--mode', default='gray', type=str, choices=['gray', 'color', 'noise', 'blur'])

parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--lambda_KD', default=0.5, type=float)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--weight-decay', default=5e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)

parser.add_argument('--dropout', default=False, type=bool)
parser.add_argument('--subset', default=False, type=bool)
parser.add_argument('--extraaug', default=False, type=bool)


args = parser.parse_args()
print(args)


def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs/3.0, dim=1)
    softmax_targets = F.softmax(targets/3.0, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

args.batch_size = 64
args.workers = 2

if args.mode == 'noise':
    args.noise_std = 0.04
    args.contrast = 0.2
elif args.mode == 'blur':
    if 'cifar' in args.dataset:
        args.blur_std = 1.0
        args.kernel_size = 7
    elif args.dataset == 'imagenet':
        args.blur_std = 1.0
        args.kernel_size = 49

start = time.time()
trainloader, testloader = get_dataloaders(args)
print("Time to get_dataloaders: {}".format(time.time() - start))

net = None
start = time.time()
if args.depth == 0:
    net = small_resnet.resnet_small(num_classes=args.class_num, align="CONV")
    print("using resnet small")
if args.depth == 18:
    net = sresnet.resnet18(num_classes=args.class_num, align="CONV", have_dropout=args.dropout)
    print("using resnet 18")
if args.depth == 34:
    net = sresnet.resnet34(num_classes=args.class_num, align="CONV")
    print("using resnet 34")
if args.depth == 50:
    net = sresnet.resnet50(num_classes=args.class_num, align="CONV")
    print("using resnet 50")
if args.depth == 101:
    net = sresnet.resnet101(num_classes=args.class_num, align="CONV")
    print("using resnet 101")
if args.depth == 152:
    net = sresnet.resnet152(num_classes=args.class_num, align="CONV")
    print("using resnet 152")

net.to(device)
print("Time to create network: {}".format(time.time() - start))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

if __name__ == "__main__":
    best_acc = 0
    # create experiment directory
    expt_id = "{}_{}_resnet{}_{}_cls{}_lr{}_ep{}_lkd{}_{}".format(
        args.tag,
        time.strftime('%Y.%m.%d_%H.%M.%S'),
        args.depth,
        args.dataset,
        args.class_num,
        args.lr,
        args.epoch,
        args.lambda_KD,
        args.mode
    )
    root = os.path.join('expt/train/{}'.format(args.dataset), expt_id)
    os.makedirs(root, exist_ok=True)

    if (args.dataset=='imagenet') & (args.class_num==16):
        sixteen_class_map_master = pd.read_csv('../datasets/imagenet_mapping.csv', header=0)
        thousand_class_ids = sixteen_class_map_master['thousand_class_id'].values
        sixteen_class_mapping = pd.Series(sixteen_class_map_master['sixteen_class_id'].values,
                                        index=sixteen_class_map_master['thousand_class_id']).to_dict()
    with open(os.path.join(root, "config.json"), 'w') as f:
        json.dump(vars(args), f)

    print("Start Training")
    all_train_accs = np.array([])
    all_test_accs = np.array([])
    for epoch in range(args.epoch):
        # print("EPOCH", str(epoch))
        correct4, correct3, correct2, correct1, correct0 = 0, 0, 0, 0, 0
        predicted4, predicted3, predicted2, predicted1, predicted0 = 0, 0, 0, 0, 0
        if epoch in [75, 130, 180]:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0

        # start = time.time()

        for i, data in enumerate(trainloader, 0):
            # print("BATCH", i)
            # print("Time to load batch: {}".format(time.time() - start))

            length = len(trainloader)
            inputs, labels = data

            start = time.time()
            if args.dataset == "imagenet" and args.class_num == 16:
                idx_keep = torch.tensor([x.item() in thousand_class_ids for x in labels])
                labels = labels[idx_keep]
                inputs = inputs[idx_keep]

                labels = torch.tensor([sixteen_class_mapping[x.item()] for x in labels])

                if inputs.size(0) == 0:
                    continue

            inputs, labels = inputs.to(device), labels.to(device)
            # print("Time to modify labels:", str(time.time() - start))

            # start = time.time()
            outputs, feature_loss = net(inputs)

            ensemble = sum(outputs[:-1])/len(outputs)
            ensemble.detach_()
            ensemble.requires_grad = False
            # print("Time for forward pass:", str(time.time() - start))

            # start = time.time()
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
            # print("Time for loss + backward pass:", str(time.time() - start))
            # print()


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

            print('[%s][epoch:%d, iter:%d] Loss: %.03f | Acc: 4/4: %.2f%% 3/4: %.2f%% 2/4: %.2f%%  1/4: %.2f%%'
                    ' Ensemble: %.2f%%' % (datetime.datetime.now(), epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
                                            100 * correct0 / total, 100 * correct1 / total,
                                            100 * correct2 / total, 100 * correct3 / total,
                                            100 * correct4 / total))
            # start = time.time()
        
        if len(all_train_accs) == 0:
            all_train_accs = np.array([100*correct0/total, 100*correct1/total, 100*correct2/total, 100*correct3/total, 100*correct4/total]).reshape(1,-1)
        else:
            all_train_accs = np.append(all_train_accs, np.array([100*correct0/total, 100*correct1/total, 100*correct2/total, 100*correct3/total, 100*correct4/total]).reshape(1,-1), axis=0)

        print("Waiting Test!")
        with torch.no_grad():
            correct4, correct3, correct2, correct1, correct0 = 0, 0, 0, 0, 0
            predicted4, predicted3, predicted2, predicted1, predicted0 = 0, 0, 0, 0, 0
            correct = 0.0
            total = 0.0
            for i, data in enumerate(testloader):

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

        if len(all_test_accs) == 0:
            all_test_accs = np.array([100*correct0/total, 100*correct1/total, 100*correct2/total, 100*correct3/total, 100*correct4/total]).reshape(1,-1)
        else:
            all_test_accs = np.append(all_test_accs, np.array([100*correct0/total, 100*correct1/total, 100*correct2/total, 100*correct3/total, 100*correct4/total]).reshape(1,-1), axis=0)


        # PLOT ALL ACCURACIES FOR TRAINING AND TESTING
        assert len(all_train_accs) == len(all_test_accs), "all_train_accs: {}, all_test_accs: {}".format(all_train_accs, all_test_accs)
        fig = plt.figure()
        for i, m in zip([0,4], ['o', 's']):
            plt.plot(range(1, epoch+2), all_train_accs[:,i], marker=m, color='r', linestyle='-', label='Train: exit {}'.format(i))
            plt.plot(range(1, epoch+2), all_test_accs[:,i], marker=m, color='b', linestyle='-', label='Test: exit {}'.format(i))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        fig.savefig(os.path.join(root, 'train_test_progress.png'), dpi=fig.dpi) 
        
    print("Training Finished, Total Epochs = %d" % args.epoch)
