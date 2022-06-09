import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sresnet
import small_resnet
import torch.nn.functional as F
import numpy as np
import pandas as pd

import time
import os
import json

from utils import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(100)
torch.cuda.manual_seed(100)

parser = argparse.ArgumentParser(description='SCAN Testing')
parser.add_argument('--tag', type=str)
parser.add_argument('--load-path', type=str)
parser.add_argument('--depth', default=18, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--class_num', default=10, type=int)
parser.add_argument('--mode', default='gray', type=str, choices=['gray', 'color', 'noise', 'blur'])
parser.add_argument('--sweep-step', default=1, type=int)
parser.add_argument('--subsample', default=False, type=bool)

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
args.lr = 0.1
args.workers = 4

if args.mode == 'noise':
    noise_stds = list(np.arange(0.04, 0.13, 0.02))
    args.noise_std = noise_stds[args.sweep_step - 1]
    args.contrast = 0.1
elif args.mode == 'blur':
    if 'cifar' in args.dataset:
        blur_stds = list(np.arange(0.0, 3.0, 0.25))
        args.blur_std = blur_stds[args.sweep_step - 1]
        args.kernel_size = 7
    elif args.dataset == 'imagenet':
        blur_stds = [1.0, 3.0, 6.0, 9.0]
        args.blur_std = blur_stds[args.sweep_step - 1]
        args.kernel_size = 49


net = None
if args.depth == 0:
    net = small_resnet.resnet_small(num_classes=args.class_num, align="CONV")
    net_ops = [76758016., 80550912., 96201728., 103437312., 173136896.]
    print("using resnet small")
if args.depth == 18:
    net = sresnet.resnet18(num_classes=args.class_num, align="CONV")
    net_ops = [190856192., 308321280., 437431296., 558019584., 627719168.]
    print("using resnet 18")
if args.depth == 34:
    net = sresnet.resnet34(num_classes=args.class_num, align="CONV")
    net_ops = [266943488., 535993344., 967683072., 1163842560., 1233542144.]
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
net.load_state_dict(torch.load(args.load_path))

if __name__ == "__main__":
    best_acc = 0

    # create experiment directory
    expt_id = "{}_{}_resnet{}_{}_cls{}_{}_ss{}".format(
        args.tag,
        time.strftime('%Y.%m.%d_%H.%M.%S'),
        args.depth,
        args.dataset,
        args.class_num,
        args.mode,
        args.sweep_step
    )
    root = os.path.join('expt/testonhuman/{}'.format(args.dataset), expt_id)
    os.makedirs(root, exist_ok=True)

    with open(os.path.join(root, "config.json"), 'w') as f:
        json.dump(vars(args), f)

    print("Waiting Test!")
    flops = []
    accs = []
    results = {}
    for c in range(0,5):
        caught = [0, 0, 0, 0, 0]
        t = str(c)
        args.dirs = [t]
        testloader = get_dataloaders(args, train=False, human=True)

        with torch.no_grad():
            correct4, correct3, correct2, correct1, correct0 = 0, 0, 0, 0, 0
            predicted4, predicted3, predicted2, predicted1, predicted0 = 0, 0, 0, 0, 0
            
            if args.mode == 'color' or args.mode == 'gray':
                right = dict([(k,[0,0]) for k in ['c', 'g']])
            elif args.mode == 'noise':
                right = dict([(k,[0,0]) for k in ['0', '0.04', '0.16']])
            elif args.mode == 'blur':
                right = dict([(k,[0,0]) for k in ['0', '3', '9']])

            for data in testloader:
                net.eval()
                images, labels, modes = data

                images, labels = images.to(device), labels.to(device)
                outputs, feature_loss = net(images)
                ensemble = sum(outputs) / len(outputs)
                outputs.reverse()

                for index in range(len(outputs)):
                    outputs[index] = F.softmax(outputs[index])

                for index in range(images.size(0)):
                    ok = False

                    if c < 4:
                        logits = outputs[c][index]
                        caught[c] += 1
                        predict = torch.argmax(logits)
                        if predict.cpu().numpy().item() == labels[index]:
                            right[modes[index]][0] += 1

                        ok = True

                    if not ok: # i.e. c == 5
                        caught[-1] += 1
                        #   print(index, "ensemble")
                        logits = ensemble[index]
                        predict = torch.argmax(logits)
                        if predict.cpu().numpy().item() == labels[index]:
                            right[modes[index]][0] += 1

                    right[modes[index]][1] += 1
                # total += float(labels.size(0))

            # compute accuracy
            exit_accuracies = dict([(k, v[0]*100/v[1]) for k, v in right.items()])
            
            for k, v in exit_accuracies.items():
                if k not in results:
                    results[k] = []

                results[k].append(exit_accuracies[k])

            # FLOPs (only for resnet18)

            # print("caughts:", caught)
            # acceleration_ratio = 1/((0.32 * caught[0] + 0.53* caught[1] + 0.76*caught[2] + 1.0 * caught[3] + 1.07 * caught[4])/total)
            # accr_msg = "Acceleration ratio: %.4f " % acceleration_ratio

            # ops = sum([i*j for i,j in zip(net_ops, caught)])/total
            # ops_msg = 'FLOPs: %.4f ' % ops

            # print(ops_msg)
            # print(accr_msg)

            # tst_msg = 'Test Set Accuracy:  %.4f%% ' % (100 * right / total) 
            # print(tst_msg)
            # print("-----------------")
            # accs.append(100 * right / total)
            # flops.append(ops)

            # with open(os.path.join(root, "summary.log"), 'a') as f:
            #     f.write(tst_msg + accr_msg + ops_msg + '\n')

    # print('flops = {}'.format(flops))
    # print('accs = {}'.format(accs))
    print(results)
    results['flops'] = net_ops
    
    with open(os.path.join(root, 'results.json'), 'w') as fp:
        json.dump(results, fp)

