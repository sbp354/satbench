#!/bin/bash
#SBATCH --job-name=sin-test
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --export=ALL
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1080ti:1
#SBATCH --constraint=gpu_12gb
#SBATCH --mem=32G
#SBATCH -c 1

# activate conda env
module load cuda-10.2
source /misc/vlgscratch5/PichenyGroup/as15003/miniconda3/etc/profile.d/conda.sh
conda activate scan

# see GPUs
nvidia-smi

# python -u testonhuman.py --tag s-i-n --depth 18 --dataset imagenet --class_num 16 --mode noise --load-path /misc/vlgscratch5/PichenyGroup/as15003/anytime-prediction/scan/expt/train/imagenet/s-i-n_2022.04.12_14.40.55_resnet18_imagenet_cls16_lr0.1_ep200_lkd0.5_noise/bestmodel.pth
python -u testonhuman.py --tag s-i-b --depth 18 --dataset imagenet --class_num 16 --mode blur --load-path /scratch/as15003/dynamic-nn/scan/expt/train/imagenet/s-i-b_2022.04.16_00.31.52_resnet18_imagenet_cls16_lr0.1_ep200_lkd0.5_blur/bestmodel.pth
