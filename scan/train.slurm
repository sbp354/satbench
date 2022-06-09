#!/bin/bash
#SBATCH --job-name=sin-train
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_32gb
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH -c 2

# activate conda env
module load cuda-10.2
source /misc/vlgscratch5/PichenyGroup/as15003/miniconda3/etc/profile.d/conda.sh
conda activate scan

# see GPUs
nvidia-smi

# train
# python -u train.py --depth=18 --class_num=10 --epoch=200 --lambda_KD=0.5
# python -u train_color.py --depth=34 --class_num=10 --epoch=200 --lambda_KD=0.5
# python -u train_color.py --depth=0 --class_num=10 --epoch=200 --lambda_KD=0.5
# python -u train_color.py --tag s-i-g --depth=18 --class_num=16 --epoch=200 --lambda_KD=0.5 --dataset imagenet --color gray
# python -u train_color.py --tag s-i-c --depth=18 --class_num=16 --epoch=200 --lambda_KD=0.5 --dataset imagenet
# python -u train_noise.py --tag s-i-n --depth=18 --class_num=16 --epoch=200 --lambda_KD=0.5 --dataset imagenet
# python -u train_blur.py --tag s-i-b --depth=18 --class_num=16 --epoch=200 --lambda_KD=0.5 --dataset imagenet
# python -u train.py --tag s-i-n --depth=18 --class_num=16 --epoch=200 --lambda_KD=0.5 --dataset imagenet --mode noise
python -u train.py --tag s-i-n --depth=18 --class_num=16 --epoch=200 --lambda_KD=0.5 --dataset imagenet --mode noise
