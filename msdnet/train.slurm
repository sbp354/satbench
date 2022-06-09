#!/bin/bash
#SBATCH --job-name=mic-train
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_32gb
#SBATCH --mem=64G
#SBATCH -c 4

# activate conda env
module load cuda-10.2
source /misc/vlgscratch5/PichenyGroup/as15003/miniconda3/etc/profile.d/conda.sh
conda activate scan

# see GPUs
nvidia-smi

# python -u train.py --tag small-contrast-noise --mode noise
# python -u train.py --tag stl10-grayonly-large --mode color --color gray
# python -u train.py --tag m-i-g --data imagenet --num_classes 16 --mode color --color gray --depth m
python -u train.py --tag m-i-c --data imagenet --num_classes 16 --mode color --color color --depth m
# python -u train.py --tag m-i-n --data imagenet --num_classes 16 --mode noise --color gray --depth m
# python -u train.py --tag m-i-b --data imagenet --num_classes 16 --mode blur --color color --depth m
