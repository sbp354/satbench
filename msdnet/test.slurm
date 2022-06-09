#!/bin/bash
#SBATCH --job-name=mig-test
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --export=ALL
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_32gb
#SBATCH --mem=32G
#SBATCH -c 4

#SBATCH --array=1-10

# activate conda env
module load cuda-10.2
source /misc/vlgscratch5/PichenyGroup/as15003/miniconda3/etc/profile.d/conda.sh
conda activate scan

# see GPUs
nvidia-smi

# python -u test.py --mode color --color gray --sweep-step $SLURM_ARRAY_TASK_ID --tag gray-only-run2 --load_path model_files/neurips/run2/gray_only_2021.08.21_12.54.10/save_models/model_best.pth.tar
# small
# python -u test.py --mode noise --color gray --tag cifar10-anytime-noise --evalmode anytime --sweep-step $SLURM_ARRAY_TASK_ID --load_path /misc/vlgscratch5/PichenyGroup/as15003/anytime-prediction/msdnet/model_files/neurips/run1/cifar10_grayonly_111_ch8_2021.05.25_15.32.06/save_models/model_best.pth.tar

# medium
# python -u test.py --mode noise --color gray --tag cifar10-anytime-noise-medium --evalmode anytime --depth m --sweep-step $SLURM_ARRAY_TASK_ID --load_path /misc/vlgscratch5/PichenyGroup/as15003/anytime-prediction/msdnet/model_files/train/cifar10-grayonly-large_2021.10.20_19.08.06/save_models/model_best.pth.tar

# MIG
# python -u test.py --mode color --color gray --tag m-i-g --depth l --num_classes 16 \
# 	--data imagenet --sweep-step $SLURM_ARRAY_TASK_ID \
# 	--load_path /misc/vlgscratch5/PichenyGroup/as15003/anytime-prediction/msdnet/model_files/train/imagenet/m-i-g_2022.01.24_18.20.44/save_models/model_best.pth.tar

# # MIC
# python -u test.py --mode color --color color --tag m-i-c --depth l --num_classes 16 \
# 	--data imagenet --sweep-step $SLURM_ARRAY_TASK_ID \
# 	--load_path /misc/vlgscratch5/PichenyGroup/as15003/anytime-prediction/msdnet/model_files/train/imagenet/m-i-c_2022.01.24_18.20.44/save_models/model_best.pth.tar

# MIN
# python -u test.py --mode noise --color color --tag m-i-n --depth l --num_classes 16 \
# 	--data imagenet --sweep-step $SLURM_ARRAY_TASK_ID \
# 	--load_path /misc/vlgscratch5/PichenyGroup/as15003/anytime-prediction/msdnet/model_files/train/imagenet/m-i-n_2022.01.25_01.24.48/save_models/model_best.pth.tar


# TEST ON HUMAN
# ==============

# MIB
# python -u testonhuman.py --mode blur --color color --tag m-i-b --depth l --num_classes 16 --subsample True\
# 	--data imagenet --load_path /misc/vlgscratch5/PichenyGroup/as15003/anytime-prediction/msdnet/model_files/train/imagenet/m-i-b_2022.04.16_00.17.17/save_models/model_best.pth.tar

# MIN
# python -u testonhuman.py --mode noise --color gray --tag m-i-n --depth l --num_classes 16 --subsample True\
# 	--data imagenet --load_path /misc/vlgscratch5/PichenyGroup/as15003/anytime-prediction/msdnet/model_files/train/imagenet/m-i-n_2022.04.16_00.14.50/save_models/model_best.pth.tar

# MIC
# python -u testonhuman.py --mode color --color color --tag m-i-c --depth m --num_classes 16 --subsample True \
# 	--data imagenet --load_path /misc/vlgscratch5/PichenyGroup/as15003/anytime-prediction/msdnet/model_files/train/imagenet/m-i-c_2022.04.18_18.34.11/save_models/model_best.pth.tar

# MIG
# python -u testonhuman.py --mode color --color gray --tag m-i-g --depth m --num_classes 16 --subsample True \
# 	--data imagenet --load_path /misc/vlgscratch5/PichenyGroup/as15003/anytime-prediction/msdnet/model_files/train/imagenet/m-i-g_2022.04.18_18.33.52/save_models/model_best.pth.tar