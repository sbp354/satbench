#!/bin/bash

DATASET_ROOT="/scratch/work/public/imagenet"  # /scratch/work/public/imagenet, ../cascade_output/datasets
EXPERIMENT_ROOT="../cascade_output/experiments"  # Specify experiment root
SPLIT_IDXS_ROOT="../cascade_output/split_idx"  # Specify root of dataset split_idxs
TEST_DATASET_ROOT='/scratch/sbp354/SAT_human_data'

MODEL="resnet18"  # resnet18, resnet34, resnet50, densenet_cifar
DATASET_NAME="ImageNet2012_16classes_rebalanced"  # CIFAR10, CIFAR100, TinyImageNet, ImageNet2012
EXPERIMENT_NAME="${MODEL}_${DATASET_NAME}"

# Model params
TRAIN_MODE="cascaded"  # baseline, cascaded
CASCADED_SCHEME="parallel"  # serial, parallel

MULTIPLE_FCS=false

LAMBDA_VALS=1.0 # To sweep, set as list. E.g., LAMBDA_VALS=(0.0 0.25 0.5 0.83 1.0)
TAU_WEIGHTED_LOSS=false
PRETRAINED_WEIGHTS=false
USE_ALL_ICS=false

#Image perturbations
GRAYSCALE=true
GAUSS_NOISE=true
GAUSS_NOISE_STD=0.0
BLUR=false
BLUR_STD=0.0
BLUR_RANGE = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Optimizer / LR Scheduling
LR_MILESTONES=(30 60 90)
LR=0.01
WEIGHT_DECAY=0.0005
MOMENTUM=0.9
NESTEROV=true

# General / Dataset / Train params
DEVICE=0
USE_CPU=false
RANDOM_SEEDS=(42)  # To sweep, set as list. E.g., RANDOM_SEEDS=(42 542 1042)
EPOCHS=120
BATCH_SIZE=128  # 128
NUM_WORKERS=20
DEBUG=false

for RANDOM_SEED in "${RANDOM_SEEDS[@]}"
do
    for LAMBDA_VAL in "${LAMBDA_VALS[@]}"
    do
      cmd=( python ../CascadedNets/train.py )   # create array with one element
      cmd+=( --device $DEVICE )
      cmd+=( --random_seed $RANDOM_SEED )
      cmd+=( --dataset_root $DATASET_ROOT )
      cmd+=( --test_dataset_root $TEST_DATASET_ROOT )
      cmd+=( --dataset_name $DATASET_NAME )
      cmd+=( --split_idxs_root $SPLIT_IDXS_ROOT )
      cmd+=( --experiment_root $EXPERIMENT_ROOT )
      cmd+=( --experiment_name $EXPERIMENT_NAME )
      cmd+=( --n_epochs $EPOCHS )
      cmd+=( --model_key $MODEL )
      cmd+=( --cascaded_scheme $CASCADED_SCHEME )
      cmd+=( --lambda_val $LAMBDA_VAL )
      cmd+=( --train_mode $TRAIN_MODE )
      cmd+=( --batch_size $BATCH_SIZE )
      cmd+=( --num_workers $NUM_WORKERS )
      cmd+=( --learning_rate $LR )
      cmd+=( --lr_milestones "${LR_MILESTONES[@]}" )
      cmd+=( --momentum $MOMENTUM )
      cmd+=( --weight_decay $WEIGHT_DECAY )
      cmd+=( --gauss_noise_std $GAUSS_NOISE_STD )
      cmd+=( --blur_std $BLUR_STD )
      cmd+=( --blur_range $BLUR_RANGE )
      ${NESTEROV} && cmd+=( --nesterov )
      ${TAU_WEIGHTED_LOSS} && cmd+=( --tau_weighted_loss )
      ${PRETRAINED_WEIGHTS} && cmd+=( --use_pretrained_weights )
      ${MULTIPLE_FCS} && cmd+=( --multiple_fcs )
      ${USE_ALL_ICS} && cmd+=( --use_all_ICs )
      ${GRAYSCALE} && cmd+=( --grayscale ) #pg_grayscale
      ${GAUSS_NOISE} && cmd+=( --gauss_noise )
      ${BLUR} && cmd+=( --blur )
      ${USE_CPU} && cmd+=( --use_cpu )
      ${DEBUG} && cmd+=( --debug ) && echo "DEBUG MODE ENABLED"

      # Run command
      "${cmd[@]}"
    done
done