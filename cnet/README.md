# Improving Anytime Prediction with Parallel Cascaded Networks and a Temporal-Difference Loss

This repository is the official implementation of [Improving Anytime Prediction with Parallel Cascaded Networks and a Temporal-Difference Loss](https://arxiv.org/abs/2102.09808) accepted to the 35th Conference on Neural Information Processing Systems (NeurIPS), 2021.

## Requirements

To install requirements:

```
pip install -r requirements.txt
```

## Training
This reposiotry supports training on a 16 class subset of the ImageNet2012 dataset initially developed here [https://github.com/rgeirhos/generalisation-humans-DNNs] under 4 different image perturbation regimes:
* color (default - no perturbations applied)
* grayscale 
* gaussian noise applied to grayscale images
* gaussian blur filter applied to color images

Use `train.sh` as a template to set your own hyperparameters, which launches `train.py`.
Below are shell commands you can run to replicate training for the above perturbations where:
* DATASET_ROOT = parent directory for imagenet data. Should have at least subdirectory train
* EXPERIMENT_ROOT = parent output directory for storing model outputs and results (subdirectories for different types of experiments are generated automatically during run time)
* SPLIT_IDX_ROOT = directory for storing train/val idx splits for reproducibility and consistency
* HUMAN_DATA_DIR = directory where you have downloaded the test data provided by this repository

Note that the arguments passed in the following shell commands specify key directories, the type of CNet model to run (parallel or serial), temporal difference hyperparameter (lambda) and the image perturbation regime. The current hyperparameters in train.sh are configured to reproduce results in this paper but can be adjusted as needed or desired. Additionally, we include examples for reproducing results with the parallel and serial CNet architectures.

### Color
Parallel Architecture
```
bash train.sh DATASET_ROOT EXPERIMENT_ROOT SPLIT_IDX_ROOT HUMAN_DATA_DIR parallel 1.0 false false false
```
Serial Architecture
```
bash train.sh DATASET_ROOT EXPERIMENT_ROOT SPLIT_IDX_ROOT HUMAN_DATA_DIR serial 0.0 false false false
```

### Grayscale
Parallel Architecture
```
bash train.sh DATASET_ROOT EXPERIMENT_ROOT SPLIT_IDX_ROOT HUMAN_DATA_DIR parallel 1.0 true false false
```
Serial Architecture
```
bash train.sh DATASET_ROOT EXPERIMENT_ROOT SPLIT_IDX_ROOT HUMAN_DATA_DIR serial 0.0 true false false
```
### Gaussian Noise on Grayscale Images
Parallel Architecture
```
bash train.sh DATASET_ROOT EXPERIMENT_ROOT SPLIT_IDX_ROOT HUMAN_DATA_DIR parallel 1.0 true true false
```
Serial Architecture
```
bash train.sh DATASET_ROOT EXPERIMENT_ROOT SPLIT_IDX_ROOT HUMAN_DATA_DIR serial 0.0 true true false
```
### Gaussian Blur on Color Images
Parallel Architecture
```
bash train.sh DATASET_ROOT EXPERIMENT_ROOT SPLIT_IDX_ROOT HUMAN_DATA_DIR parallel 1.0 true true false
```
Serial Architecture
```
bash train.sh DATASET_ROOT EXPERIMENT_ROOT SPLIT_IDX_ROOT HUMAN_DATA_DIR serial 0.0 true true false
```

## Evaluation
Use `eval.sh` to load and evaluate the model stored in `EXPERIMENT_NAME`. This script will evaluate the performance of the model and, if `--keep_logits` is specified, generate and store the logits for all examples in the specified dataset. The logits are useful for downstream tasks, such as training metacognition models.

Similar to `train.sh`, specify `DATASET_ROOT`, `EXPERIMENT_ROOT`, and `SPLIT_IDXS_ROOT`.

To evaluate the model, run `eval.sh`

Analyze results with `Analysis.ipynb`

## Results

<p>
    <img src="figures/speed_acc.png" />
</p>

The key observations from the Figure above, which shows speed-accuracy trade offs for the six models on three data sets, are as follows. First, our canonical cascaded model, CascadedTD, obtains better anytime prediction than SerialTD-MultiHead (i.e., the architecture of SDN). CascadedTD also achieves higher asymptotic accuracy; its accuracy matches that of CascadedCE, a ResNet trained in the standard manner. Thus, cascaded models can exploit parallelism to obtain computational benefits in speeded perception without costs in accuracy.

Second, while MultiHead is superior to SingleHead for serial models, the reverse is true for cascaded models. This finding is consistent with the cascaded architecture's perspective on anytime prediction as unrolled iterative estimation, rather than, as cast in SDN, as distinct read out heads from different layers of the network. Third, models trained with TD outperform models trained with standard cross-entropy loss. Training for speeded responses reorganizes knowledge in the network so that earlier layers are more effective in classifying instances.


## Contact
If you have any questions, feel free to contact us through email (michael.iuzzolino@colorado.edu) or Github issues. Enjoy!
