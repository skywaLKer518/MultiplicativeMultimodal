# Learn to Combine Modalities in Multimodal Deep Learning
Code for submitted paper to KDD 2018



## CIFAR-100

### Installation
Customize paths first in `setup.sh` (data folder, model save folder, etc.).
```bash
git clone git://github.com/skywaLKer518/MultiplicativeMultimodal.git
cd MultiplicativeMultimodal/imagerecognition
# Change paths in setup.sh
# It also provides options to download CIFAR data.
./setup.sh
```

### run experiments
Vanilla resnet model
```bash
./run_resnet32_example.sh
```
Resnet-Mulmix multimodal model
```bash
./run_resnet32_mulmix_example.sh
```
Available values for `DATASET` are `cifar-100`.
Available values for `MODEL` are `resnet-32/110`.

## HIGGS


### Dataset download

Dataset needs to be downloaded in http://archive.ics.uci.edu/ml/datasets/HIGGS.

### run experiments 

