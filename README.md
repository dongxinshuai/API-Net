# Background
This is the code for reproducing the ECCV-2020 submission "API-Net: Robust Generative Classifier via a Single Discriminator" 

# Requirements
- torch
- torchvision
- argparse
- tensorboardX
- numpy

# Run
The results can be reproduced with the default hyperparameters with the following command:
source train_cifar.sh

The code can also extends to other architecture by leveraging class AdvModel.
