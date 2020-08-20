# NSNet
This in an implementation of NSNet [1] in PyTorch and PyTorch Lightning.
NSNet is a recurrent neural network for single channel speech enhancement.
This was implemented as part of my thesis for the Master in Electrical Engineering at Ghent University.

## Prerequisites
* torch 1.4
* pytorch_lightning 0.7.6
* torchaudio 1.4
* soundfile 0.10.3.post1

## How to train
A dataset containing both clean speech and corresponding noisy speech (i.e. clean speech with noise added) is required.

Running _train_nn.py_ starts the training.

The _train_dir_ variable should contain the path to a folder containing a _clean_ and a _noisy_ folder, containing the clean WAV files and the noisy WAV files respectively. The filename of a noisy WAV file must be the same as the corresponding clean WAV file, with optionally a suffix added delimited by _+_,
e.g. clean01.wav &rarr; clean01+noise.wav

The _val_dir_ follows the same convention, but this folder is used for validation.

## How to test
Running the _test_nn.py_ file results in the output (denoised) WAV files.

_testing_dir_ should point to a folder with the same structure as _train_dir_ and _val_dir_.

## Acknowledgements
[1] Y. Xia, S. Braun, C. K. A. Reddy, H. Dubey, R. Cutler, and I. Tashev, “Weighted Speech Distortion Losses for Neural-network-based Real-time Speech Enhancement,” arXiv:2001.10601 [cs, eess], Feb. 2020.