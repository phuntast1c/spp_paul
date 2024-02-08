# Temporal Convolutional Network-Based Speech Presence Probability Estimation
[![Paper](http://img.shields.io/badge/paper-TASLP)](https://ieeexplore.ieee.org/abstract/document/10224310)

## Description
This repository contains the code to train a temporal convolutional network (TCN)-based speech presence probability (SPP) estimator similar to the one used in [^1].
The difference is that the version in this repository uses not only magnitude-based features, but also phase-based features, which improved SPP estimation accuracy in internal experiments.
During training, the SPP proposed in [^2] (Eq. (18); with the noise power spectral density smoothed as in [^2]) is used as the target, and the mean squared error is used as the loss function.

## How to run
### Installation
First, install dependencies:

```bash
# clone project
git clone https://github.com/phuntast1c/spp_paul

# install and activate conda environment
cd spp_paul
conda env create -f environment.yml
conda activate spp_tcn
```

### Usage
This repository contains two pre-trained models, one for fs = 8 kHz and one for fs = 16 kHz.
These models were trained on the reverberant ICASSP 2021 Deep Noise Suppression (DNS) challenge dataset, including both simulated and measured room impulse responses.
```inference.ipynb``` demonstrates how to use these models for the prediction of the SPP given a noisy single-channel input.

When different settings or a different dataset are desired, the SPP estimator can be trained using the PyTorch Lightning (PL) command-line interface, preferably with an equipped NVIDIA GPU. The available model arguments can be printed using:

```bash
python cli.py fit --model.help SPPEstimator
```

For example, the provided models were trained using:

```bash
python cli.py fit --trainer=configs/trainer/trainer.yaml --model=configs/model/240206_spp_tcn.yaml --data=configs/data/240206_dns2_reverberant.yaml --data.fs=8000 &> log.fs_8000 &
python cli.py fit --trainer=configs/trainer/trainer.yaml --model=configs/model/240206_spp_tcn.yaml --data=configs/data/240206_dns2_reverberant.yaml --data.fs=16000 &> log.fs_16000 &
```

To handle data from the DNS challenge [^3], this repository includes a `PL LightningDataModule` implementation. For instructions on how to obtain the data, please refer to [the official repository](https://github.com/microsoft/DNS-Challenge), and adjust the paths accordingly. The configuration file used to generate the training dataset can be found in `spp/datasets/noisyspeech_synthesizer.cfg`.

## References
[^1] M. Tammen and S. Doclo, "Parameter Estimation Procedures for Deep Multi-Frame MVDR Filtering for Single-Microphone Speech Enhancement," in *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 31, pp. 3237-3248, August 2023.

[^2] T. Gerkmann and R. Hendriks, “Unbiased MMSE-Based Noise Power Estimation With Low Complexity and Low Tracking Delay,” in *IEEE Transactions on Audio, Speech, and Language Processing*, vol. 20, no. 4, pp. 1383–1393, May 2012.

[^3] C. K. A. Reddy et al., “ICASSP 2021 Deep Noise Suppression Challenge,” Oct. 2020. *Available*: http://arxiv.org/abs/2009.06122.
