# Outlier Detection for Time Series with Recurrent Autoencoder Ensembles (Torch Implementation)
This repository contains a PyTorch implementation of the paper "Outlier Detection for Time Series with Recurrent Autoencoder Ensembles" by Tung Kieu, Bin Yang, Chenjuan Guo, and Christian S. Jensen (IJCAI 2019). We referred to the code on https://github.com/tungk/OED. The goal of this project is to detect outliers in time series data using ensemble models based on recurrent autoencoders with several additional implementations: Dynamic Thresholding, Residual Connection, etc.

# Requirements
- Python 3.x
- Numpy
- Torch

Install the required packages by running:
```bash
pip install -r requirements.txt
```

# Dataset
This implementation uses the following publicly available dataset:
- NAB (Numenta Anomaly Benchmark): For anomaly detection in time series data.

To use the dataset, please refer to the original paper or download them directly from the provided sources.

E.g. Download w/ Git Commend
```bash
git clone https://github.com/numenta/NAB.git
```

# Model
This implementation includes improved models with several additional features to improve outlier detection performance.

## Base Model
- SLSTMAutoEncoder: Ensemble of Sparcely Connection LSTM and AutoEncoder
- SGRUAutoEncoder: Ensemble of Sparcely Connection GRU and AutoEncoder (In development)

## Advanced Model
- (TBD)

# Installation
```bash
pip install git+https://github.com/abcd-EGH/srnn-ae
```

# Citation
If you find this code or project useful in your research, please cite the original paper:
```scss
@inproceedings{tungbcc19,
  title={Outlier Detection for Time Series with Recurrent Autoencoder Ensembles},
  author={Kieu, Tung and Yang, Bin and Guo, Chenjuan and S. Jensen, Christian},
  booktitle={International Joint Conference on Artificial Intelligence (IJCAI '19)},
  year={2019}
}
```