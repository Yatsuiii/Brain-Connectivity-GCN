# GraphTemporalNet-PyTorch

[![GitHub stars](https://img.shields.io/github/stars/martinwhl/GraphTemporalNet-PyTorch?label=stars&maxAge=2592000)](https://gitHub.com/martinwhl/GraphTemporalNet-PyTorch/stargazers/) [![issues](https://img.shields.io/github/issues/martinwhl/GraphTemporalNet-PyTorch)](https://github.com/martinwhl/GraphTemporalNet-PyTorch/issues) [![License](https://img.shields.io/github/license/martinwhl/GraphTemporalNet-PyTorch)](./LICENSE) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/martinwhl/GraphTemporalNet-PyTorch/graphs/commit-activity) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) ![Codefactor](https://www.codefactor.io/repository/github/martinwhl/GraphTemporalNet-PyTorch/badge)

This is a PyTorch implementation of GraphTemporalNet in the following paper: [GraphTemporalNet: A Temporal Graph Convolutional Network for Traffic Prediction](https://arxiv.org/abs/1811.05320).

A stable version of this repository can be found at [the official repository](https://github.com/lehaifeng/GraphTemporalNet/tree/master/GraphTemporalNet/GraphTemporalNet-PyTorch).

Notice that [the original implementation](https://github.com/lehaifeng/GraphTemporalNet/tree/master/GraphTemporalNet/GraphTemporalNet-TensorFlow) is in TensorFlow, which performs a tiny bit better than this implementation for now.

## Requirements

* numpy
* matplotlib
* pandas
* torch
* pytorch-lightning>=1.3.0
* torchmetrics>=0.3.0
* python-dotenv

## Model Training

```bash
# GCN
python main.py --model_name GCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 0 --batch_size 64 --hidden_dim 100 --settings supervised --gpus 1
# GRU
python main.py --model_name GRU --max_epochs 3000 --learning_rate 0.001 --weight_decay 1.5e-3 --batch_size 64 --hidden_dim 100 --settings supervised --gpus 1
# GraphTemporalNet
python main.py --model_name GraphTemporalNet --max_epochs 3000 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --hidden_dim 64 --loss mse_with_regularizer --settings supervised --gpus 1
```

You can also adjust the `--data`, `--seq_len` and `--pre_len` parameters.

Run `tensorboard --logdir lightning_logs/version_0` to monitor the training progress and view the prediction results.
