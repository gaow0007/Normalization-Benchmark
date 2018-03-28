# Normalization-Benchmark
I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Pros
Pros:
- Built-in data loading and augmentation, very nice!
- Training is fast, maybe even a little bit faster.
- Very memory efficient!
- Aggreate many Normalization Strategy

## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.01  |
| [VGG16-BN](https://arxiv.org/abs/1502.03167)          | 94.12  |
| [VGG16-WN](https://arxiv.org/abs/1602.07868)          | -      |
| [VGG16-LN](https://arxiv.org/abs/1607.06450)          | -      |
| [VGG16-IN](https://arxiv.org/abs/1607.08022)          | -      |
| [VGG16-GN](https://arxiv.org/abs/1803.08494)          | -      |


| Model             | Acc.        |
| ----------------- | ----------- |
| [ResNet18](https://arxiv.org/abs/1512.03385)             | -      |
| [ResNet18-BN](https://arxiv.org/abs/1502.03167)          | -      |
| [ResNet18-WN](https://arxiv.org/abs/1602.07868)          | -      |
| [ResNet18-LN](https://arxiv.org/abs/1607.06450)          | -      |
| [ResNet18-IN](https://arxiv.org/abs/1607.08022)          | -      |
| [ResNet18-GN](https://arxiv.org/abs/1803.08494)          | -      |

## Learning rate adjustment
I manually change the `lr` during training:
- `0.01` for epoch `[0,150)`
- `0.001` for epoch `[150,250)`
- `0.0001` for epoch `[250,350)`

Resume the training with `python main.py --resume --lr=0.01`
