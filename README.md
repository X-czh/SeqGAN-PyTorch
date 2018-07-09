# SeqGAN-PyTorch
An implementation of SeqGAN (Paper: [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/pdf/1609.05473.pdf)) in PyTorch.

Note that this code requires PyTorch version 0.4.0 or above.

## Usage
```
$ python main.py
```
Please refer to ```main.py``` for supported arguemnts. You can also change model parameters there.  

## Dependency
* PyTorch 0.4.0+
* Python 3.5+
* CUDA 8.0+ (For GPU)
* numpy

## Acknowledgement
This code is based on ZiJianZhao's [SeqGAN-PyTorch](https://github.com/ZiJianZhao/SeqGAN-PyTorch) and LantaoYu's original [implementation](https://github.com/LantaoYu/SeqGAN) in Tensorflow. Many thanks to [ZiJianZhao](https://github.com/ZiJianZhao) and [LantaoYu](https://github.com/LantaoYu)!