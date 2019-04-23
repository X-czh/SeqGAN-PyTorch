# SeqGAN-PyTorch
An implementation of SeqGAN (Paper: [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/pdf/1609.05473.pdf)) in PyTorch. The code performs the experiment on synthetic data as described in the paper.

## Usage
```
$ python main.py
```
Please refer to ```main.py``` for supported arguments. You can also change model parameters there.

## Dependency
* PyTorch 0.4.0+ (1.0 ready)
* Python 3.5+
* CUDA 8.0+ & cuDNN (For GPU)
* numpy

## Hacks and Observations
- Using Adam for Generator and SGD for Discriminator
- Discriminator should neither be trained too powerful (fail to provide useful feedback) nor too ill-performed (randomly guessing, unable to guide generation)
- The GAN phase may not always lead to massive drops in NLL (sometimes very minimal or even increases NLL)

## Sample Learning Curve
Learning curve of generator obtained after MLE training for 120 steps (1 epoch per round) followed by adversarial training for 150 rounds (1 epoch per round):

![alt tag](https://raw.githubusercontent.com/X-czh/SeqGAN-PyTorch/master/gen_loss.png)

Learning curve of discriminator obtained after MLE training for 50 steps (3 epochs per step) followed by adversarial training for 150 rounds (9 epoch per round):

![alt tag](https://raw.githubusercontent.com/X-czh/SeqGAN-PyTorch/master/dis_loss.png)


## Acknowledgement
This code is based on Zhao Zijian's [SeqGAN-PyTorch](https://github.com/ZiJianZhao/SeqGAN-PyTorch), Surag Nair's [SeqGAN](https://github.com/suragnair/seqGAN) and Lantao Yu's original [implementation](https://github.com/LantaoYu/SeqGAN) in Tensorflow. Many thanks to [Zhao Zijian](https://github.com/ZiJianZhao), [Surag Nair](https://github.com/suragnair) and [Lantao Yu](https://github.com/LantaoYu)!
