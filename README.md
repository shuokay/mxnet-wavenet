# MXNet Implementation of WaveNet #
I am trying to reproduce the WaveNet result using MXNet. Here is the training code of generating without any condition, but the training process can't convergence since the *mae* is always around 126. Hoping someone can raise your advices.

## How to Run ##
1. Install mxnet and fix the dilate bug according to https://github.com/dmlc/mxnet/issues/3479
2. Download VCTK cprpus and extract to the root folder
3. Start training by `python train.py`

## Implementation Note ##
* Padding zero on left side by *Concat* operator since the convolution op of mxnet can't pad only on one side.


