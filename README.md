# MXNet Implementation of WaveNet #
I am trying to reproduce the WaveNet result using MXNet. Here is the training code of generating without any condition, ~~but the training process can't convergence since the *mae* is always around 126. Hoping someone can raise your advices.~~
The model convergence in fact. The reason of "mae is always around 126" is the mxnet office mae evaluation metric does not match this net config. I have defined a new `EvalMetric`
[Training log](https://gist.github.com/shuokay/28de2c02c7857ab6ab7be2cd26b76915)
## How to Run ##
1. Install mxnet and fix the dilate bug according to https://github.com/dmlc/mxnet/issues/3479
2. Download VCTK cprpus and extract to the root folder
3. Start training by `python train.py`

## Implementation Note ##
* Padding zero on left side by *Concat* operator since the convolution op of mxnet can't pad only on one side.
