import mxnet as mx
import numpy as np
import logging
import librosa
import os
import fnmatch
import multiprocessing
import random
from scipy.ndimage.interpolation import shift

def causal_layer(data=None, name="causal"):
    assert isinstance(data, mx.symbol.Symbol)
    zero = mx.symbol.Variable(name=name+"-zero")
    concat = mx.symbol.Concat(*[data, zero], dim=3, name=name+"-concat")
    causal = mx.symbol.Convolution(data=concat, kernel=(1, 2), stride=(1, 1), num_filter=32, name=name)
    return causal

def residual_block(data=None, kernel=(1, 2), dilate=None, num_filter=32, name=None, stride=(1, 1), output_channel=None):
    assert name is not None
    assert dilate is not None
    assert output_channel is not None
    assert isinstance(data, mx.symbol.Symbol)
    zero = mx.symbol.Variable(name=name+"-zero")
    concat = mx.symbol.Concat(*[data, zero], dim=3, name=name+"-concat")
    conv_filter = mx.symbol.Convolution(data=concat, kernel=kernel, stride=stride, dilate=dilate, num_filter=num_filter, name=name+"conv-filter")
    conv_gate = mx.symbol.Convolution(data=concat, kernel=kernel, stride=stride, dilate=dilate, num_filter=num_filter, name=name+"conv-gate")
    output_filter = mx.symbol.Activation(data=conv_filter, act_type="tanh", name=name+"act_filter")
    output_gate = mx.symbol.Activation(data=conv_gate, act_type="sigmoid", name=name+"act_gate")
    output = output_filter * output_gate
    out_dense = mx.symbol.Convolution(data=output, kernel=(1, 1), num_filter=output_channel, name=name+"out_dense")
    out_skip = mx.symbol.Convolution(data=output, kernel=(1, 1), num_filter=output_channel, name=name+"out_skip")
    return out_skip+data, out_dense

class DataBatch(mx.io.DataBatch):
    def __init__(self, data, label):
        self.data = data
        self.label = label

class DataIter(mx.io.DataIter):
    def __init__(self, batch_size, length, names, shape):
        self.provide_data = [(k, v) for k, v in shape.iteritems()]
        self.provide_label = [("softmax_label", (batch_size, length))]
        self.cur_batch = 0
        self.num_batch = len(names)/batch_size
        self.batch_size = batch_size
        self.length = length
        self.names = names
        self.q = multiprocessing.Queue(maxsize=4)
        self.pws = [multiprocessing.Process(target=self.get_batch) for i in xrange(4)]
        for pw in self.pws:
            pw.daemon = True
            pw.start()

    def reset(self):
        self.cur_batch = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def get_batch(self):
        while True:
            data_all = np.empty(shape=(self.batch_size, 1, 1, self.length))
            label_all = np.empty(shape=(self.batch_size, self.length))
            mx_data = []
            mx_label = []
            idx = 0
            while idx < self.batch_size:
                name = random.choice(self.names)
                audio, _ = librosa.load(name, sr=16000, mono=True)
                if audio.shape[0] < self.length:
                    continue
                audio = audio[:self.length]
                magnitude = 1.0*np.log(1+255*np.abs(audio))/np.log(1.0+255)
                signal = np.sign(audio) * magnitude
                audio = ((signal+1)/2.0*255+0.5).astype(np.int16)
                label = shift(audio, -1, cval=0)
                audio = audio.reshape(1, 1, self.length)
                data_all[idx, :, :, :] = audio
                label_all[idx, :] = label
                idx += 1
            for k, v in shape.iteritems():
                if "input" in k:
                    data = mx.nd.array(np.array(data_all))
                else:
                    data = mx.nd.array(np.zeros(shape=v))
                mx_data.append(data)
            label = mx.nd.array(np.array(label_all))
            mx_label.append(label)
            self.q.put(obj=DataBatch(mx_data, mx_label), block=True, timeout=None)

    def next(self):
        if self.q.empty():
            logging.debug("waiting for data......")
        if self.cur_batch < self.num_batch:
            self.cur_batch += 1
            return self.q.get(block=True, timeout=None)
        else:
            raise StopIteration

class MYMAE(mx.metric.EvalMetric):
    """Calculate Mean Absolute Error loss"""

    def __init__(self):
        super(MYMAE, self).__init__('mymae')

    def update(self, labels, preds):

        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()

            if len(label.shape) == 1:
                label = label.reshape(label.shape[0], 1)
            self.sum_metric += np.abs(label - np.argmax(pred, axis=1).reshape(label.shape)).mean()
            self.num_inst += 1 # numpy.prod(label.shape)

if __name__ == "__main__":
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)
    dilate = [2**i for i in range(1, 10)]
    shape = {}
    params = {'length': 2**15, 'batch_size': 1}
    batch_size = params['batch_size']
    length = params['length']
    data = mx.symbol.Variable(name="input")
    net = causal_layer(data=data, name="causal")
    shape = {
        "input": (batch_size, 1, 1, length),
        "causal-zero": (batch_size, 1, 1, 1)
    }
    residual = []
    outs = []
    for d in dilate:
        name = "residual-"+str(d)
        output_channel = 32
        net, out = residual_block(data=net, kernel=(1, 2), dilate=(1, d), num_filter=32, stride=(1, 1), output_channel=output_channel, name=name)
        residual.append(net)
        outs.append(out)
        shape[name+"-zero"] = (batch_size, output_channel, 1, d)
    # net = outs[0]+outs[1]+outs[2]+outs[3]+outs[4]+outs[5]+outs[6]+outs[7]
    # net=sum(outs)
    net = outs[0]
    for out in outs[1:]:
        net += out
    net = mx.symbol.Activation(data=net, act_type="relu", name="sum-activation")
    net = mx.symbol.Convolution(data=net, kernel=(1, 1), num_filter=128, name="post-conv1")
    net = mx.symbol.Activation(data=net, act_type="relu", name="post-activation1")
    net = mx.symbol.Convolution(data=net, kernel=(1, 1), num_filter=256, name="post-conv2")
    net = mx.symbol.SoftmaxOutput(data=net, name="softmax", multi_output=True)
    # mx.viz.plot_network(symbol=net, shape=shape, node_attrs={"fixedsize": "false"}).render(filename="tts", cleanup=True, view=True)
    target = "./VCTK-Corpus/wav48/"
    names = []
    for root, dirnames, filenames in os.walk(target):
        for filename in fnmatch.filter(filenames, "*.wav"):
            names.append(os.path.join(root, filename))
    # names = names[:100]
    data = DataIter(batch_size=params['batch_size'], length=params['length'], names=names, shape=shape)
    opt = mx.optimizer.SGD(momentum=0.9, learning_rate=1e-3)
    init = mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2)
    model = mx.model.FeedForward(symbol=net, ctx=mx.gpu(0), num_epoch=10, optimizer=opt, initializer=init)
    mon = mx.monitor.Monitor(interval=1, stat_func=None, pattern=".*softmax_output", sort=False)
    mon = None
    model.fit(X=data, eval_metric=MYMAE(), monitor=mon, batch_end_callback=mx.callback.Speedometer(batch_size, 10))
