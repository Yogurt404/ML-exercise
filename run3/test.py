import os
import glob
import numpy as np
import tensorflow as tf

from data_loader import DataProvider 
from model import ResNet50, VGG19
from dict_map import ClassMap

# hyper-parameter setting
output_path = 'pin/results/'

eval_batch_size = 15

# load data
data_path = 'pin/testing/*.jpg'
data_list = glob.glob(data_path)
assert len(data_list) % eval_batch_size == 0, 'Wrong batch size'

test_provider = DataProvider(data_list, need_labels=False, is_shuffle=False)

# build network
net = VGG19(15, retrain=True)

# check point
ckpt = tf.train.Checkpoint(net=net)
ckpt.restore(output_path + '/ckpt/final')

    
# evaluation on test set
preds = None
for i in range(test_provider.size() // eval_batch_size):
    xs, ys = test_provider(eval_batch_size)
    logits = net(xs)
    sub_preds = tf.nn.softmax(logits)
    sub_preds = np.argmax(sub_preds, -1)
    if preds is None:
        preds = sub_preds
    else:
        preds = np.concatenate((preds, sub_preds), 0)
    print('pred: {}'.format(i))

classmap = ClassMap()
with open(output_path + '/test.txt', 'a+') as f:
    for i, p in enumerate(data_list):
        idx = os.path.split(p)[-1]
        result = classmap.get_str(preds[i])
        f.write('{} {}\n'.format(idx, result))





