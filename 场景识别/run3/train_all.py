import glob
import numpy as np
import tensorflow as tf

from data_loader import DataProvider 
from model import ResNet50, VGG19

def logits_accuracy(logits, ys):
    pred = tf.nn.softmax(logits)
    pred_argmax = np.argmax(pred, -1)
    ys = np.argmax(ys, -1)
    accuracy = np.mean(np.equal(pred_argmax, ys))
    return accuracy

# hyper-parameter setting
epochs = 30
learning_rate = 0.000005
batch_size = 20
dropout = 0.25
output_path = 'pin/results_all/'

eval_batch_size = 20

# load data
data_path = 'pin/training/**/*.jpg'
data_list = glob.glob(data_path)

train_provider = DataProvider(data_list, is_shuffle=True)


# build network
net = VGG19(15, retrain=True)

# select optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate)

# loss function (cross entropy)
loss_function = tf.nn.softmax_cross_entropy_with_logits

# check point saver
ckpt = tf.train.Checkpoint(net=net)

# calculate iterations each epoch
assert train_provider.size() % batch_size == 0, 'Wrong batch size!'
its = train_provider.size() // batch_size

# train
best_loss = float('inf')
for ep in range(epochs):
    for _ in range(its):
        xs, ys = train_provider(batch_size)
        with tf.GradientTape() as tape:
            logits = net(xs, dropout)
            loss = loss_function(logits=logits, labels=ys)                       
        grads = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))

    # evaluation
    train_loss = []
    train_acc = []
    # train set
    for _ in range(train_provider.size() // eval_batch_size):
        xs, ys = train_provider(eval_batch_size)
        logits = net(xs)
        sub_loss = loss_function(logits=logits, labels=ys)
        sub_acc = logits_accuracy(logits, ys)
        train_loss.append(sub_loss)
        train_acc.append(sub_acc)
        
    train_loss = np.mean(train_loss)
    train_acc = np.mean(train_acc)

    # train log printing
    train_log = ('epoch {}:'.format(ep) + \
                '\n  training   : loss {:.4f}, accuracy {:.4f}'.format(train_loss, train_acc))
    print(train_log)

ckpt.write(output_path + '/ckpt/final')


