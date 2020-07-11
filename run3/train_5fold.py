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
epochs = 50
learning_rate = 0.0005
batch_size = 20
dropout = 0.5
output_path = 'pin/results_vgg_final_4096_1024_128_0.25_30_5e-6_random/'

eval_batch_size = 20

# load data
data_path = 'pin/training/**/*.jpg'
data_list = glob.glob(data_path)
# shuffle data
# np.random.seed(100)
np.random.shuffle(data_list)

# 5-fold cross validation
n_fold = 5
fold_size = len(data_list) // n_fold

overall_acc = []
for fold in range(n_fold):
    fold_output_path = output_path + '/fold_' + str(fold)
    # divide data for 5 fold, each fold contains 300 data
    # every iteration, 1200 data for training, 300 data for validation
    valid_list = data_list[fold*fold_size:(fold+1)*fold_size]
    train_list = [d for d in data_list if d not in valid_list]

    train_provider = DataProvider(train_list, is_shuffle=True)
    valid_provider = DataProvider(valid_list, is_shuffle=False)

    # build network
    net = VGG19(15, retrain=True, weights=None)

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
        valid_loss = []
        valid_acc = []
        # train set
        for _ in range(train_provider.size() // eval_batch_size):
            xs, ys = train_provider(eval_batch_size)
            logits = net(xs)
            sub_loss = loss_function(logits=logits, labels=ys)
            sub_acc = logits_accuracy(logits, ys)
            train_loss.append(sub_loss)
            train_acc.append(sub_acc)
            
        # validation set
        for _ in range(valid_provider.size() // eval_batch_size):
            xs, ys = valid_provider(eval_batch_size)
            logits = net(xs)
            sub_loss = loss_function(logits=logits, labels=ys)
            sub_acc = logits_accuracy(logits, ys)
            valid_loss.append(sub_loss)
            valid_acc.append(sub_acc)
        
        train_loss = np.mean(train_loss)
        train_acc = np.mean(train_acc)
        valid_loss = np.mean(valid_loss)
        valid_acc = np.mean(valid_acc)

        # save best checkpoint according to the loss of validation set
        if valid_loss < best_loss:
            ckpt.write(fold_output_path + '/ckpt/best')
            best_loss = valid_loss

        # train log printing
        train_log = ('epoch {}:'.format(ep) + \
                    '\n  training   : loss {:.4f}, accuracy {:.4f}'.format(train_loss, train_acc) + \
                    '\n  validation : loss {:.4f}, accuracy {:.4f}'.format(valid_loss, valid_acc))
        print(train_log)

    ckpt.write(fold_output_path + '/ckpt/final')
    overall_acc.append(valid_acc)

# print and save results
results_str = 'Accuracy for 5 fold: '
for i in range(n_fold):
    results_str += '{:.4f} '.format(overall_acc[i])
results_str += ' Overall accuracy: {:.4f}'.format(np.mean(overall_acc))
print(results_str)
with open(output_path + '/results.txt', 'a+') as f:
    f.write(results_str + '\n')

