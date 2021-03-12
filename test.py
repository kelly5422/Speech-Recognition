import os
os.environ['KMP_WARNINGS']='off'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import operator
import random
import time

import numpy as np
import csv
import pandas as pd
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from audio_reader import AudioReader
from file_logger import FileLogger
from utils import FIRST_INDEX, sparse_tuple_from
from utils import convert_inputs_to_ctc_format
from fast_ctc_decode import beam_search, viterbi_search
#from scipy.special import softmax

sample_rate = 16000
# Some configs
num_features = 78  # log filter bank or MFCC features
# Accounting the 0th index +  space + blank label = 28 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1

# Hyper-parameters
num_epochs = 1
num_hidden = 1024
batch_size = 346
num_examples = 1
num_batches_per_epoch = 1

# make sure the values match the ones in generate_audio_cache.py
audio = AudioReader(audio_dir='test',
                    cache_dir='cache_test',
                    sample_rate=sample_rate)

def next_batch(bs=batch_size, train=True):
    x_batch = []
    y_batch = []
    seq_len_batch = []
    original_batch = []
    i=0
    for k in range(bs):
        ut_length_dict = dict([(k, len(v['target'])) for (k, v) in audio.cache.items()])
        utterances = sorted(ut_length_dict.items(), key=operator.itemgetter(1))
        test_index = 346
        if train:
            utterances = [a[0] for a in utterances[test_index:]]
        else:
            utterances = [a[0] for a in utterances[:test_index]]
        training_element = audio.cache[utterances[i]]
        target_text = training_element['target']
        audio_buffer = training_element['audio']
        x, y, seq_len, original = convert_inputs_to_ctc_format(audio_buffer,
                                                               sample_rate,
                                                               'whatever',
                                                               num_features)
        x_batch.append(x)
        y_batch.append(y)
        seq_len_batch.append(seq_len)
        original_batch.append(original)
        i+=1

    y_batch = sparse_tuple_from(y_batch)
    seq_len_batch = np.array(seq_len_batch)[:, 0]
    for i, pad in enumerate(np.max(seq_len_batch) - seq_len_batch):
        x_batch[i] = np.pad(x_batch[i], ((0, 0), (0, pad), (0, 0)), mode='constant', constant_values=0)

    x_batch = np.concatenate(x_batch, axis=0)

    return x_batch, y_batch, seq_len_batch, original_batch


def decode_batch(d, original, phase='training'):
    for jj in range(batch_size):
        values = d.values[np.where(d.indices[:, 0] == jj)[0]]
        str_decoded = ''.join([chr(x) for x in np.asarray(values) + FIRST_INDEX])
        # Replacing blank label to none
        str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
        # Replacing space label to space
        str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
        
        #print(str_decoded)

        f = open('pre.txt', 'a')
        print(str_decoded, file=f)
        f.close()

    with open('pre.txt', 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split(',') for line in stripped if line)
        with open('pre.csv', 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(['text'])
            writer.writerows(lines)


    df = pd.read_csv('pre.csv')
    df.index = np.arange(1, len(df) + 1)
    df.index.names = ['id']
    df.to_csv('CTC_result.csv')


def run_ctc():
    graph = tf.Graph()
    with graph.as_default():
        # e.g: log filter bank or MFCC features
        # Has size [batch_size, max_step_size, num_features], but the
        # batch_size and max_step_size can vary along each step
        inputs = tf.placeholder(tf.float32, [None, None, num_features], name='inputs')

        # Here we use sparse_placeholder that will generate a
        # SparseTensor required by ctc_loss op.
        targets = tf.sparse_placeholder(tf.int32, name='targets')

        # 1d array of size [batch_size]
        seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

        # Defining the cell
        # Can be:
        #   tf.nn.rnn_cell.RNNCell
        #   tf.nn.rnn_cell.GRUCell
        def get_a_cell():
            return tf.nn.rnn_cell.GRUCell(num_hidden)

        # Stacking rnn cells
        stack = tf.contrib.rnn.MultiRNNCell([get_a_cell() for _ in range(1)], state_is_tuple=True)

        # The second output is the last state and we will no use that
        outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)


        shape = tf.shape(inputs)
        batch_s, max_time_steps = shape[0], shape[1]

        # Reshaping to apply the same weights over the timesteps
        outputs = tf.reshape(outputs, [-1, num_hidden])

        # Truncated normal with mean 0 and stdev=0.1
        # Tip: Try another initialization
        # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
        W = tf.Variable(tf.truncated_normal([num_hidden,
                                             num_classes],
                                            stddev=0.1))
        # Zero initialization
        # Tip: Is tf.zeros_initializer the same?
        b = tf.Variable(tf.constant(0., shape=[num_classes]))

        # Doing the affine projection
        logits = tf.matmul(outputs, W) + b

        # Reshaping back to the original shape
        logits = tf.reshape(logits, [batch_s, -1, num_classes])

        # Time major
        logits = tf.transpose(logits, (1, 0, 2))

        loss = tf.nn.ctc_loss(targets, logits, seq_len)
        cost = tf.reduce_mean(loss)

        # optimizer = tf.train.AdamOptimizer().minimize(cost)
        # optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(cost)
        optimizer = tf.train.AdamOptimizer(learning_rate=5e-4).minimize(cost)

        # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

        # Inaccuracy: label error rate
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                              targets))

    with tf.Session(graph=graph) as session:

        saver = tf.train.Saver(max_to_keep=None)

        if os.path.exists('./model/checkpoint'):
            saver.restore(session, './model/1919-1920')
        else:
            init = tf.global_variables_initializer()
            session.run(init)

        for curr_epoch in range(num_epochs):
            for batch in range(num_batches_per_epoch):
                val_inputs, val_targets, val_seq_len, val_original = next_batch(train=False)
                val_feed = {inputs: val_inputs,
                        targets: val_targets,
                        seq_len: val_seq_len}

                d = session.run(decoded[0], feed_dict=val_feed)
                decode_batch(d, val_original, phase='validation')


if __name__ == '__main__':
    run_ctc()
