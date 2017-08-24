from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import tensorflow as tf
import numpy as np

from core import GlimpseNet, LocNet
from core import weight_variable, bias_variable, loglikelihood
from config import Config

import lmdb
from core import mnistpair_pb2

from tensorflow.examples.tutorials.mnist import input_data

logging.getLogger().setLevel(logging.INFO)

loc_mean_arr = []
sampled_loc_arr = []

def getImageSize(cursor):
    cursor.first()
    _, value = cursor.item()
    datum = mnistpair_pb2.Datum()
    datum.ParseFromString(value)
    return datum.channels, datum.height, datum.width

def getBatch(cursor, imageSize, batchSize):
    datum = mnistpair_pb2.Datum()
    index = 0
    frames1  = np.empty((0, imageSize * imageSize), int)
    frames2  = np.empty((0, imageSize * imageSize), int)
    labels = np.empty((0), int)
    while index < batchSize:
        _, value = cursor.item()
        datum.ParseFromString(value)
        frame1 = np.fromstring(datum.frames[0], dtype=np.uint8)
        frame2 = np.fromstring(datum.frames[1], dtype=np.uint8)
        frames1 = np.vstack((frames1, frame1))
        frames2 = np.vstack((frames2, frame2))
        labels = np.hstack((labels, datum.label))
        index = index + 1
        if not cursor.next():
            cursor.first()
    return cursor, frames1, frames2, labels

def parseArgs():
    parser = argparse.ArgumentParser(description='Train Dual Attention Model on Paired Data')
    parser.add_argument('--data', dest='data', default='data/mnistpair', type=str)
    parser.add_argument('--save', dest='save', default='output/dualram', type=str)
    return parser.parse_args()

def main():
    args = parseArgs()
    rnn_cell = tf.nn.rnn_cell
    seq2seq = tf.contrib.legacy_seq2seq

    config = Config()

    # placeholders
    images_ph1 = tf.placeholder(tf.float32, [None, config.original_size * config.original_size * config.num_channels], name="images_ph1")
    images_ph2 = tf.placeholder(tf.float32, [None, config.original_size * config.original_size * config.num_channels], name="images_ph2")
    labels_ph = tf.placeholder(tf.int64, [None], name="labels_ph")

    # Build the aux nets.
    with tf.variable_scope('glimpse_net'):
        gl1 = GlimpseNet(config, images_ph1)
        gl2 = GlimpseNet(config, images_ph2)
    with tf.variable_scope('loc_net'):
        loc_net1 = LocNet(config)
        loc_net2 = LocNet(config)

    def get_next_input(output, i):
        input1, input2 = tf.split(output, num_or_size_splits=2, axis=1)
        loc1, loc_mean1 = loc_net1(input1)
        loc2, loc_mean2 = loc_net2(input2)
        loc_mean_arr.append(tf.concat([loc_mean1, loc_mean2], 1))
        sampled_loc_arr.append(tf.concat([loc1, loc2], 1))
        gl_next = tf.concat([gl1(loc1), gl2(loc2)], 1)
        return gl_next

    # number of examples
    N = tf.shape(images_ph1)[0]
    init_loc1 = tf.random_uniform((N, 2), minval=-1, maxval=1)
    init_loc2 = tf.random_uniform((N, 2), minval=-1, maxval=1)
    init_glimpse1 = gl1(init_loc1)
    init_glimpse2 = gl2(init_loc2)

    # Core network.
    lstm_cell = rnn_cell.LSTMCell(config.cell_size, state_is_tuple=True)
    init_state = lstm_cell.zero_state(N, tf.float32)

    inputs = [tf.concat([init_glimpse1, init_glimpse2], 1)]
    inputs.extend([0] * config.num_glimpses)
    outputs, _ = seq2seq.rnn_decoder(inputs, init_state, lstm_cell, loop_function=get_next_input)

    # Time independent baselines
    with tf.variable_scope('baseline'):
        w_baseline = weight_variable((config.cell_output_size, 1))
        b_baseline = bias_variable((1,))
    baselines = []
    for _, output in enumerate(outputs[1:]):
        baseline_t = tf.nn.xw_plus_b(output, w_baseline, b_baseline)
        baseline_t = tf.squeeze(baseline_t)
        baselines.append(baseline_t)
    baselines = tf.stack(baselines)  # [timesteps, batch_sz]
    baselines = tf.transpose(baselines)  # [batch_sz, timesteps]

    # Take the last step only.
    output = outputs[-1]
    # Build classification network.
    with tf.variable_scope('cls'):
        w_logit = weight_variable((config.cell_output_size, config.num_classes))
        b_logit = bias_variable((config.num_classes,))
    logits = tf.nn.xw_plus_b(output, w_logit, b_logit)
    softmax = tf.nn.softmax(logits, name='softmax')

    # cross-entropy.
    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph)
    xent = tf.reduce_mean(xent)
    pred_labels = tf.argmax(logits, 1)
    # 0/1 reward.
    reward = tf.cast(tf.equal(pred_labels, labels_ph), tf.float32)
    rewards = tf.expand_dims(reward, 1)  # [batch_sz, 1]
    rewards = tf.tile(rewards, (1, config.num_glimpses))  # [batch_sz, timesteps]
    logll = loglikelihood(loc_mean_arr, sampled_loc_arr, config.loc_std)
    advs = rewards - tf.stop_gradient(baselines)
    logllratio = tf.reduce_mean(logll * advs)
    reward = tf.reduce_mean(reward)

    baselines_mse = tf.reduce_mean(tf.square((rewards - baselines)))
    var_list = tf.trainable_variables()

    # hybrid loss
    loss = -logllratio + xent + baselines_mse  # `-` for minimize
    grads = tf.gradients(loss, var_list)
    grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)

    # learning rate
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    steps_per_epoch = config.steps_per_epoch
    starter_learning_rate = config.lr_start

    # decay per training epoch
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, steps_per_epoch, 0.97, staircase=True)
    learning_rate = tf.maximum(learning_rate, config.lr_min)
    opt = tf.train.AdamOptimizer(learning_rate)
    train_op = opt.apply_gradients(zip(grads, var_list), global_step=global_step)

    saver = tf.train.Saver()
    env = lmdb.open(args.data, readonly=True)

    with env.begin() as txn:
        cursor = txn.cursor()
        cursor.first()
        _, _, width = getImageSize(cursor)
        cursor, framesval1, framesval2, labelsval = getBatch(cursor, width, config.eval_batch_size)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for epoch in xrange(config.epochs):
                cursor, frames1, frames2, labels = getBatch(cursor, width, config.batch_size)
                frames1 = np.tile(frames1, [config.M, 1])
                frames2 = np.tile(frames2, [config.M, 1])
                labels = np.tile(labels, [config.M])
                loc_net1.sampling = True
                loc_net2.sampling = True
                for step in xrange(steps_per_epoch):
                    adv_val, baselines_mse_val, xent_val, logllratio_val, reward_val, loss_val, lr_val, _ \
                        = sess.run([advs, baselines_mse, xent, logllratio, reward, loss, learning_rate, train_op],
                            feed_dict={ images_ph1: frames1, images_ph2: frames2, labels_ph: labels })
                    if step and step % (steps_per_epoch // 10) == 0:
                        logging.info('epoch {} step {}: lr = {:3.6f}'.format(epoch, step, lr_val))
                        logging.info('epoch {} step {}: reward = {:3.4f}\tloss = {:3.4f}\txent = {:3.4f}'.format(epoch, step, reward_val, loss_val, xent_val))
                        logging.info('llratio = {:3.4f}\tbaselines_mse = {:3.4f}'.format(logllratio_val, baselines_mse_val))

                # Do validation once per epoch
                correct_cnt = 0
                framesnew1 = np.tile(framesval1, [config.M, 1])
                framesnew2 = np.tile(framesval2, [config.M, 1])
                labelsnew = np.tile(labelsval, [config.M])
                softmax_val = sess.run(softmax, feed_dict={ images_ph1: framesnew1, images_ph2: framesnew2, labels_ph: labelsnew })
                softmax_val = np.reshape(softmax_val, [config.M, -1, config.num_classes])
                softmax_val = np.mean(softmax_val, 0)
                pred_labels_val = np.argmax(softmax_val, 1)
                pred_labels_val = pred_labels_val.flatten()
                correct_cnt += np.sum(pred_labels_val == labelsval)
                acc = correct_cnt / config.eval_batch_size
                logging.info('valid accuracy = {}'.format(acc))
                saver.save(sess, args.save, global_step=epoch)

if __name__ == '__main__':
    main()
