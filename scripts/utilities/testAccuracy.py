import argparse
import mnistpair_pb2
import numpy as np
import tensorflow as tf
import lmdb

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
    parser = argparse.ArgumentParser(description='Dual RAM Accuracy')
    parser.add_argument('--meta', dest='meta', default='output/dualram-0.meta', type=str)
    parser.add_argument('--ckpt', dest='ckpt', default='output', type=str)
    parser.add_argument('--data', dest='data', default='data/mnistpair', type=str)
    parser.add_argument('--batch', dest='batch', default=1000, type=int)
    parser.add_argument('--classes', dest='classes', default=2, type=int)
    parser.add_argument('--mc', dest='mc', default=10, type=int)
    return parser.parse_args()

def main():
    args = parseArgs()
    env = lmdb.open(args.data, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        cursor.first()
        _, _, width = getImageSize(cursor)
        cursor, frames1, frames2, labels = getBatch(cursor, width, args.batch)

    saver = tf.train.import_meta_graph(args.meta)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, tf.train.latest_checkpoint(args.ckpt))

        frames1 = np.tile(frames1, [args.mc, 1])
        frames2 = np.tile(frames2, [args.mc, 1])
        labels_val = np.tile(labels, [args.mc])
        graph = tf.get_default_graph()
        softmax = graph.get_tensor_by_name("softmax:0")
        images_ph1 = graph.get_tensor_by_name("images_ph1:0")
        images_ph2 = graph.get_tensor_by_name("images_ph2:0")
        labels_ph = graph.get_tensor_by_name("labels_ph:0")
        softmax_val = sess.run(softmax, feed_dict={ images_ph1: frames1, images_ph2: frames2, labels_ph: labels_val })
        softmax_val = np.reshape(softmax_val, [args.mc, -1, args.classes])
        softmax_val = np.mean(softmax_val, 0)
        pred_labels_val = np.argmax(softmax_val, 1)
        pred_labels_val = pred_labels_val.flatten()
        print(pred_labels_val.shape, labels.shape)
        correct_cnt = np.sum(pred_labels_val == labels)
        acc = correct_cnt / args.batch
        print('valid accuracy = {}'.format(acc))


if __name__ == '__main__':
    main()
