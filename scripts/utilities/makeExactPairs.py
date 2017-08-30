import argparse
import numpy as np
import random
import lmdb
import mnistpair_pb2
from tensorflow.examples.tutorials.mnist import input_data
from numpy.random import random_integers

OUTPUT_IMAGE_LENGTH = 40
INPUT_IMAGE_LENGTH = 28

def makeDatum(frame1, frame2, label):
    datum = mnistpair_pb2.Datum()
    datum.channels = 1
    datum.width = frame1.shape[0]
    datum.height = frame1.shape[1]
    datum.frames.extend([frame1.tobytes(), frame2.tobytes()])
    datum.label = label
    return datum

def makeTransNist(image, inLength, outLength):
    image = image.reshape(inLength, inLength)
    topLeft = random_integers(low=0, high=outLength - inLength, size=2)
    transNist = np.zeros((outLength, outLength))
    transNist[topLeft[0]:topLeft[0]+inLength, topLeft[1]:topLeft[1]+inLength] = image
    return transNist

def parseArgs():
    parser = argparse.ArgumentParser(description='MNIST Exact Pair Maker')
    parser.add_argument('--count', dest='count', default=0, type=int)
    parser.add_argument('--output', dest='output', default='data/mnistexactpair', type=str)
    return parser.parse_args()

def main():
    args = parseArgs()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    images = mnist.train.images
    labels = mnist.train.labels

    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    env = lmdb.open(args.output, map_size=1024 * 1024 * 1024 * 1024)

    with env.begin(write=True) as txn:
        lmdbIndex = 0
        for index, (image, label) in enumerate(zip(images, labels)):
            if args.count and index > args.count:
                break
            print 'Now creating a pair from here: ', index, images.shape[0]
            shuffled = random.sample(digits, len(digits))
            for digit in shuffled:
                if digit == label:
                    for _ in range(len(digits) - 1):
                        frame1 = makeTransNist(image, INPUT_IMAGE_LENGTH, OUTPUT_IMAGE_LENGTH)
                        frame2 = makeTransNist(image, INPUT_IMAGE_LENGTH, OUTPUT_IMAGE_LENGTH)
                        datum = makeDatum(frame1, frame2, 1)
                        txn.put('{:08}'.format(lmdbIndex), datum.SerializeToString())
                        lmdbIndex += 1
                else:
                    for nomatch, differentLabel in zip(images[index:, :], labels[index:]):
                        if differentLabel == digit:
                            frame1 = makeTransNist(image, INPUT_IMAGE_LENGTH, OUTPUT_IMAGE_LENGTH)
                            frame2 = makeTransNist(nomatch, INPUT_IMAGE_LENGTH, OUTPUT_IMAGE_LENGTH)
                            datum = makeDatum(frame1, frame2, 0)
                            txn.put('{:08}'.format(lmdbIndex), datum.SerializeToString())
                            lmdbIndex += 1
                            break

if __name__ == '__main__':
    main()
