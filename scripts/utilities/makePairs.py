import argparse
import numpy as np
import idx2numpy
from scipy.io import loadmat
import random
import lmdb
import mnistpair_pb2

def makeDatum(frame1, frame2, label):
    datum = mnistpair_pb2.Datum()
    datum.channels = 1
    datum.width = frame1.shape[0]
    datum.height = frame1.shape[1]
    datum.frames.extend([frame1.tobytes(), frame2.tobytes()])
    datum.label = label
    return datum

def onehotToDigit(onehot):
    return np.where(onehot)[0][0]

def digitToOnehot(digit):
    onehot = np.zeros(10)
    onehot[digit] = 1
    return onehot

def show(example, label):
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(example, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    pyplot.title(label)
    pyplot.show()

def parseArgs():
    parser = argparse.ArgumentParser(description='MNIST Pair Maker')
    parser.add_argument('--mnist', dest='mnist', type=str)
    parser.add_argument('--output', dest='output', type=str)
    return parser.parse_args()

def main():
    args = parseArgs()
    mnist = loadmat(args.mnist)
    images = np.transpose(mnist['affNISTdata'][0][0][2])
    onehots = np.transpose(mnist['affNISTdata'][0][0][4])

    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    env = lmdb.open(args.output, map_size=1024 * 1024 * 1024 * 1024)

    with env.begin(write=True) as txn:
        lmdbIndex = 0
        for index, (image, onehot) in enumerate(zip(images, onehots)):
            print 'Now creating a pair from here: ', index, images.shape[0]
            shuffled = random.sample(digits, len(digits))
            label = onehotToDigit(onehot)
            for digit in shuffled:
                if digit == label:
                    matchCounter = 0
                    for match, sameOnehot in zip(images[index:, :], onehots[index:, :]):
                        if matchCounter > len(digits) - 1:
                            break
                        if onehotToDigit(sameOnehot) == digit:
                            datum = makeDatum(image.reshape(40, 40), match.reshape(40, 40), 1)
                            txn.put('{:08}'.format(lmdbIndex), datum.SerializeToString())
                            matchCounter += 1
                            lmdbIndex += 1
                else:
                    for nomatch, differentOnehot in zip(images[index:, :], onehots[index:, :]):
                        if onehotToDigit(differentOnehot) == digit:
                            datum = makeDatum(image.reshape(40, 40), nomatch.reshape(40, 40), 0)
                            txn.put('{:08}'.format(lmdbIndex), datum.SerializeToString())
                            lmdbIndex += 1
                            break

if __name__ == '__main__':
    main()
