import argparse
import numpy as np
import idx2numpy
from scipy.io import loadmat
import random

def onehotToDigit(onehot):
    return np.where(onehot)[0][0]

def digitToOnehot(digit):
    onehot = np.zeros(10)
    onehot[digit] = 1
    return onehot

def show(example, label):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
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

    for image, onehot in zip(images, onehots):
        shuffled = random.sample(digits, len(digits))
        label = onehotToDigit(onehot)
        show(image.reshape(40,40), label)

    # example = images[2, :].reshape(40, 40)
    # label = onehotToDigit(onehots[2, :])
    # show(example, label)


if __name__ == '__main__':
    main()
