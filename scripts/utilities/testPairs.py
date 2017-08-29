import argparse
import mnistpair_pb2
import numpy as np
import lmdb

def show(frame1, frame2, key, label):
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax1 = fig.add_subplot(1,2,1)
    imgplot = ax1.imshow(frame1, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax2 = fig.add_subplot(1,2,2)
    imgplot = ax2.imshow(frame2, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    pyplot.title('key: ' + key + ', label: ' + str(label))
    pyplot.show()

def parseArgs():
    parser = argparse.ArgumentParser(description='Visual MNIST Pair Tester')
    parser.add_argument('--mnist', dest='mnist', default='data/mnistshuffled', type=str)
    parser.add_argument('--count', dest='count', default=20, type=int)
    return parser.parse_args()

def main():
    args = parseArgs()
    env = lmdb.open(args.mnist, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        cursor.first()
        index = 0
        while cursor.next():
            if index > args.count:
                break
            key, value = cursor.item()
            datum = mnistpair_pb2.Datum()
            datum.ParseFromString(value)
            frame1 = np.fromstring(datum.frames[0], dtype=np.uint8)
            frame1 = frame1.reshape(datum.channels, datum.height, datum.width)
            frame2 = np.fromstring(datum.frames[1], dtype=np.uint8)
            frame2 = frame2.reshape(datum.channels, datum.height, datum.width)
            label = datum.label
            show(np.squeeze(frame1), np.squeeze(frame2), key, label)
            index += 1

if __name__ == '__main__':
    main()
