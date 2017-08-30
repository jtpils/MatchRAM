import argparse
import mnistpair_pb2
from random import shuffle
import lmdb

def parseArgs():
    parser = argparse.ArgumentParser(description='Dual RAM Accuracy')
    parser.add_argument('--data', dest='data', default='data/mnistexactpair', type=str)
    parser.add_argument('--output', dest='output', default='data/mnistexactshuffled', type=str)
    return parser.parse_args()

def main():
    args = parseArgs()
    readEnv = lmdb.open(args.data, readonly=True)
    writeEnv = lmdb.open(args.output, map_size=1024 * 1024 * 1024 * 1024)

    with readEnv.begin() as readTxn:
        with writeEnv.begin(write=True) as writeTxn:
            cursor = readTxn.cursor()
            cursor.last()
            lastKey, _ = cursor.item()
            keys = ['{:08}'.format(k) for k in range(int(lastKey))]
            shuffle(keys)
            writeIndex = 0
            for key in keys:
                value = readTxn.get(key)
                writeTxn.put('{:08}'.format(writeIndex), value)
                writeIndex += 1

if __name__ == '__main__':
    main()
