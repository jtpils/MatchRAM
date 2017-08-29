import argparse
import mnistpair_pb2
from random import shuffle
import lmdb

def parseArgs():
    parser = argparse.ArgumentParser(description='Dual RAM Accuracy')
    parser.add_argument('--data', dest='data', default='data/mnistpair', type=str)
    parser.add_argument('--output', dest='output', default='data/mnistshuffled', type=str)
    return parser.parse_args()

def main():
    args = parseArgs()
    readEnv = lmdb.open(args.data, readonly=True)

    with readEnv.begin() as txn:
        cursor = txn.cursor()
        cursor.last()
        key, _ = cursor.item()
        print(type(key))

    # print('Finished reading dataset, now shuffling')
    # shuffle(mnist)
    #
    # print('Writing the dataset')
    # with writeEnv.begin(write=True) as txn:
    #     for key, value in mnist:
    #         txn.put(key, value)

if __name__ == '__main__':
    main()
