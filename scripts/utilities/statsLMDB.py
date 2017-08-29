from __future__ import division
import argparse
import mnistpair_pb2
import lmdb

def parseArgs():
    parser = argparse.ArgumentParser(description='Statistics of MNIST Pairs')
    parser.add_argument('--mnist', dest='mnist', default='data/mnistshuffled', type=str)
    return parser.parse_args()

def main():
    args = parseArgs()
    env = lmdb.open(args.mnist, readonly=True)
    positives = 0
    negatives = 0
    with env.begin() as txn:
        cursor = txn.cursor()
        cursor.first()
        while cursor.next():
            _, value = cursor.item()
            datum = mnistpair_pb2.Datum()
            datum.ParseFromString(value)
            if datum.label == 0:
                positives += 1
            else:
                negatives += 1
    total = positives + negatives
    print("=================================")
    print("Statistics:")
    print("=================================")
    print("Positives     : " + str(positives))
    print("Negatives     : " + str(negatives))
    print("Total         : " + str(total))
    print("Positives (\%): " + str(positives/total))
    print("Negatives (\%): " + str(negatives/total))
    print("=================================")

if __name__ == '__main__':
    main()
