import pdb, traceback, sys # EDIT

import numpy as np
import six
import time
from tqdm import tqdm as tq
import itertools

def one_hot_encode_sequence(seq):
    # the order of the letters is not arbitrary.
    # Flip the matrix up-down and left-right for reverse compliment
    seq = seq.lower()
    letterdict = {'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1],
               'n': [0.25, 0.25, 0.25, 0.25]}
    result = np.array([letterdict[x] for x in seq])
    return result.T



class MultiModalData(object):
    def __init__(self, train_h5_handle, batch_size):
        self.train_h5_handle = train_h5_handle
        self.batch_size = batch_size

    def batcher(self):
        for batchIdx in itertools.cycle(xrange(0, self.train_h5_handle.values()[0].shape[0]-self.batch_size, self.batch_size)):
            yield {key: inp[batchIdx:(batchIdx + self.batch_size)] for key, inp in self.train_h5_handle.items()}


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print 'elapsed time: %f ms' % self.msecs
