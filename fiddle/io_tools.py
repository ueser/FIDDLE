"""Author: Umut Eser
Documentation: Dylan Marshall

'io_tools.py' handles the various file input and output processing steps
that take place in FIDDLE

Usage:
    To utilize methods and classes in io_tools.py, place the following import
    command in the imports of a python file.

        > from io_tools import *
"""

import pdb, traceback, sys
import numpy as np
import six
import time
from tqdm import tqdm as tq
import itertools

def one_hot_encode_sequence(seq):
    """Transforms DNA sequence to vector form.

    Converts from categorical ATGC characters to an orthogonal, vectorized form.
    The order of characters is not arbitrary. Flip the matrix up-down and
    left-right for the reverse compliment.

    Args:
        :param seq: (string) DNA sequence

    Returns:
        numpy vector: one hot encoded DNA sequence
    """
    seq = seq.lower()
    letterdict = {'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0],
                  't': [0, 0, 0, 1], 'n': [0.25, 0.25, 0.25, 0.25]}
    result = np.array([letterdict[x] for x in seq])
    return result.T

class MultiModalData(object):
    """Training data object capable of being iterated through easily"""

    def __init__(self, train_h5_handle, batch_size):
        """
        Args:
            :param train_h5_handle: (h5py.File) file object in Readonly mode
            :param batch_size: (int) batch input data size, defined in main FLAGS
        """
        self.train_h5_handle = train_h5_handle
        self.batch_size = batch_size

    def batcher(self):
        """Data iterator of input hdf5 dataset, discretized in batch_size segments

        Returns:
            dictionary iterator: for {key = training input types, values = sequencing data}
        """

        iterable = xrange(0, self.train_h5_handle.values()[0].shape[0] - self.batch_size, self.batch_size)
        for batchIdx in itertools.cycle(iterable):
            yield {key: inp[batchIdx:(batchIdx + self.batch_size)] for key, inp in self.train_h5_handle.items()}


class Timer(object):
    """Timer object to monitor rate of computationally intensive steps"""

    def __init__(self, verbose=False):
        """
        Args:
            :param verbose: (boolean) print elapsed time without prompting
        """
        self.verbose = verbose

    def __enter__(self):
        """
        Returns:
            self: time elapsed
        """
        self.start = time.time()
        return self

    def __exit__(self, *args):
        """ """
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print 'elapsed time: %f ms' % self.msecs
