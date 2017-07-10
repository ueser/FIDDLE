from itertools import groupby
import json
import logging
from operator import itemgetter
import os
import shutil
import tempfile

import numpy as np
import scipy.sparse
from scipy.sparse import coo_matrix

import bcolz
from pybedtools import BedTool
from pysam import FastaFile
from tqdm import tqdm
import wWigIO

from .util import makedirs
from .util import one_hot_encode_sequence

_logger = logging.getLogger('genomedatalayer')
SHM_DIR = '/dev/shm'
NUM_SEQ_CHARS = 4

_blosc_params = bcolz.cparams(clevel=5, shuffle=bcolz.SHUFFLE, cname='lz4')

_array_writer = {
    'numpy': lambda arr, path: np.save(path, arr),
    'bcolz': lambda arr, path: bcolz.carray(
        arr, rootdir=path, cparams=_blosc_params, mode='w').flush()
}


def extract_fasta_to_npy(fasta, output_dir):
    fasta_file = FastaFile(fasta)
    file_shapes = {}
    for chrom, size in zip(fasta_file.references, fasta_file.lengths):
        data = np.empty((NUM_SEQ_CHARS, size), dtype=np.float32)
        seq = fasta_file.fetch(chrom)
        one_hot_encode_sequence(seq, data)
        np.save('{}.npy'.format(os.path.join(output_dir, chrom)), data)
        file_shapes[chrom] = data.shape

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as fp:
        json.dump({'file_shapes': file_shapes,
                   'type': 'array',
                   'source': fasta}, fp)


def extract_bigwig_to_npy(bigwig, output_dir, dtype=np.float32):
    wWigIO.open(bigwig)
    chrom_sizes = wWigIO.getChromSize(bigwig)
    file_shapes = {}
    for chrom, size in zip(*chrom_sizes):
        data = np.empty(size)
        wWigIO.getData(bigwig, chrom, 0, size, data)
        np.save('{}.npy'.format(os.path.join(output_dir, chrom)),
                data.astype(dtype))
        file_shapes[chrom] = data.shape
    wWigIO.close(bigwig)

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as fp:
        json.dump({'file_shapes': file_shapes,
                   'type': 'array',
                   'source': bigwig}, fp)





def read_genome_sizes(genome_file):
    with open(genome_file) as fp:
        chr2size = {}
        for line in fp:
            chrom, size = line.split()
            chr2size[chrom] = int(size)
    return chr2size




class Array2D(object):
    # This object exists because it's more efficient for 2D arrays to be stored as
    # their transpose, but bcolz.carray does not support transposition
    # efficiently. Furthermore, the carray is efficient at slicing on the
    # first dimension, so we store the transpose since we typically slice on
    # the second (coords).

    def __init__(self, rootdir, mode='r'):
        self._arr = bcolz.open(rootdir, mode=mode)

    def __getitem__(self, key):
        r, c = key
        return self._arr[c, r].T

    def __setitem__(self, key, item):
        r, c = key
        self._arr[c, r] = item

    @property
    def shape(self):
        return self._arr.shape[::-1]

    @property
    def ndim(self):
        return self._arr.ndim

    def copy(self):
        """exposes bcolz.carray.copy()"""
        self._arr = self._arr.copy()
        return self


