from bx.bbi.bigwig_file import BigWigFile
import pdb, traceback, sys # EDIT
import pickle
import json
import h5py
import os
import itertools
import numpy as np
import six
import bcolz
from pysam import FastaFile
import roman as rm
import time
from tqdm import tqdm as tq
import tensorflow as tf

NUM_SEQ_CHARS = 4
_blosc_params = bcolz.cparams(clevel=5, shuffle=bcolz.SHUFFLE, cname='lz4')

def one_hot_encode_sequence(seq):
    # the order of the letters is not arbitrary.
    # Flip the matrix up-down and left-right for reverse compliment
    seq = seq.lower()
    letterdict = {'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1],
               'n': [0.25, 0.25, 0.25, 0.25]}
    result = np.array([letterdict[x] for x in seq])
    return result.T


def extract_bigwig_to_file(bigwigs, output_dir, chrom_sizes, overwrite, dtype=np.float32):
    """
    Returns compressed version of bigwig file for a quickly accessible memory map
    Args:
        bigwigs: list of bigwig files corresponding to strands or channels of the same dataset
        output_dir: output directory for memory map location
        chrom_sizes: a dict object that holds the pairs of chromosome names and sizes
        dtype: data type - single precision float
        overwrite: boolean - whether to overwrite current memory map
    """
    for i in [0, 1]:
        if overwrite:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            bw = [BigWigFile(open(bigwig, 'r')) for bigwig in bigwigs]
            file_shapes = {}
            for chrom, size in six.iteritems(chrom_sizes):
                data = np.empty((len(bw), size[1]))
                for j in range(len(bw)):
                    data[j, :] = bw[j].get_as_array(chrom, 0, size[1]).astype(dtype)
                bcolz.carray(data, rootdir=os.path.join(output_dir, chrom), cparams=_blosc_params, mode='w').flush()
                file_shapes[chrom] = data.shape
            metadata = {'file_shapes': file_shapes,
                        'type': 'array_{}'.format('bcolz'),
                        'extractor': 'CompressedBigwigExtractor',
                        'source': bigwigs}
            with open(os.path.join(output_dir, 'metadata.json'), 'w') as fp:
                json.dump(metadata, fp)
                overwrite = False
            break
        else:
            try:
                with open(os.path.join(output_dir, 'metadata.json'), 'r') as fp:
                    metadata = json.load(fp)
                break
            except IOError as e:
                print("I/O error({0}): {1} for {2}".format(e.errno, e.strerror, output_dir))
                print("There is a problem with opening the metadata. Recreating the mmap files and overwriting...")
                overwrite = True
    return metadata


def extract_fasta_to_file(fasta, output_dir, overwrite):
    """
    Returns compressed version of fasta file for a quickly accessible memory map
    Args:
        fasta: fasta file to be converted
        output_dir: output directory for memory map location
        overwrite: boolean - whether to overwrite current memory map
    """

    for i in [0, 1]:
        if overwrite:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            fasta_file = FastaFile(fasta)
            file_shapes = {}
            for chrom, size in zip(fasta_file.references, fasta_file.lengths):
                seq = fasta_file.fetch(chrom)
                data = one_hot_encode_sequence(seq)
                file_shapes[chrom] = data.shape
                bcolz.carray(data, rootdir=os.path.join(output_dir, chrom), cparams=_blosc_params, mode='w').flush()
            mode = '2D_transpose_bcolz'
            metadata = {'file_shapes': file_shapes,
                        'type': 'array_{}'.format(mode),
                        'extractor': 'CompressedFastaExtractor',
                        'source': fasta}
            with open(os.path.join(output_dir, 'metadata.json'), 'w') as fp:
                json.dump(metadata, fp)
                overwrite = False
        else:
            try:
                with open(os.path.join(output_dir, 'metadata.json'), 'r') as fp:
                    metadata = json.load(fp)
                break
            except IOError as e:
                print("I/O error({0}): {1} for {2}".format(e.errno, e.strerror, output_dir))
                print("There is a problem with opening the metadata. Recreating the mmap files and overwriting...")
                overwrite = True
    return metadata


def save_for_fast_training_hdf5(output_dir, data_tracks, regions, filename):
    """Saves to '../data/hdf5datasets/' a hdf5 file of data from the provided bigwig and
    fasta files at the specified chromosomal regions of interest.

    Args:
        data_tracks: object containing sequencing dataset inputs in iterable format (bigwig, fasta)
        regions: file listing regions to extract from data_tracks object, ideally constructed with the
                 provided generate_regions.py script
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    f = h5py.File(os.path.join(output_dir, filename + '.h5'), 'w') # initialize hdf5 file
    h5_handle = {}
    regions.sort()

    width = regions[0].length # length of regions of interest
    wrongShape = 0 # counter of wrongly shaped numpy vector insertion attempts
    numInsane = 0 # number of vector choices through regions that have NaNs
    regionCount = 0 # initializing count of viable regions

    resultingData = dict.fromkeys(data_tracks.keys())
    for key in resultingData.keys():
        resultingData[key] = np.zeros((regions.count(), data_tracks[key]['metadata']['input_height'], width, 1))

    tempData = dict.fromkeys(data_tracks.keys())
    for key in tempData.keys():
        tempData[key] = np.zeros((data_tracks[key]['metadata']['input_height'], width))

    for reg in tq(regions):
        any_NaNs = False
        is_wrong_shape = False
        for key in data_tracks.keys():
            if key != "DNAseq":
                tempBigWigVector = [] # placeholder for bigwig vectors
                for i, caller in enumerate(data_tracks[key]['caller']):
                    # pull out data from bigwig file at region of interest
                    bigWigVector = caller(reg.chrom, 0, regions.chromsizes[reg.chrom][1])
                    tempBigWigVector.append(bigWigVector)
                fetched_data = np.array(tempBigWigVector)
                vector = fetched_data[:, reg.start:reg.end]
            else: # one hot encode DNAseq data
                for i, caller in enumerate(data_tracks["DNAseq"]['caller']):
                    # pull out data from fasta file at region of interest
                    # pdb.set_trace()
                    fastaMatrix = one_hot_encode_sequence(caller(str(reg.chrom), 0, regions.chromsizes[reg.chrom][1]))
                fetched_data = fastaMatrix
                vector = fetched_data[:, reg.start:reg.end]
            if not sanity_check(vector):
                any_NaNs = True
            else:
                if tempData[key][:, :].shape == vector.shape:
                    tempData[key][:, :] = vector
                else:
                    is_wrong_shape = True
        if any_NaNs:
            numInsane += 1
            if is_wrong_shape:
                wrongShape += 1
            continue
        for key in resultingData.keys():
            resultingData[key][regionCount, :, :, 0] = tempData[key][:, :]
        regionCount += 1

    print('number of regions with NaN data = ' + str(numInsane))
    print('number of regions with incorrect shape = ' + str(wrongShape))
    idx = np.arange(regionCount) # randomly shuffle inputs
    np.random.shuffle(idx)
    for key in data_tracks.keys():
        h5_handle[key] = f.create_dataset(key, (regionCount, data_tracks[key]['metadata']['input_height'], width, 1))
        h5_handle[key][:] = resultingData[key][idx]
    f.close()


def sanity_check(vec):
    if np.isnan(vec).any():
        return False
    else:
        return True


def read_genome_sizes(genome_file):
    with open(genome_file) as fp:
        chr2size = {}
        for line in fp:
            chrom, size = line.split()
            chr2size[chrom] = int(size)
    return chr2size


def load_directory(base_dir, metadata=None):
    # type: (str) -> object
    if metadata is None:
        with open(os.path.join(base_dir, 'metadata.json'), 'r') as fp:
            metadata = json.load(fp)

    if metadata['type'] == 'array_bcolz':
        data = {chrom: bcolz.open(os.path.join(base_dir, chrom), mode='r')
                for chrom in metadata['file_shapes']}

        for chrom, shape in six.iteritems(metadata['file_shapes']):
            if data[chrom].shape != tuple(shape):
                raise ValueError('Inconsistent shape found in metadata file: '
                                 '{} - {} vs {}'.format(chrom, shape,
                                                        data[chrom].shape))

    elif metadata['type'] in ['array_2D_transpose_bcolz']:
        data = {chrom: Array2D(os.path.join(base_dir, chrom), mode='r')
                for chrom in next(os.walk(base_dir))[1]}
        for chrom, shape in six.iteritems(metadata['file_shapes']):
            if (data[chrom].shape[0] != shape[1]) or (data[chrom].shape[1] != shape[0]):
                raise ValueError('Inconsistent shape found in metadata file: '
                                 '{} - {} vs {}'.format(chrom, shape,
                                                        data[chrom].shape))
    else:
        raise ValueError('Not found...')

    return data, metadata


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


class BaseExtractor(object):
    dtype = np.float32

    def __init__(self, datafile, height=None, **kwargs):
        self._datafile = datafile
        self._height = height

    def __call__(self, intervals, to_mirror=None, out=None, **kwargs):
        data = self._check_or_create_output_array(intervals, out, height=self._height)

        self._extract(intervals, data, **kwargs)
        if to_mirror is not None:
            self.mirror(data, to_mirror)
        return data

    @classmethod
    def _check_or_create_output_array(cls, intervals, out, height=None):
        width = intervals[0].stop - intervals[0].start
        output_shape = (len(intervals), height, width, 1)

        if out is None:
            out = np.zeros(output_shape, dtype=cls.dtype)
        else:
            if out.shape != output_shape:
                raise ValueError('out array has incorrect shape: {} '
                                 '(need {})'.format(out.shape, output_shape))
            if out.dtype != cls.dtype:
                raise ValueError('out array has incorrect dtype: {} '
                                 '(need {})'.format(out.dtype, cls.dtype))
        return out

    def _extract(self, intervals, out, **kwargs):
        'Subclassses should implement this and return the data'
        raise NotImplementedError

    @staticmethod
    def mirror(data, to_mirror):
        for index, mirror in enumerate(to_mirror):
            if mirror:
                data[index, :, :, :] = data[index, :, ::-1, :]


class MemmappedExtractor(BaseExtractor):
    multiprocessing_safe = False

    def __init__(self, datafile, metadata=None, **kwargs):
        super(MemmappedExtractor, self).__init__(datafile, metadata=None, **kwargs)
        self._data, _ = load_directory(datafile, metadata=metadata)
        ix = 1 if metadata['extractor'] == 'CompressedFastaExtractor' else 0
        self._height = metadata['file_shapes'].values()[0][ix]

    def _mm_extract(self, intervals, out, **kwargs):
        mm_data = self._data
        for index, interval in enumerate(intervals):

            try:
                out[index, :, :, 0] = \
                    mm_data[interval.chrom][:, interval.start:interval.stop]
            except ValueError:
                print(interval.chrom, interval.start, interval.stop)
                print(mm_data[interval.chrom].shape, out.shape)
                raise

    _extract = _mm_extract

    @staticmethod
    def setup_mmap_arrays(datafile, output_dir, overwrite=False):
        """Subclasses should implement this.
        This function is responsible for creating the output_dir."""
        raise NotImplementedError


class FastaExtractor(BaseExtractor):
    multiprocessing_safe = True

    def _extract(self, intervals, out, **kwargs):
        fasta = FastaFile(self._datafile)

        for index, interval in enumerate(intervals):
            seq = fasta.fetch(str(interval.chrom), interval.start,
                              interval.stop)

            out[index, :, :, 0] = one_hot_encode_sequence(seq)

        return out

    def _set_height(self):
        self._height = NUM_SEQ_CHARS

    @staticmethod
    def mirror(data, to_mirror):
        for index, mirror in enumerate(to_mirror):
            if mirror:
                data[index, :, :, :] = data[index, ::-1, ::-1, :]


class CompressedFastaExtractor(MemmappedExtractor, FastaExtractor):
    multiprocessing_safe = False
    setup_mmap_arrays = staticmethod(extract_fasta_to_file)


class CompressedBigwigExtractor(MemmappedExtractor):
    def _extract(self, intervals, out, **kwargs):
        out[:] = self._bigwig_extractor(intervals, **kwargs)

        return out

    def _bigwig_extractor(self, intervals, out=None, **kwargs):
        if out is None:
            width = intervals[0].stop - intervals[0].start
            out = np.empty((len(intervals), self._height, width, 1))

        self._mm_extract(intervals, out, **kwargs)
        return out

    setup_mmap_arrays = staticmethod(extract_bigwig_to_file)


def batch_iter(iterable, batch_size):
    '''iterates in batches.
    '''
    it = iter(iterable)
    try:
        while True:
            values = []
            for n in xrange(batch_size):
                values += (it.next(),)
            yield values
    except StopIteration:
        # yield remaining values
        yield values


def infinite_batch_iter(iterable, batch_size):
    '''iterates in batches indefinitely.
    '''
    return batch_iter(itertools.cycle(iterable),
                      batch_size)



#TODO: create an alternative to mmap such that the train data is saved in parallel, or queue multi process

class MultiModalData(object):
    def __init__(self, data_tracks):
        '''
        :param data_tracks: dict object that holds the names and other properties of input datasets
        '''
        self.data_tracks = data_tracks

        # Instantiate an extractor object for each data track
        self.extractors =\
            {track_name: eval(self.data_tracks[track_name]['metadata']['extractor'])(self.data_tracks[track_name]['path_to_mmap'],
                                                                                     metadata=self.data_tracks[track_name]['metadata'])
        for track_name in self.data_tracks.keys()}

    def batcher(self, shuffled_intervals_iter, batch_size=128):
        '''
        :param shuffled_intervals_iter: pybedtools.BedTool object that provides shuffled intervals
        :param batch_size: batch size
        :return : dictionary of input-batch data pairs
        '''

        shuffled_intervals_batch_iter = batch_iter(shuffled_intervals_iter, batch_size)

        for interval in shuffled_intervals_batch_iter:
            yield {track_name: self.extractors[track_name](interval)
                   for track_name in self.extractors.keys()}

    def validation_data(self, regions):
        return {track_name: self.extractors[track_name](regions)
                for track_name in self.extractors.keys()}


class MultiThreadRunner(object):
    """
    This class manages the  background threads needed to fill
        a queue full of data.
    """
    def __init__(self, train_h5_handle, inputs, outputs, architecture):
        '''
        :param train_h5_handle: hdf5 handle -or pointer-
        :param inputs: a dict object that holds tf placeholders for each input track
        :param outputs: a dict object that holds tf placeholders for each output track
        :param architecture: a nested dict object that holds architecture of the network
        '''

        self.train_h5_handle = train_h5_handle
        self.inputs = inputs
        self.outputs = outputs
        # The actual queue of config.FLAGS.data. The queue contains a vector for input and output data
        try:
            all_shapes = [[self.train_h5_handle.get(track_name).shape[1:3], 1]
                          for track_name in self.inputs.keys()]+\
                        [[self.train_h5_handle.get(track_name).shape[1:3], 1]
                          for track_name in self.outputs.keys()]

        except KeyError:
            print(self.inputs, self.outputs)
            print('Architecture keys: ', architecture.keys())
            raise
        tmp = self.inputs.copy()
        tmp.update(self.outputs)
        self.queue = tf.RandomShuffleQueue(shapes=all_shapes,
                                           dtypes=len(all_shapes)*[tf.float32],
                                           capacity=2000,
                                           names=tmp.keys(),
                                           min_after_dequeue=1000)

        self.enqueue_op = self.queue.enqueue_many(tmp)

    def get_batch(self):
        """
        Return's tensors containing a batch of inputs and outputs
        """
        dequeued_batch = self.queue.dequeue_many()

        return dequeued_batch[:len(self.inputs)], dequeued_batch[len(self.inputs):]

    def _batcher(self, batch_size=128):

        max_len = self.train_h5_handle.values()[0].shape[0]
        current_idx = 0
        current_idx_0 = 0
        while True:
            if (current_idx + batch_size) >= max_len:
                current_idx = current_idx_0 + 13
                current_idx_0 = current_idx
            yield {track_name: self.train_h5_handle.get(track_name)[current_idx:(current_idx + batch_size)]
                   for track_name in self.inputs},\
                  {track_name: self.train_h5_handle.get(track_name)[current_idx:(current_idx + batch_size)]
                   for track_name in self.outputs}
            current_idx += batch_size

    def thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for input_batch_dict, output_batch_dict in self._batcher():
            feed_dict = {track_place_holder: input_batch_dict[track_name]
                         for track_name, track_place_holder in self.inputs.items()}
            feed_dict.update({track_place_holder: output_batch_dict[track_name]
                              for track_name, track_place_holder in self.outputs.items()})
            sess.run(self.enqueue_op, feed_dict=feed_dict)

    def start_threads(self, sess, n_threads=4):
        """ Start background threads to feed queue """
        threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True  # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads

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
