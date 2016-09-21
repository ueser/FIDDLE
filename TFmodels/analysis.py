from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import threading
import numpy as np
import h5py # alternatively, tables module can be used
from tqdm import tqdm as tq
import cPickle as pickle
from dataClass import *
from auxilary import *
# from matplotlib import pylab as pl



flags = tf.app.flags
flags.DEFINE_string('runName', 'experiment', 'Running name.')
flags.DEFINE_string('filename', '../data/hdf5datasets/NSMSDSRSCSTS.hdf5', 'Data path.')
flags.DEFINE_string('inputs', 'NS_MS_DS_RS_CS', 'Input symbols [NS: NETseq, MS:MNaseseq, RS:RNAseq, DS:DNAseq, CS:ChIPseq, e.g. NS_RS_MS]')
flags.DEFINE_string('outputs', 'TS', 'Output symbols [TS: TSSseq, e.g. TS]')
flags.DEFINE_boolean('restore', True, 'If true, restores models from the ../results/XXtrained/')
flags.DEFINE_string('dataDir', '../data', 'Directory for input data')
flags.DEFINE_string('resultsDir', '../results', 'Directory for results data')


FLAGS = flags.FLAGS
config.FLAGS=FLAGS




def main(_):
    '''
    This code imports NNscaffold class from models.py module for training, testing etc.

    Usage with defaults:
    python analysis.py (options)

    Train by initializing from pre-trained models:
    python analysis.py --restore True (options)
    This option restores models from the ../results/XXtrained/ where XX is the input abbreviations.

    Currently, input names are abbreviated [NS: NETseq, MS:MNaseseq, RS:RNAseq, DS:DNAseq, CS:ChIPseq, e.g. NS_RS_MS]
    To use different inputs and outputs, change the dictionary in the function get_network_architecture().

    '''
    FLAGS.savePath = FLAGS.resultsDir+'/'+FLAGS.runName
    if not tf.gfile.Exists(FLAGS.savePath):
        tf.gfile.MakeDirs(FLAGS.savePath)

    network_architecture, restore_dirs, outputList = get_network_architecture()

    print('Getting HDF5 pointer...')
    hdf5Pointer = h5py.File(FLAGS.filename,'r')
    FLAGS.data = multiModalData(hdf5Pointer,network_architecture.keys(),outputList)
    print('Done')

    FLAGS.testSize = FLAGS.data.sampleSize
    FLAGS.data.splitTest()

    print('Setting batcher...')
    with tf.device("/cpu:0"):
        batcher = FLAGS.data.dataBatcher(chunkSize=FLAGS.data.sampleSize)
        print('Getting test data...')
        testInput, testOutput = FLAGS.data.getTestData()
    print('Done')

    model = NNscaffold(network_architecture,
                 learning_rate=FLAGS.learningRate,
                 batch_size=FLAGS.batchSize)

    model.initialize(restore_dirs)


    print('Calculating the scores')
    suffDict,necDict = model.suffnec(testInput,testOutput)

    print('Saving...')
    pd.DataFrame(suffDict).to_csv(FLAGS.savePath+"/"+"SufficiencyCosts.csv")
    pd.DataFrame(necDict).to_csv(FLAGS.savePath+"/"+"NecessityCosts.csv")
    print('Done...')
if __name__ == '__main__':
    tf.app.run()
