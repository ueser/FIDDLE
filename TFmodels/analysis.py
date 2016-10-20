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
from models import *
from auxilary import *
import pandas as pd
# from matplotlib import pylab as pl



flags = tf.app.flags
flags.DEFINE_string('runName', 'experiment', 'Running name.')
flags.DEFINE_string('filename', '../data/hdf5datasets/NSMSDSRSCSTSRI.hdf5', 'Data path.')
flags.DEFINE_string('inputs', 'NS_MS_DS_RS_CS', 'Input symbols [NS: NETseq, MS:MNaseseq, RS:RNAseq, DS:DNAseq, CS:ChIPseq, e.g. NS_RS_MS]')
flags.DEFINE_string('outputs', 'TS', 'Output symbols [TS: TSSseq, e.g. TS]')
flags.DEFINE_boolean('restore', False, 'If true, restores models from the ../results/XXtrained/')
flags.DEFINE_boolean('suffnec', False, 'If true, calculates sufficiency and necessity costs')
flags.DEFINE_boolean('predict', False, 'If true, predicts the output')
flags.DEFINE_string('dataDir', '../data', 'Directory for input data')
flags.DEFINE_string('resultsDir', '../results', 'Directory for results data')
flags.DEFINE_integer('trainSize', None, 'Train size.')
flags.DEFINE_integer('testSize', None, 'Test size.')

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

    network_architecture, outputList = get_network_architecture()

    print('Getting HDF5 pointer...')
    hdf5Pointer = h5py.File(FLAGS.filename,'r')
    FLAGS.data = multiModalData(hdf5Pointer,network_architecture.keys(),outputList)
    print('Done')

    # FLAGS.testSize = FLAGS.data.sampleSize
    FLAGS.data.splitTest()

    print('Setting batcher...')
    with tf.device("/cpu:0"):
        batcher = FLAGS.data.dataBatcher(chunkSize=FLAGS.data.sampleSize)
        print('Getting test data...')
        testInput, testOutput = FLAGS.data.getTestData()
    print('Done')

    model = NNscaffold(network_architecture)

    model.load(FLAGS.savePath)
    #
    # print([v.name for v in tf.trainable_variables()])
    # for v in tf.trainable_variables():
    #     if v.name =='DNAseq/conv1/weights:0':
    #         Wtest=v.copy()
    #         break
    # refs = model.get_reference(refInput)

    if FLAGS.suffnec:

        print('Calculating the scores')
        suffDict,necDict = model.suffnec(testInput,testOutput)
        print('Saving...')
        pd.DataFrame(suffDict).to_csv(FLAGS.savePath+"/"+"SufficiencyCosts.csv")
        pd.DataFrame(necDict).to_csv(FLAGS.savePath+"/"+"NecessityCosts.csv")
        print('Done...')

    if FLAGS.predict:
        predictions = model.predict(testInput)
        infodata = hdf5Pointer.get('info')[FLAGS.data.testIdx]
        f = h5py.File(FLAGS.savePath+"/"+"predictions.hdf5",'w')
        pred = f.create_dataset('predictions',predictions.shape)
        info  = f.create_dataset('info',infodata.shape)

        pred[:] = predictions
        info[:] = infodata

        f.close()
        print('Done...')

if __name__ == '__main__':
    tf.app.run()
