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
flags.DEFINE_boolean('full_data', True, 'If true, uses full data')
flags.DEFINE_boolean('suffnec', False, 'If true, calculates sufficiency and necessity costs')
flags.DEFINE_boolean('predict', False, 'If true, predicts the output')
flags.DEFINE_string('suffix', '', 'Suffix to save')
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



    model = NNscaffold(network_architecture,outputMode=FLAGS.outputs)
    model.load(FLAGS.savePath)
    # return 0
    #
    # print([v.name for v in tf.trainable_variables()])
    # for v in tf.trainable_variables():
    #     if v.name =='DNAseq/conv1/weights:0':
    #         Wtest=v.copy()
    #         break
    # refs = model.get_reference(refInput)

    print('Getting HDF5 pointer...')
    hdf5Pointer = h5py.File(FLAGS.filename,'r')
    if FLAGS.suffnec:

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

        print('Calculating the scores')
        suffDict,necDict = model.suffnec(testInput, testOutput)
        print('Saving...')
        pd.DataFrame(suffDict).to_csv(FLAGS.savePath+"/"+"SufficiencyCosts.csv")
        pd.DataFrame(necDict).to_csv(FLAGS.savePath+"/"+"NecessityCosts.csv")
        print('Done...')

    if FLAGS.predict:

        FLAGS.data = multiModalData(hdf5Pointer,network_architecture.keys(),[])
        print('Done')

        if FLAGS.full_data:
            print('Getting all data...')
            with tf.device("/cpu:0"):
                testInput,testOutput = FLAGS.data.getAllData()
                infodata = hdf5Pointer.get('info')[:]
        else:
            # FLAGS.testSize = FLAGS.data.sampleSize
            FLAGS.data.splitTest()

            print('Setting batcher...')
            with tf.device("/cpu:0"):
                batcher = FLAGS.data.dataBatcher(chunkSize=FLAGS.data.sampleSize)
                print('Getting test data...')
                testInput, testOutput = FLAGS.data.getTestData()
                infodata = hdf5Pointer.get('info')[FLAGS.data.testIdx]
        print('Done')
        predictions = model.predict({ky:testInput[ky][:2] for ky in testInput.keys()})
        print((testInput.values()[0].shape[0],)+predictions.shape[1:])
        f = h5py.File(FLAGS.savePath+"/"+"predictions_"+FLAGS.suffix+".hdf5",'w')
        pred = f.create_dataset('predictions',(testInput.values()[0].shape[0],)+predictions.shape[1:])
        info  = f.create_dataset('info',infodata.shape)

        for batchIdx in tq(range(0,testInput.values()[0].shape[0],5000)):
            if (batchIdx+5000)<testInput.values()[0].shape[0]:
                prd = model.predict({key:testInput[key][batchIdx:(batchIdx+5000)] for key in testInput.keys()})
                pred[batchIdx:(batchIdx+5000),:] = prd
            else:
                predictions = model.predict({key:testInput[key][batchIdx:testInput.values()[0].shape[0]] for key in testInput.keys()})
                pred[batchIdx:testInput.values()[0].shape[0],:] = predictions

        info[:] = infodata
        f.close()
        print('Done...')

if __name__ == '__main__':
    tf.app.run()
