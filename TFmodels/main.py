from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
#import threading
import numpy as np
import h5py # alternatively, tables module can be used
from tqdm import tqdm as tq
import cPickle as pickle
from dataClass import *
from models import *
from auxilary import *
import config
import copy
# from matplotlib import pylab as pl


flags = tf.app.flags
flags.DEFINE_string('runName', 'experiment', 'Running name.')
flags.DEFINE_string('filename', '../data/hdf5datasets/NSMSDSRSCSTSRI.hdf5', 'Data path.')
flags.DEFINE_string('inputs', 'NS_MS_DS_RS_CS', 'Input symbols [NS: NETseq, MS:MNaseseq, RS:RNAseq, DS:DNAseq, CS:ChIPseq, e.g. NS_RS_MS]')
flags.DEFINE_string('outputs', 'TS', 'Output symbols [TS: TSSseq, e.g. TS]')
flags.DEFINE_boolean('dataTesting', False, 'If true, tests for the data and prints statistics about data for unit testing.')
flags.DEFINE_boolean('restore', False, 'If true, restores models from the ../results/XXtrained/')
flags.DEFINE_integer('maxEpoch', 1000, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batchSize', 100, 'Batch size.')
flags.DEFINE_integer('filterWidth', 10, 'Filter width for convolutional network')
flags.DEFINE_integer('sampleSize', None, 'Sample size.')
flags.DEFINE_integer('testSize', None, 'Test size.')
flags.DEFINE_integer('trainSize', None, 'Train size.')
flags.DEFINE_integer('chunkSize', 1000, 'Chunk size.')
flags.DEFINE_float('learningRate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('trainRatio', 0.8, 'Train data ratio')
flags.DEFINE_float('dropout', 0.5, 'Keep probability for training dropout.')
flags.DEFINE_string('dataDir', '../data', 'Directory for input data')
flags.DEFINE_string('resultsDir', '../results', 'Directory for results data')


FLAGS = flags.FLAGS
config.FLAGS = FLAGS



def main(_):
    FLAGS.savePath = FLAGS.resultsDir+'/'+FLAGS.runName
    if not tf.gfile.Exists(FLAGS.savePath):
        tf.gfile.MakeDirs(FLAGS.savePath)

    network_architecture, outputList = get_network_architecture()

    pickle.dump(network_architecture,open(FLAGS.savePath+"/network_architecture.pkl",'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    # FLAGS.configuration = conf
    print('Getting HDF5 pointer...')
    hdf5Pointer = h5py.File(FLAGS.filename,'r')
    FLAGS.data = multiModalData(hdf5Pointer,network_architecture.keys(),outputList,sampleSize=FLAGS.sampleSize)
    FLAGS.data.splitTest(FLAGS.trainRatio)
    print('Done')



    print('Setting batcher...')
    with tf.device("/cpu:0"):
        batcher = FLAGS.data.dataBatcher(chunkSize=FLAGS.chunkSize)
        print('Getting test data...')
        print(FLAGS.testSize)
        testInput, testOutput = FLAGS.data.getTestData()
    print('Done')

    model = NNscaffold(network_architecture,
                 learning_rate=FLAGS.learningRate)

    if FLAGS.restore:
        model.load()
    else:
        model.initialize()
    model.create_monitor_variables(FLAGS.savePath)

    # Launch the graph

    saver = tf.train.Saver()
    with open((FLAGS.savePath+"/"+"train.txt"), "w") as trainFile:
        trainFile.write('Loss\tAccuracy\n')

    with open((FLAGS.savePath+"/"+"test.txt"), "w") as testFile:
        testFile.write('Loss\tAccuracy\n')


    print('Pre-train test run:')
    tssLoss,acc = model.test(testInput,testOutput,accuracy=True)
    print("Pre-train test loss: " + str(tssLoss))
    print("Pre-train test accuracy (%): " + str(100.*acc/FLAGS.data.testSize))

    totIteration = int(FLAGS.data.trainSize/FLAGS.batchSize)
    globalMinLoss = np.inf
    step=0
    totalTrainLoss =0
    totalTrainAcc = 0
    iterationNo = 0
    nn=0

    trainInput, trainOutput = next(batcher)

    for it in xrange(FLAGS.maxEpoch*totIteration):

        for iterationNo in tq(range(10)):
            trainInp, trainOut = next(batcher)
            loss,acc = model.train(trainInp,trainOut,accuracy=True)
            totalTrainLoss +=loss
            totalTrainAcc +=acc
            step+=1
            if np.isnan(tssLoss):
                print(loss)
                print(trainInp[0])
                print(trainOut[0])
                raise ValueError('NaN detected')


        meanTrainLoss = totalTrainLoss/(iterationNo+0.)
        meanTrainAcc = totalTrainAcc/(iterationNo+0.)
        testLoss,acc = model.test(testInput,testOutput,accuracy=True)
        print("Training step " + str(it) + " : \n Train loss: " + str(meanTrainLoss) +\
        "\n Train accuracy(%): "+ str(100.*meanTrainAcc/FLAGS.batchSize))
        print("Test loss: " +  str(testLoss)+\
        "\n Test accuracy(%): "+ str(100.*acc/FLAGS.data.testSize))

        with open((FLAGS.savePath+"/"+"train.txt"), "a") as trainFile:
            trainFile.write(str(meanTrainLoss)+'\t'+str(100.*meanTrainAcc/FLAGS.batchSize)+'\n')
        with open((FLAGS.savePath+"/"+"test.txt"), "a") as testFile:
            testFile.write(str(testLoss)+'\t'+str(100.*acc/FLAGS.data.testSize)+'\n')

        model.summarize(step)

        totalTrainLoss=0
        totalTrainAcc =0

        if testLoss < globalMinLoss:
            globalMinLoss = testLoss
            save_path = saver.save(model.sess, FLAGS.savePath+"/model.ckpt")
            print("Model saved in file: %s" % FLAGS.savePath)


    trainFile.close()
    testFile.close()
    model.sess.close()





if __name__ == '__main__':
    tf.app.run()
