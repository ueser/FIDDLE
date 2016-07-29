##FROM: https://indico.io/blog/tensorflow-data-input-part2-extensions/##

import tensorflow as tf
import time
import threading
import numpy as np
import h5py # alternatively, tables module can be used
from tqdm import tqdm as tq
import config

from dataClass import *
from models import *

# list the inputs and the outputs name
# This should match with the hdf5 records.
inputList = ['DNAseq','NETseq','ChIPseq','MNaseseq','RNAseq']
# inputList = ['DNAseq']
outputList = ['TSSseq']

inputHeights = {'DNAseq':4,'NETseq':2,'ChIPseq':2,'MNaseseq':2,'RNAseq':1}
outputHeights = {'TSSseq': 2}


flags = tf.app.flags
flags.DEFINE_string('runName', 'experiment', 'Running name.')
flags.DEFINE_string('filename', '../data/hdf5datasets/NSMSDSRSCSTS.hdf5', 'Data path.')
flags.DEFINE_boolean('dataTesting', False, 'If true, tests for the data and prints statistics about data for unit testing.')
flags.DEFINE_integer('maxEpoch', 1000, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batchSize', 100, 'Batch size.')
flags.DEFINE_integer('sampleSize', None, 'Sample size.')
flags.DEFINE_float('learningRate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('trainRatio', 0.8, 'Train data ratio')
flags.DEFINE_float('dropout', 0.5, 'Keep probability for training dropout.')
flags.DEFINE_string('dataDir', '../data', 'Directory for input data')
flags.DEFINE_string('resultsDir', '../results', 'Directory for results data')


FLAGS = flags.FLAGS
config.FLAGS=FLAGS

defaultConfDict = { 'width':500,
                    'height':1,
                    'channel':1,
                    'filterWidth':5,
                    'filterHeight':4,
                    'filterAmount':[50,20],
                    'poolSize':3,
                    'poolStride':3}

conf={  'FC1width':1024,
        'combinedFilterAmount':30,
        'combinedFilterWidth':10,
        'outputShape':[500,2],
        'learningRate':FLAGS.learningRate
        }

for inputName in inputList:
    conf[inputName] = defaultConfDict.copy()
    conf[inputName]['height']=inputHeights[inputName]
    conf[inputName]['filterHeight']=inputHeights[inputName]



def train():
    print FLAGS.sampleSize

    # FLAGS.configuration = conf
    print('Getting HDF5 pointer...')
    hdf5Pointer = h5py.File(FLAGS.filename,'r')
    FLAGS.data = multiModalData(hdf5Pointer,inputList,outputList,sampleSize=FLAGS.sampleSize)
    FLAGS.data.splitTest(FLAGS.trainRatio)
    print('Done')

    # Placeholder parameters
    tfmode = tf.placeholder(tf.string)
    keepProb = tf.placeholder(tf.float32)

    # # Make sure that the data are processed in CPU.
    # # Symbolically generates input and output batches
    # print('Symbolic input-output parameters are being created...')
    # with tf.device("/cpu:0"):
    #     custom_runner = CustomRunner()
    #     inputBatch, outputBatch = custom_runner.getBatch()
    #     testInput, testOutput = custom_runner.getTestData()
    # print('Done')
    # #
    #
    print('Setting batcher...')

    batcher = config.FLAGS.data.dataBatcher(chunkSize=FLAGS.batchSize*10)
    print('Done')

    print('Getting test data...')
    testInput, testOutput = config.FLAGS.data.getTestData()
    testFeed = {tfmode:'test',keepProb:1.}
    print testInput[0].shape
    print testOutput[0].shape


    runner = CustomRunner()
    inputPlaceholders = runner.inputs
    outputPlaceholders =  runner.outputs

    testFeed.update({i: d for i, d in zip(inputPlaceholders+outputPlaceholders, testInput+testOutput)})
    print('Done')

    def getTrainFeed(trainInp, trainOut):
        trainFeed = {tfmode:'train',keepProb:FLAGS.dropout}
        trainFeed.update({i: d for i, d in zip(inputPlaceholders+outputPlaceholders, trainInp+trainOut)})
        return trainFeed



    # Concatenate the outputs of the sub models and add a convolutional layer on top of it
    print('Sub models are being created...')
    modelList=[]
    for i,inputName in enumerate(inputList):
        weights,biases = getConvNetParams(inputName,conf)
        modelList.append(makeSubModel(inputPlaceholders[i],conf[inputName],weights,biases))
    combinedLayer = tf.concat(1,modelList)
    print('Done')

    wcComb= tf.Variable(tf.random_normal([len(inputList), conf['combinedFilterWidth'], 1, conf['combinedFilterAmount']]),name='CombinedFilters')
    bcComb = tf.Variable(tf.random_normal([conf['combinedFilterAmount']]),name='CombinedBiasConv')
    combShape = conf['FC1width']- conf['combinedFilterWidth'] +1

    wdComb = tf.Variable(tf.random_normal([combShape*1*conf['combinedFilterAmount'], 1024]),name='CombinedFC')
    bdComb = tf.Variable(tf.random_normal([1024]),name='CombinedBiasFC')
    wdOut1 = tf.Variable(tf.random_normal([1024,conf['outputShape'][0]]),name='outFC1')
    bdOut1 = tf.Variable(tf.random_normal([conf['outputShape'][0]]),name='outBias1')
    # wdOut2 = tf.Variable(tf.random_normal([1024,conf['outputShape'][1]]),name='outFC2')
    # bdOut2 = tf.Variable(tf.random_normal([conf['outputShape'][1]]),name='outBias2')
    combinedLayer = tf.reshape(combinedLayer, shape=[-1,len(inputList), conf['FC1width'], 1])

    convComb = conv2d(combinedLayer,wcComb,bcComb)

    fc2 = tf.reshape(convComb, [-1,combShape*1*conf['combinedFilterAmount']])
    fc2 = tf.add(tf.matmul(fc2, wdComb), bdComb)
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2,keepProb)

    out1 = tf.add(tf.matmul(fc2, wdOut1), bdOut1)
    # out2 = tf.add(tf.matmul(fc2, wdOut2), bdOut2)

    loss1 = tf.nn.softmax_cross_entropy_with_logits(out1,outputPlaceholders[0], name='TSSpredictLoss')
    # loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(out2, outputBatch[1], name='PromoterClassLoss')


    # for monitoring
    meanLoss1 = tf.reduce_mean(loss1)
    # meanLoss2 = tf.reduce_mean(loss2)

    train_op = tf.train.AdamOptimizer(learning_rate=conf['learningRate']).minimize(meanLoss1)

    sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8))
    init = tf.initialize_all_variables()
    sess.run(init)
    print('Session initialized.')

    # print 'debugging'
    # debugSess = tf.Session()
    # debugSess.run(tf.initialize_all_variables())
    # debugInp = tf.placeholder(tf.float32,shape=[None,4,500])
    # tst = tf.reshape(debugInp, shape=[-1, 4, 500, 1])

    # trainInp, trainOut = next(batcher)
    #
    # print(debugSess.run(out1,feed_dict=getTrainFeed(trainInp,trainOut)))
    # print(trainOut[0])
    # print(trainInp[0])

    # # start the tensorflow QueueRunner's
    # tf.train.start_queue_runners(sess=sess)
    # print('Q-runner started.')
    #
    # # start our custom queue runner's threads
    # custom_runner.start_threads(sess)
    # print('Threads started.')
    #
    # # print 'test input shapes'
    # # print [t.shape for t in testInput]
    # #
    # # print 'test output shapes'
    # # print [t.shape for t in testOutput]
    #
    # inpTmp = sess.run(inputBatch,feed_dict=feedDict)
    # outTmp = sess.run(outputBatch,feed_dict=feedDict)
    #
    # print 'train input shapes'
    # print [t.shape for t in inpTmp]
    #
    # print 'train output shapes'
    # print [t.shape for t in outTmp]


    # Launch the graph
    trainFile = open((FLAGS.savePath+"/"+"train.txt"), "w")
    testFile = open((FLAGS.savePath+"/"+"test.txt"), "w")
    iterationNo=0
    epoch=0
    tssGlobalLoss = np.inf
    #
    # # Profile tensorflow
    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()
    # sess.run(res, options=run_options, run_metadata=run_metadata)
    #
    # # Create the Timeline object, and write it to a json
    # tl = timeline.Timeline(run_metadata.step_stats)
    # ctf = tl.generate_chrome_trace_format()
    # with open('timeline.json', 'w') as f:
    #     f.write(ctf)

    print('Pre-train test run:')
    tssLoss = sess.run([meanLoss1 ],feed_dict=testFeed)
    print tssLoss[0]
    print "Testing Epoch " + str(epoch) + " :" + \
       ", TSS Loss= " + str(tssLoss[0])


    for epoch in range(FLAGS.maxEpoch):
        totalTrainTssLoss=0
        for iterationNo in tq(range(FLAGS.data.trainSize/FLAGS.batchSize)):
            trainInp, trainOut = next(batcher)


            _, tssLoss = sess.run([train_op, meanLoss1],feed_dict=getTrainFeed(trainInp,trainOut))
            totalTrainTssLoss+=tssLoss
            if np.isnan(tssLoss):
                print tssLoss
                print trainInp[0]
                print trainOut[0]
                raise ValueError('NaN detected')

        meanTrainTssLoss = totalTrainTssLoss/(iterationNo+0.)
        print "Training Epoch " + str(epoch) + " :" + \
           ", TSS Loss= " + str(meanTrainTssLoss)

        tssLoss = sess.run([meanLoss1],feed_dict=testFeed)

        print "Testing Epoch " + str(epoch) + " :" + \
           ", TSS Loss= " +  str(tssLoss[0])

        trainFile.write("{:.2f}".format(meanTrainTssLoss) + " \n")
        testFile.write("{:.2f}".format(tssLoss[0]) + " \n")

        iterationNo=0
        if tssLoss < tssGlobalLoss:
            tssGlobalLoss = tssLoss
            save_path = saver.save(sess, FLAGS.savePath+"/model.ckpt")
            print("Model saved in file: %s" % FLAGS.savePath)

    trainFile.close()
    testFile.close()


def main(_):
    # if tf.gfile.Exists(FLAGS.summariesDir):
    #     tf.gfile.DeleteRecursively(FLAGS.summariesDir)
    FLAGS.savePath = FLAGS.resultsDir+'/'+FLAGS.runName
    if not tf.gfile.Exists(FLAGS.savePath):
        tf.gfile.MakeDirs(FLAGS.savePath)
    train()


if __name__ == '__main__':
    tf.app.run()
