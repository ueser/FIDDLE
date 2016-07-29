import numpy as np
import tensorflow as tf
import threading

import config



class multiModalData(object):
    """
    This class manages the data i/o for a multi modal multi task model.
    """
    def __init__(self,hdf5Pointer,inputList,outputList=None,sampleSize=None):
        self.inputList = inputList
        self.outputList = outputList
        self.inputs = {}
        self.inputShape = {}
        for inp in inputList:
            self.inputs[inp] = hdf5Pointer.get(inp)
            self.inputShape[inp] = self.inputs[inp].shape[1:]
        self.outputs = {}
        self.outputShape = {}
        for inp in outputList:
            self.outputs[inp] = hdf5Pointer.get(inp)
            # print self.outputs[inp].shape[2:]+(1,)
            self.outputShape[inp] = self.outputs[inp].shape[2:]
        self.sampleSize = sampleSize if sampleSize is not None else self.inputs[inputList[0]][:].shape[0]

    def splitTest(self,ratio=0.8):
        assert self.sampleSize>1, 'Sample size must be greater than 1'
        assert ratio > 1./self.sampleSize, 'Train ratio is unreasonably low. Please consider a reasonable ratio, e.g. 0.8'
        if ratio >(1-1./self.sampleSize):
            print 'Train ratio is unreasonably high. It is set to the default value i.e. 0.8'
            ratio = 0.8
        self.trainSize = np.int(self.sampleSize*ratio)

        self.testSize = self.sampleSize - self.trainSize
        # self.testSize -= self.testSize % config.FLAGS.batchSize

        allIdx = np.arange(0, self.sampleSize)
        np.random.shuffle(allIdx)
        self.trainIdx = allIdx[:self.trainSize].tolist()
        self.testIdx = np.sort(allIdx[self.trainSize:(self.trainSize+self.testSize)]).tolist()



    def dataBatcher(self,chunkSize=10):
        """ An generator object for batching the input-output """
        assert chunkSize>=config.FLAGS.batchSize,'Chunk size must be at least batch size..'
        print 'batch size: ' + str(config.FLAGS.batchSize)
        print 'chunk size:' +  str(chunkSize)
        print 'train size:' +  str(self.trainSize)
        while True:
            # shuffle outputs and inputs
            for chunkIdx in range(0,self.trainSize-chunkSize, chunkSize):
                chunkSliceIdx = np.sort(self.trainIdx[chunkIdx:(chunkIdx + chunkSize)]).tolist()
                inputChunk=[]
                outputChunk=[]
                for key in self.inputList:
                    inputChunk.append(self.inputs[key][chunkSliceIdx])
                for key in self.outputList:
                    outputChunk.append(np.squeeze(self.outputs[key][chunkSliceIdx,0,:]))

                for batchIdx in range(0, chunkSize-config.FLAGS.batchSize, config.FLAGS.batchSize):
                    yield [inp[batchIdx:(batchIdx + config.FLAGS.batchSize)] for inp in inputChunk],\
                     [inp[batchIdx:(batchIdx + config.FLAGS.batchSize)] for inp in outputChunk]

    def getTestData(self):
        print('start...')
        inputBatch = []
        outputBatch = []
        print self.inputList
        for key in self.inputList:
            inputBatch.append(self.inputs[key][self.testIdx])
            print('input done ' + key)
        for key in self.outputList:
            outputBatch.append(np.squeeze(self.outputs[key][self.testIdx,0,:]))

        return inputBatch, outputBatch


class CustomRunner(object):
    """
    This class manages the  background threads needed to fill
        a queue full of data.
    """
    def __init__(self):
        allShapes=[]
        self.inputs = []
        self.outputs = []

        for inputName in config.FLAGS.data.inputList:
            shp =[sh for sh in config.FLAGS.data.inputShape[inputName]]
            self.inputs.append(tf.placeholder(tf.float32, shape=[None]+shp,name=inputName))
            allShapes.append(shp)
        for outputName in config.FLAGS.data.outputList:
            shp =[sh for sh in config.FLAGS.data.outputShape[outputName]]
            self.outputs.append(tf.placeholder(tf.float32, shape=[None]+shp,name=outputName))
            allShapes.append(shp)
        print allShapes
        print self.inputs+self.outputs
        # The actual queue of config.FLAGS.data. The queue contains a vector for input and output data

        self.queue = tf.RandomShuffleQueue(shapes=allShapes,
                                           dtypes=len(allShapes)*[tf.float32],
                                           capacity=200,
                                           min_after_dequeue=100)

        self.enqueue_op = self.queue.enqueue_many(self.inputs+self.outputs)

    def getBatch(self):
        """
        Return's tensors containing a batch of inpust and outputs
        """
        dequedBatch = self.queue.dequeue_many(config.FLAGS.batchSize)
        inputBatch = dequedBatch[:len(self.inputs)]
        outputBatch = dequedBatch[len(self.inputs):]

        return inputBatch, outputBatch


    def thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for inputBatch, outputBatch in config.FLAGS.data.dataBatcher():
            sess.run(self.enqueue_op, feed_dict={i: d for i, d in zip(self.inputs+self.outputs, inputBatch+outputBatch)})

    def start_threads(self, sess,n_threads=1):
        """ Start background threads to feed queue """
        threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads
