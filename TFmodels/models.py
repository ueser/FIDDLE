from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import tflearn

def multi_softmax(target, axis=1, name=None):
    """
    Takes a tensor and returns the softmax applied to a particular axis

    """
    with tf.name_scope(name):
        mx = tf.reduce_max(target, axis, keep_dims=True)
        Q = tf.exp(target-mx)
        Z = tf.reduce_sum(Q, axis, keep_dims=True)
    return Q/Z


class NNscaffold(object):
    """
    Scaffold class to combine different NN modules at their final layers
    """
    def __init__(self, architecture,
                 learning_rate=0.001,outputMode='TSSseq'):
        """
        Initiates a scaffold network with default values
        Inputs:
            architecture: A nested dictionary where the highest level
            keywords are the input names. e.g. {
                                                'NETseq':{
                                                            'inputShape':[2,500,1],
                                                            'outputWidth:500',
                                                            'numberOfFilters':[80,80]},
                                                'DNAseq':{
                                                            'inputShape':[4,500,1],
                                                            'outputWidth:500',
                                                            'numberOfFilters':[50,20]}}

        """
        self.architecture = architecture
        self.learning_rate = learning_rate

        self.inputs={}
        self.representations =list()
        # tf Graph input
        for key in architecture.keys():
            self.inputs[key] = tf.placeholder(tf.float32, [None] + architecture[key]["inputShape"],name=key)
            self._add_representations(inp,architecture[inp])

        self.output = tf.placeholder(tf.float32, [None]+ architecture[outputMode]["outputShape"],name='output')


        self.dropout = tf.placeholder(tf.float32)
        self.keep_prob_input = tf.placeholder(tf.float32)
        self.inp_size = tf.placeholder(tf.int32)
        self.outputMode = outputMode

        self._combine_representations()

        self._encapsulate_models()


        # Define loss function based variational upper-bound and
        # corresponding optimizer
        self._create_loss_optimizer()

    def _add_representations(self,key,new_representation):
        self.representations[key]=new_representation

    def _combine_representations(self):
        self.combined_representation = tf.concat(1,self.representations)


    def initialize(self,restore_dirs=None):
        """
        Initialize the scaffold model either from saved checkpoints (pre-trained)
        or from scratch
        """

        # Launch the session
        self.sess = tf.Session()
        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()
        self.sess.run(init)
        print('Session initialized.')
        # if restore_dirs is not None:
        #     for key in self.architecture.keys():
        #         saver = tf.train.Saver([v for v in tf.trainable_variables() if key in v.name])
        #         saver.restore(self.sess,restore_dirs[key]+'model.ckpt')
        #         print('Session restored for '+key)
    def load(self,modelPath):
        """
            loads the pretrained model from the specified path
        """
        # Launch the session
        self.sess = tf.Session()
        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()
        self.sess.run(init)
        print([v.name for v in tf.trainable_variables()])
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(self.sess,modelPath+'/model.ckpt')
        print('Model loaded with the pretrained parameters')

    def _encapsulate_models(self):

        with tf.variable_scope('scaffold'):
            self.net = tf.reshape(self.combined_representation, shape=[-1,len(self.representations), self.architecture["FCwidth"], 1])
            self.net = tf.nn.dropout(tf.identity(self.net),self.keep_prob_input,noise_shape=[self.inp_size,len(self.net),1,1])

            self.net = tflearn.conv_2d(self.net, self.architecture['numberOfFilters'],
                                        self.architecture['filterSize'],
                                        activation='relu', regularizer="L2",padding='valid',name='conv_combined')
            self.net = tflearn.layers.conv.avg_pool_1d(self.net, self.architecture['pool_size'],
                                                strides=self.architecture['pool_stride'],
                                                padding='valid', name='AvgPool_combined')
            self.net = tflearn.layers.normalization.batch_normalization(self.net,name='batch_norm_2')
            self.net = tflearn.flatten(self.net)
            if self.outputMode != 'DNAseq':
                self.net = tf.nn.dropout(self.net, self.dropout, scope='dropout3')
                self.representation = tflearn.fully_connected(self.net,
                                                                self.architecture[self.outputMode]['outputShape'],name='representation')
                self.net = tf.nn.softmax(self.representation)
            else:
                self.representation = tflearn.fully_connected(self.net,
                                                                self.architecture[self.outputMode]['outputShape'][0]*\
                                                                self.architecture[self.outputMode]['outputShape'][1],name='representation')
                self. = tf.reshape(self.net,[-1,4, self.architecture['outputWidth'])
                self.net = multi_softmax(self.net,axis=1,name='multiSoftmax')


    def _create_network(self,key):
        with tf.variable_scope(key):
            net = tflearn.input_data(shape=[None] + self.architecture[key]['inputShape'], name='input')
            net = tflearn.conv_2d(net, self.architecture[key]['numberOfFilters'][0],
                                     self.architecture[key]['filterSize'][0],
                                     activation='relu', regularizer="L2",padding='valid',name='conv_1')
            net = tflearn.layers.conv.avg_pool_1d(net, self.architecture[key]['pool_size'],
                                             strides=self.architecture[key]['pool_stride'],
                                             padding='valid', name='AvgPool_1')
            net = tflearn.layers.normalization.batch_normalization(net,name='batch_norm_1')
            net = tflearn.conv_2d(net, self.architecture[key]['numberOfFilters'][1],
                                     self.architecture[key]['filterSize'][1],
                                     activation='relu', regularizer="L2",padding='valid',name='conv_2')
            net = tflearn.layers.conv.avg_pool_1d(net, self.architecture[key]['pool_size'],
                                             strides=self.architecture[key]['pool_stride'],
                                             padding='valid', name='AvgPool_2')
            net = tflearn.layers.normalization.batch_normalization(net,name='batch_norm_2')

        return net

    def _create_loss_optimizer(self):

        if self.outputMode != 'DNAseq':
            self.loss = tf.reduce_sum(tf.mul(self.output+1e-10,tf.sub(tf.log(self.output+1e-10),tf.log(self.net+1e-10))),1)
            width =  self.architecture.values()[0]["outputWidth"][0]
            target = tf.floor((10.*tf.cast(tf.argmax(self.output,dimension=1),tf.float32))/np.float(width))
            pred = tf.floor((10.*tf.cast(tf.argmax(self.net,dimension=1),tf.float32))/np.float(width))
            self.accuracy = tf.reduce_sum(tf.cast(tf.equal(pred,target),tf.int32))


        else:
            self.loss = tf.reduce_sum(tf.mul(self.output+1e-10,tf.sub(tf.log(self.output+1e-10),tf.log(self.net+1e-10))),[1,2])
            target = tf.argmax(self.output,dimension=1)
            pred = tf.argmax(self.net,dimension=1)
            self.accuracy = tf.reduce_sum(tf.cast(tf.equal(pred,target),tf.float32))/target.get_shape()[1].value


        self.cost = tf.reduce_mean(self.loss)   # average over batch


        self.global_step = tf.Variable(0, name='globalStep', trainable=False)

        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost,global_step = self.global_step)

    def train(self, trainInp,trainOut,accuracy=None,inp_dropout=0.9):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        train_feed = {self.output:trainOut.values()[0], self.dropout:self.architecture.values()[0]["dropout"],\
        self.keep_prob_input:inp_dropout,self.inp_size:trainOut.values()[0].shape[0]}
        train_feed.update({self.inputs[key]: trainInp[key] for key in self.architecture.keys()})

        if accuracy is not None:
            _ , cost,accuracy = self.sess.run((self.optimizer, self.cost, self.accuracy), feed_dict=train_feed)
        else:
            _ , cost = self.sess.run((self.optimizer, self.cost), feed_dict=train_feed)
            accuracy = None
        return cost,accuracy

    def test(self,testInp,testOut,accuracy=None):
        """Test model based on mini-batch of input data.

        Return cost of test.
        """

        if not hasattr(self,'test_feed'):
            self.test_feed = {self.output:testOut.values()[0], self.dropout:1,self.keep_prob_input:1.,self.inp_size:testOut.values()[0].shape[0]}
            self.test_feed.update({self.inputs[key]: testInp[key] for key in self.architecture.keys()})
        if accuracy is not None:
            cost,accuracy = self.sess.run((self.cost, self.accuracy), feed_dict=self.test_feed)
        else:
            cost = self.sess.run( self.cost, feed_dict=self.test_feed)
            accuracy = None

        return cost,accuracy

    def suffnec(self,trainInp,trainOut):
        """Calculates the sufficiency and necessity score of an input dataset by setting the
        rest of the inputs to their average values and the input of interest to its average value, respectively.

        Returns cost dictionary.
        """

        print('Calculating sufficiency...')
        suffDict ={}


        self.train_feed = {self.output:trainOut.values()[0], self.dropout:1,self.keep_prob_input:1.,self.inp_size:trainOut.values()[0].shape[0]}
        for key in self.architecture.keys():
            self.train_feed.update({self.inputs[key]: np.tile(trainInp[key].mean(axis=0),(trainInp[key].shape[0],1,1,1))})
        suffDict['NoneActive'] = self.sess.run(self.loss, feed_dict=self.train_feed)

        for inputName in self.architecture.keys():
            print('...'+inputName)
            self.train_feed = {self.output:trainOut.values()[0], self.dropout:1,self.keep_prob_input:1.,self.inp_size:trainOut.values()[0].shape[0]}
            for key in self.architecture.keys():
                if key==inputName:
                    self.train_feed.update({self.inputs[key]: trainInp[key]})
                else:
                    self.train_feed.update({self.inputs[key]: np.tile(trainInp[key].mean(axis=0),(trainInp[key].shape[0],1,1,1))})

            suffDict[inputName] = self.sess.run(self.loss, feed_dict=self.train_feed)

        print('Calculating necessity...')
        necDict ={}
        for inputName in self.architecture.keys():
            print('...'+inputName)
            self.train_feed = {self.output:trainOut.values()[0], self.dropout:1,self.keep_prob_input:1.,self.inp_size:trainOut.values()[0].shape[0]}
            for key in self.architecture.keys():
                if key is not inputName:
                    self.train_feed.update({self.inputs[key]: trainInp[key]})
                else:
                    self.train_feed.update({self.inputs[key]: np.tile(trainInp[key].mean(axis=0),(trainInp[key].shape[0],1,1,1))})
            necDict[inputName] = self.sess.run(self.loss, feed_dict=self.train_feed)

        self.train_feed = {self.output:trainOut.values()[0], self.dropout:1,self.keep_prob_input:1.,self.inp_size:trainOut.values()[0].shape[0]}
        for key in self.architecture.keys():
            self.train_feed.update({self.inputs[key]: trainInp[key]})
        necDict['AllActive'] = self.sess.run(self.loss, feed_dict=self.train_feed)




        return suffDict, necDict

    def getWeight(self,layerName):
        return self.sess.run([v for v in tf.trainable_variables() if v.name == layerName+'\weights:0'][0])


    def predict(self,testInp):
        """Return the result of a flow based on mini-batch of input data.

        """
        self.test_feed = {self.dropout:1, self.keep_prob_input:1.,self.keep_prob_input:1.,self.inp_size:testInp.values()[0].shape[0]}
        self.test_feed.update({self.inputs[key]: testInp[key] for key in self.architecture.keys()})
        return self.sess.run( self.net, feed_dict=self.test_feed)

    def summarize(self,step):
        summaryStr = self.sess.run(self.summary_op, feed_dict=self.test_feed)
        self.summaryWriter.add_summary(summaryStr, step)
        self.summaryWriter.flush()

    def create_monitor_variables(self,savePath):
        # for monitoring
        tf.scalar_summary('KL divergence', self.cost)
        tf.scalar_summary('Accuracy', self.accuracy)
        self.summary_op = tf.merge_all_summaries()
        self.summaryWriter = tf.train.SummaryWriter(savePath, self.sess.graph)
