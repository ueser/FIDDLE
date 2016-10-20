from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim


class NNscaffold(object):
    """
    Scaffold class to combine different NN modules at their final layers
    """
    def __init__(self, network_architecture,
                 learning_rate=0.001):
        """
        Initiates a scaffold network with default values
        Inputs:
            network_architecture: A nested dictionary where the highest level
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
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate

        self.inputs={}
        # tf Graph input
        for key in network_architecture.keys():
            self.inputs[key] = tf.placeholder(tf.float32, [None] + network_architecture[key]["inputShape"],name=key)
            print(network_architecture[key])

        self.output = tf.placeholder(tf.float32, [None]+ network_architecture[key]["outputWidth"],name='output')
        self.dropout = tf.placeholder(tf.float32)
        self.keep_prob_input = tf.placeholder(tf.float32)
        self.inp_size = tf.placeholder(tf.int32)

        self.net =list()

        self._encapsulate_models()


        # Define loss function based variational upper-bound and
        # corresponding optimizer
        self._create_loss_optimizer()


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
        #     for key in self.network_architecture.keys():
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
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(self.sess,modelPath+'/model.ckpt')
        print('Model loaded with the pretrained parameters')

    def _encapsulate_models(self):
        # Create Convolutional network
        for key in self.network_architecture.keys():
            with tf.variable_scope(key):
                self.net.append(self._create_network(key))

        combined_layer = tf.concat(1,self.net)

        with slim.arg_scope([slim.conv2d],
                     activation_fn=tf.nn.relu,
                     weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                     weights_regularizer=slim.l2_regularizer(0.0005),padding='VALID',
                     stride=1):
            if len(self.net)>1:
                combined_layer = tf.reshape(combined_layer, shape=[-1,len(self.net), self.network_architecture[key]["FCwidth"], 1])
                combined_layer = tf.nn.dropout(tf.identity(combined_layer),self.keep_prob_input,noise_shape=[self.inp_size,len(self.net),1,1])
                self.net = slim.conv2d(combined_layer,
                                   40,
                                   [len(self.net),10],
                                   scope='conv1')
                self.net = slim.avg_pool2d(self.net, self.network_architecture[key]["pool_size"],
                                        stride=self.network_architecture[key]["pool_stride"],
                                        scope='pool2')
                self.net = slim.batch_norm(self.net,activation_fn=None)
                self.net = slim.flatten(self.net)
            else:
                self.net = combined_layer

            self.net = slim.dropout(self.net, self.dropout, scope='dropout3')
            self.net = slim.fully_connected(self.net,  self.network_architecture[key]["outputWidth"][0], activation_fn=None, scope='out')
            self.net = tf.nn.softmax(self.net)

    def _create_network(self,key):

         with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005),padding='VALID',
                      stride=1):
            net={}
            net[key+'_conv1'] = slim.conv2d(  self.inputs[key],
                                self.network_architecture[key]['numberOfFilters'][0],
                                self.network_architecture[key]['filterSize'][0],
                                scope='conv1')
            net[key+'_pool1'] = slim.avg_pool2d(net[key+'_conv1'],
                                    self.network_architecture[key]["pool_size"],
                                    stride=self.network_architecture[key]["pool_stride"],
                                    scope='pool1')
            net[key+'_batch_norm1'] = slim.batch_norm(net[key+'_pool1'],activation_fn=None,scope='batch_norm1')
            net[key+'_conv2'] = slim.conv2d(net[key+'_batch_norm1'],
                                self.network_architecture[key]['numberOfFilters'][1],
                                self.network_architecture[key]['filterSize'][1],
                                scope='conv2')
            net[key+'_pool2'] = slim.avg_pool2d(net[key+'_conv2'], self.network_architecture[key]["pool_size"],
                                    stride=self.network_architecture[key]["pool_stride"],
                                    scope='pool2')
            net = slim.batch_norm(net[key+'_pool2'],activation_fn=None)
            net = slim.flatten(net)
            net = slim.dropout(net, self.dropout, scope='dropout2')
            net = slim.fully_connected(net,  self.network_architecture[key]["FCwidth"], scope='fc3')

            return net

    def _create_loss_optimizer(self):

        self.loss = tf.reduce_sum(tf.mul(self.output+1e-10,tf.sub(tf.log(self.output+1e-10),tf.log(self.net+1e-10))),1)

        self.cost = tf.reduce_mean(self.loss)   # average over batch
        width =  self.network_architecture.values()[0]["outputWidth"][0]

        target = tf.floor((10.*tf.cast(tf.argmax(self.output,dimension=1),tf.float32))/np.float(width))
        pred = tf.floor((10.*tf.cast(tf.argmax(self.net,dimension=1),tf.float32))/np.float(width))

        self.accuracy = tf.reduce_sum(tf.cast(tf.equal(pred,target),tf.int32))

        self.global_step = tf.Variable(0, name='globalStep', trainable=False)

        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost,global_step = self.global_step)

    def train(self, trainInp,trainOut,accuracy=None):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        train_feed = {self.output:trainOut.values()[0], self.dropout:self.network_architecture.values()[0]["dropout"],\
        self.keep_prob_input:self.network_architecture.values()[0]["input_dropout"],self.inp_size:trainOut.values()[0].shape[0]}
        train_feed.update({self.inputs[key]: trainInp[key] for key in self.network_architecture.keys()})

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
            self.test_feed.update({self.inputs[key]: testInp[key] for key in self.network_architecture.keys()})
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
        for key in self.network_architecture.keys():
            self.train_feed.update({self.inputs[key]: np.tile(trainInp[key].mean(axis=0),(trainInp[key].shape[0],1,1,1))})
        suffDict['NoneActive'] = self.sess.run(self.loss, feed_dict=self.train_feed)

        for inputName in self.network_architecture.keys():
            print('...'+inputName)
            self.train_feed = {self.output:trainOut.values()[0], self.dropout:1,self.keep_prob_input:1.,self.inp_size:trainOut.values()[0].shape[0]}
            for key in self.network_architecture.keys():
                if key==inputName:
                    self.train_feed.update({self.inputs[key]: trainInp[key]})
                else:
                    self.train_feed.update({self.inputs[key]: np.tile(trainInp[key].mean(axis=0),(trainInp[key].shape[0],1,1,1))})

            suffDict[inputName] = self.sess.run(self.loss, feed_dict=self.train_feed)

        print('Calculating necessity...')
        necDict ={}
        for inputName in self.network_architecture.keys():
            print('...'+inputName)
            self.train_feed = {self.output:trainOut.values()[0], self.dropout:1,self.keep_prob_input:1.,self.inp_size:trainOut.values()[0].shape[0]}
            for key in self.network_architecture.keys():
                if key is not inputName:
                    self.train_feed.update({self.inputs[key]: trainInp[key]})
                else:
                    self.train_feed.update({self.inputs[key]: np.tile(trainInp[key].mean(axis=0),(trainInp[key].shape[0],1,1,1))})
            necDict[inputName] = self.sess.run(self.loss, feed_dict=self.train_feed)

        self.train_feed = {self.output:trainOut.values()[0], self.dropout:1,self.keep_prob_input:1.,self.inp_size:trainOut.values()[0].shape[0]}
        for key in self.network_architecture.keys():
            self.train_feed.update({self.inputs[key]: trainInp[key]})
        necDict['AllActive'] = self.sess.run(self.loss, feed_dict=self.train_feed)




        return suffDict, necDict

    def getWeight(self,layerName):
        return self.sess.run([v for v in tf.trainable_variables() if v.name == layerName+'\weights:0'][0])


    def predict(self,testInp):
        """Return the result of a flow based on mini-batch of input data.

        """
        if not hasattr(self,'test_feed'):
            self.test_feed = {self.dropout:1, self.keep_prob_input:1.,self.keep_prob_input:1.,self.inp_size:testInp.values()[0].shape[0]}
            self.test_feed.update({self.inputs[key]: testInp[key] for key in self.network_architecture.keys()})

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
