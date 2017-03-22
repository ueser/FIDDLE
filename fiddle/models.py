from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pdb, traceback, sys # EDIT
import tensorflow as tf
import numpy as np
import tflearn

#######################
# Auxiliary functions #
#######################


def multi_softmax(target, axis=1, name=None):
    """Takes a tensor and returns the softmax applied to a particular axis
    """
    with tf.name_scope(name):
        mx = tf.reduce_max(target, axis, keep_dims=True)
        Q = tf.exp(target-mx)
        Z = tf.reduce_sum(Q, axis, keep_dims=True)
    return Q/Z


def KL_divergence(P_, Q_):
    """Takes Tensorflow tensors and returns the KL divergence for each row i.e. D(P_, Q_)
    """
    return tf.reduce_sum(tf.mul(P_+1e-16, tf.sub(tf.log(P_+1e-16), tf.log(Q_ + 1e-16))), 1)


def transform_track(track_data_placeholder, option='pdf'):
    """Converts input placeholder tensor to probability distribution function
        :param track_data_placeholder:
        :param option: pdf: converts every entry to a pdf
        :              categorical: discretisizes the continuous input (To be implemented)
        :              standardize: zero mean, unit variance
        :return:
    """
    if option == 'pdf':
        output_tensor = tf.reshape(track_data_placeholder,
                        [-1, (track_data_placeholder.get_shape()[1]*track_data_placeholder.get_shape()[2]).value]) + 1e-16
        output_tensor = tf.div(output_tensor, tf.reduce_sum(output_tensor, 1, keep_dims=True))
    # NOT completed yet
    elif option == 'standardize':
        raise NotImplementedError
        from scipy import stats
        output_tensor = stats.zscore(output_tensor, axis=1)
    return output_tensor


#################
# Model Classes #
#################

class NNscaffold(object):
    """Neural Network object
    """
    def __init__(self, architecture, learning_rate):
        """Initiates a scaffold network with default values
        Args:
            architecture: JSON file outlining neural network scaffold
            learning_rate: floating point number established in main.py FLAGS.learningRate
        """
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.inputs = {} # initializes input dictionary
        self.representations = list() # initializes representations list
        for key in architecture['Inputs']:
            # feeds to output key a placeholder with key's input height and with
            self.inputs[key] = tf.placeholder(tf.float32, [None, architecture['Modules'][key]["input_height"],
                                               architecture['Modules'][key]["input_width"], 1], name=key)
            # appends deep learning layer framework for each key to representations list
            self.representations.append(self._create_track_module(key))
        self.outputs = {} # initializes output dictionary
        self.output_tensor = {} # initializes output_tensor
        for key in architecture['Outputs']:
            # feeds to output key a placeholder with key's input height and with
            self.outputs[key] = tf.placeholder(tf.float32, [None, architecture['Modules'][key]["input_height"],
                                                architecture['Modules'][key]["input_width"], 1], name='output_' + key)
            try:
                # converts output key placeholder to probability distribution function
                self.output_tensor[key] = transform_track(self.outputs[key], option='pdf')
            except TypeError:
                print(key, self.outputs[key])
                print(type(self.outputs[key].get_shape()[0].value))
                raise
        self.dropout = tf.placeholder(tf.float32) # initializing data type input for dropout
        self.keep_prob_input = tf.placeholder(tf.float32) # initializing data type input for keep_prob_input
        # Used for modality-wise dropout. Equivalent to batch_size for training, test size for testing
        self.inp_size = tf.placeholder(tf.int32) # initializing data type input for keep_prob_input
        self._combine_representations(mode='convolution') # combines representations into convolutional layer
        self._encapsulate_models() # ... what the heck
        # Define loss function based variational upper-bound and corresponding optimizer
        self._create_loss_optimizer()

    def _combine_representations(self, mode):
        """Concatenates tensors in representations list to either a convolution or fully connected representation
        Args:
            mode: convolution or fully_connected
        """
        if mode == 'convolution':
            self.combined_representation = tf.concat(1, self.representations)
        elif mode == 'fully_connected':
            raise NotImplementedError
            self.combined_representation = tf.concat(0, self.representations)
        else:
            raise NotImplementedError

    def initialize(self, restore_dirs=None):
        """Initialize the scaffold model either from saved checkpoints (pre-trained)
        or from scratch
        """
        self.sess = tf.Session() # Launch the session
#        #################debugger########################################
#        from tensorflow.python import debug as tf_debug
#        self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
#        self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
#        #################debugger########################################
        # Initializing the tensor flow variables
        # init = tf.initialize_all_variables() # EDIT
        init = tf.global_variables_initializer() # std out recommended this instead
        self.sess.run(init)
        print('Session initialized.')
        # if restore_dirs is not None:
        #     for key in self.architecture.keys():
        #         saver = tf.train.Saver([v for v in tf.trainable_variables() if key in v.name])
        #         saver.restore(self.sess,restore_dirs[key]+'model.ckpt')
        #         print('Session restored for '+key)

    def load(self, model_path):
        """
        loads the pretrained model from the specified path
        """
        #TODO: add frozen model loading option...
        #TODO: add partially pre-trained module loading option ...
        # Launch the session
        self.sess = tf.Session()
        # Initializing the tensor flow variables
        # init = tf.initialize_all_variables() # EDIT
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print([v.name for v in tf.trainable_variables()])
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(self.sess, model_path+'/model.ckpt')
        print('Model loaded with the pre-trained parameters')

    def _encapsulate_models(self):
        with tf.variable_scope('scaffold'):
            self.net = tf.reshape(self.combined_representation, shape=[-1, len(self.representations), self._scaffold_width, 1])
            self.net = tf.nn.dropout(tf.identity(self.net), self.keep_prob_input, noise_shape=[self.inp_size, len(self.representations), 1, 1])
            self.net = tflearn.conv_2d(self.net,
                                       self.architecture['Scaffold']['Layer1']['number_of_filters'],
                                       [len(self.representations), self.architecture['Scaffold']['Layer1']['filter_width']],
                                       activation=self.architecture['Scaffold']['Layer1']['activation'],
                                       regularizer="L2", padding='valid', name='conv_combined')
            self.net = tflearn.layers.conv.avg_pool_2d(self.net, [1, self.architecture['Scaffold']['Layer1']['pool_size']],
                                                       strides=[1, self.architecture['Scaffold']['Layer1']['pool_stride']],
                                                       padding='valid', name='AvgPool_combined')
            self.net = tflearn.layers.normalization.batch_normalization(self.net, name='batch_norm_2')
            self.net = tflearn.flatten(self.net)
            self.scaffold_representation = tflearn.fully_connected(self.net,
                                                                   self.architecture['Scaffold']['representation_width'],
                                                                   activation='linear', name='representation')
            self.predictions = {}
            for key in self.architecture['Outputs']:
                self.net = tflearn.fully_connected(self.scaffold_representation,
                                                   self.architecture['Modules'][key]['input_height'] *
                                                   self.architecture['Modules'][key]['input_width'], name='final_FC')
                if key == 'DNAseq':
                    self.net = tf.reshape(self.net, [-1, 4, self.architecture['Modules']['DNAseq']['input_width'], 1])
                    self.predictions[key] = multi_softmax(self.net, axis=1, name='multiSoftmax')

                else:
                    self.predictions[key] = tf.nn.softmax(self.net, name='softmax')

    def _create_track_module(self, key):
        with tf.variable_scope(key):

            # net = tflearn.input_data(shape=[None,
            #                                 self.architecture['Modules'][key]['input_height'],
            #                                 self.architecture['Modules'][key]['input_width'],
            #                                 1], name='input')

            net = tflearn.conv_2d(self.inputs[key], self.architecture['Modules'][key]['Layer1']['number_of_filters'],
                                  [self.architecture['Modules'][key]['Layer1']['filter_height'],
                                   self.architecture['Modules'][key]['Layer1']['filter_width']],
                                  activation=self.architecture['Modules'][key]['Layer1']['activation'],
                                  regularizer="L2",
                                  padding='valid',
                                  name='conv_1')
            net = tflearn.layers.conv.avg_pool_2d(net, [1, self.architecture['Modules'][key]['Layer1']['pool_size']],
                                                  strides=[1, self.architecture['Modules'][key]['Layer1']['pool_stride']],
                                                  padding='valid', name='AvgPool_1')
            net = tflearn.layers.normalization.batch_normalization(net, name='batch_norm_1')
            net = tflearn.conv_2d(net, self.architecture['Modules'][key]['Layer2']['number_of_filters'],
                                  [self.architecture['Modules'][key]['Layer2']['filter_height'],
                                   self.architecture['Modules'][key]['Layer2']['filter_width']],
                                  activation=self.architecture['Modules'][key]['Layer2']['activation'],
                                  regularizer="L2",
                                  padding='valid',
                                  name='conv_2')
            net = tflearn.layers.conv.avg_pool_2d(net, [1, self.architecture['Modules'][key]['Layer2']['pool_size']],
                                                  strides=[1, self.architecture['Modules'][key]['Layer2']['pool_stride']],
                                                  padding='valid',
                                                  name='AvgPool_2')
            net = tflearn.layers.normalization.batch_normalization(net, name='batch_norm_2')
            net = tflearn.fully_connected(net,
                                          self.architecture['Modules'][key]['representation_width'],
                                          name='representation')

            # seems that _scaffold_width is defaulted to representation width?
        if not hasattr(self, '_scaffold_width'):
            self._scaffold_width = self.architecture['Modules'][key]['representation_width']
        return net

    def _create_loss_optimizer(self):
        self.accuracy = {}
        self.cost = 0
        for key in self.architecture['Outputs']:
            if key != 'DNAseq':
                self.loss = KL_divergence(self.predictions[key], self.output_tensor[key])
                width = self.architecture['Modules'][key]["input_width"] * self.architecture['Modules'][key]["input_height"]
                target = tf.floor((10.*tf.cast(tf.argmax(self.output_tensor[key], dimension=1), tf.float32))/np.float(width))
                pred = tf.floor((10.*tf.cast(tf.argmax(self.predictions[key], dimension=1), tf.float32))/np.float(width))
                self.accuracy[key] = tf.reduce_sum(tf.cast(tf.equal(pred, target), tf.int32))

            else:
                #TODO implement for DNA seq # but is necessary??
                self.loss = tf.reduce_sum(tf.mul(self.output_tensor[key]+1e-10,
                                                 tf.sub(tf.log(self.output_tensor[key]+1e-10),
                                                        tf.log(self.predictions[key]+1e-10))), [1, 2])
                target = tf.argmax(self.output_tensor[key], dimension=1)
                pred = tf.argmax(self.predictions[key], dimension=1)
                self.accuracy[key] = tf.reduce_sum(tf.cast(tf.equal(pred, target), tf.float32))/target.get_shape()[1].value

            self.cost += tf.reduce_mean(self.loss)   # average over batch

        self.global_step = tf.Variable(0, name='globalStep', trainable=False)
#pdb.set_trace()
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, global_step=self.global_step)

    def train(self, train_data, accuracy=None, inp_dropout=0.9):
        """Trains model based on mini-batch of input data. Returns cost of mini-batch.
        """
        train_feed = {self.outputs[key]: train_data[1][key] for key in self.architecture['Outputs']}
        train_feed.update({self.inputs[key]: train_data[0][key] for key in self.architecture['Inputs']})
        tmpKey = train_data[1].keys()[0]
        train_feed.update({self.dropout: self.architecture['Scaffold']['dropout'],
                           self.keep_prob_input: inp_dropout,
                           self.inp_size: train_data[1][tmpKey].shape[0]})
                           # revisit this...
                           #self.inp_size: train_data.values()[0].shape[0]}) # unsure whether accessing 128 or 500...?
        fetches = {'_': self.optimizer, 'cost': self.cost}
        if accuracy is not None:
            fetches.update({'accuracy_' + key: val for key, val in self.accuracy.items()})
        return_dict = self._run(fetches, train_feed)
        return return_dict

    def validate(self, validation_data, accuracy=None):
        """Tests model based on mini-batch of input data. Returns cost of test.
        """
        if not hasattr(self, 'test_feed'):
            self.test_feed = {self.outputs[key]: validation_data[key] for key in self.architecture['Outputs']}
#pdb.set_trace()
            self.test_feed.update({self.inputs[key]: validation_data[key] for key in self.architecture['Inputs']})
#pdb.set_trace()
            # this step right here... adds in placeholders :0, _1:0, _2:0 ... culprit?
            self.test_feed.update({self.dropout: 1.,
                                   self.keep_prob_input: 1.,
                                   self.inp_size: validation_data.values()[0].shape[0]})
#pdb.set_trace()
        fetches = {'cost': self.cost}
#        #################debugger########################################
#        from tensorflow.python import debug as tf_debug
#        self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
#        self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
#        #################debugger########################################
        if accuracy is not None:
#pdb.set_trace()
            fetches.update({'accuracy_' + key: val for key, val in self.accuracy.items()})
        return_dict = self._run(fetches, self.test_feed)
#pdb.set_trace()
        return return_dict

    def suffnec(self, trainInp, trainOut):
        """Calculates the sufficiency and necessity score of an input dataset by setting the
        rest of the inputs to their average values and the input of interest to its average value, respectively.

        Returns cost dictionary.
        """

        print('Calculating sufficiency...')
        suffDict ={}

        self.train_feed = {self.output:trainOut.values()[0], self.dropout:1, self.keep_prob_input:1.,self.inp_size:trainOut.values()[0].shape[0]}
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
        necDict = {}
        for inputName in self.architecture.keys():
            print('...'+inputName)
            self.train_feed = {self.output:trainOut.values()[0], self.dropout:1,self.keep_prob_input:1.,self.inp_size:trainOut.values()[0].shape[0]}
            for key in self.architecture.keys():
                if key is not inputName:
                    self.train_feed.update({self.inputs[key]: trainInp[key]})
                else:
                    self.train_feed.update({self.inputs[key]: np.tile(trainInp[key].mean(axis=0),(trainInp[key].shape[0],1,1,1))})
            necDict[inputName] = self.sess.run(self.loss, feed_dict=self.train_feed)

        self.train_feed = {self.output: trainOut.values()[0], self.dropout:1,self.keep_prob_input:1.,self.inp_size:trainOut.values()[0].shape[0]}
        for key in self.architecture.keys():
            self.train_feed.update({self.inputs[key]: trainInp[key]})
        necDict['AllActive'] = self.sess.run(self.loss, feed_dict=self.train_feed)

        return suffDict, necDict

    def predict(self, testInp):
        """Return the result of a flow based on mini-batch of input data.
        """
        self.test_feed = {self.dropout:1, self.keep_prob_input:1.,self.keep_prob_input:1.,self.inp_size:testInp.values()[0].shape[0]}
        self.test_feed.update({self.inputs[key]: testInp[key] for key in self.architecture.keys()})
        return self.sess.run( self.net, feed_dict=self.test_feed)

    def summarize(self, step):
        summaryStr = self.sess.run(self.summary_op, feed_dict=self.test_feed)
        self.summaryWriter.add_summary(summaryStr, step)
        self.summaryWriter.flush()

    def create_monitor_variables(self, savePath):
        # for monitoring
        # tf.scalar_summary('KL divergence', self.cost) # supposedly deprecated: # EDIT
        tf.summary.scalar('KL divergence', self.cost)
        for key, val in self.accuracy.items():
            # tf.scalar_summary(key+'/Accuracy', val) # EDIT
            tf.summary.scalar(key+'/Accuracy', val)
        self.summary_op = tf.merge_all_summaries()
        # self.summaryWriter = tf.train.SummaryWriter(savePath, self.sess.graph) # supposedly deprecated: # EDIT
        self.summaryWriter = tf.summary.FileWriter(savePath, self.sess.graph)


    def _run(self, fetches, feed_dict):
        """Wrapper for making Session.run() more user friendly.
        Adapted from @Styrke : https://github.com/tensorflow/tensorflow/issues/1941
        With this function, fetches can be either a list or a dictionary.

        If fetches is a list, this function will behave like
        tf.session.run() and return a list in the same order as well. If
        fetches is a dict then this function will also return a dict where
        the returned values are associated with the corresponding keys from
        the fetches dict.

        Keyword arguments:
        session -- An open TensorFlow session.
        fetches -- A list or dict of ops to fetch.
        feed_dict -- The dict of values to feed to the computation graph.
        """
        if isinstance(fetches, dict):
#pdb.set_trace()
            thing1 = tf.Print(fetches['cost'], [fetches['cost']], "fetches['cost']: ")
            keys, values = fetches.keys(), list(fetches.values())
#pdb.set_trace()
            #################debugger########################################
            from tensorflow.python import debug as tf_debug
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            #################debugger########################################
            thing2 = tf.Print(values[0], [values[0]], "this is values[0]")
            thing3 = tf.Print(values[1], [values[1]], "this is values[1]")
#pdb.set_trace()
            res = self.sess.run(values, feed_dict)
            print('intermediate')
#pdb.set_trace()
            return {key: value for key, value in zip(keys, res)}
        else:
            print('actually, this one')
            return self.sess.run(fetches, feed_dict)

    def profile(self):
        from tensorflow.python.client import timeline

        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()
        self.sess.run(self.optimizer, options=self.run_options, run_metadata=self.run_metadata, feed_dict=self.test_feed)

        # Create the Timeline object, and write it to a json
        tl = timeline.Timeline(self.run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)
