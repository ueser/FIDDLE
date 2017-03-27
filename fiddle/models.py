from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

###running on a shared cluster ###
import sys
sys.path.append('/home/ue4/tfvenv/lib/python2.7/site-packages/')
##################################


import pdb, traceback, sys # EDIT
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Lambda, Conv2D, concatenate, Reshape, AveragePooling2D, Flatten, BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.objectives import kullback_leibler_divergence
import json, six, copy



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
    return tf.reduce_sum(tf.multiply(K.clip(P_, K.epsilon(), 1),
                                     tf.subtract(tf.log(K.clip(P_, K.epsilon(), 1)),
                                                 tf.log(K.clip(Q_, K.epsilon(), 1)))), 1)


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
                                   [-1, (track_data_placeholder.get_shape()[1] * track_data_placeholder.get_shape()[
                                       2]).value]) + K.epsilon()
        output_tensor = tf.divide(output_tensor, tf.reduce_sum(output_tensor, 1, keep_dims=True))
    # NOT completed yet
    elif option == 'standardize':
        raise NotImplementedError
        from scipy import stats
        output_tensor = stats.zscore(output_tensor, axis=1)
    return output_tensor

def byteify(json_out):
    '''
    Recursively reads in .json to string conversion into python dictionary format
    '''
    if isinstance(json_out, dict):
        return {byteify(key): byteify(value)
                for key, value in six.iteritems(json_out)}
    elif isinstance(json_out, list):
        return [byteify(element) for element in json_out]
    elif isinstance(json_out, unicode):
        return json_out.encode('utf-8')
    else:
        return json_out

#################
# Model Classes #
#################

class NNscaffold(object):
    """Neural Network object
    """
    def __init__(self, configuration_path='configurations.json', architecture_path='architecture.json',
                 learning_rate=0.01, batch_norm=False):
        """Initiates a scaffold network with default values
        Args:
            architecture: JSON file outlining neural network scaffold
            learning_rate: floating point number established in main.py FLAGS.learningRate
        """
        with open(configuration_path) as fp:
            self.config = byteify(json.load(fp))
        print('Stranded:', self.config['Options']['Stranded'])
        self.batch_norm=False
        self._parse_parameters(architecture_path)
        self.learning_rate = learning_rate
        self.representations = list() # initializes representations list
        self.inputs = {}  # initializes input dictionary
        for key in self.architecture['Inputs']:
            # feeds to output key a placeholder with key's input height and with
            self.inputs[key] = tf.placeholder(tf.float32, [None, self.architecture['Modules'][key]["input_height"],
                                                           self.architecture['Modules'][key]["input_width"], 1],
                                              name=key)
            # appends deep learning layer framework for each key to representations list
            self.representations.append(self._create_track_module(key))

        self.output_tensor = {}  # initializes output_tensor
        self.outputs = {}  # initializes output dictionary
        for key in self.architecture['Outputs']:
            # feeds to output key a placeholder with key's input height and with
            self.outputs[key] = tf.placeholder(tf.float32, [None, self.architecture['Modules'][key]["input_height"],
                                                            self.architecture['Modules'][key]["input_width"], 1],
                                               name='output_' + key)
            if self.config['Options']['Stranded']:
                self.positive_strand = tf.slice(self.outputs[key],[0,0,0,0], [-1,1,-1,-1])
                self.output_tensor[key] = transform_track(self.positive_strand, option='pdf')
            else:
                # converts output key placeholder to probability distribution function
                self.output_tensor[key] = transform_track(self.outputs[key], option='pdf')




        self.dropout = tf.placeholder(tf.float32) # initializing data type input for dropout
        self.keep_prob_input = tf.placeholder(tf.float32) # initializing data type input for keep_prob_input
        # Used for modality-wise dropout. Equivalent to batch_size for training, test size for testing
        self.inp_size = tf.placeholder(tf.int32) # initializing data type input for keep_prob_input
        self._combine_representations(mode='convolution') # combines representations into convolutional layer
        self._encapsulate_models() # ... what the heck
        # Define loss function based variational upper-bound and corresponding optimizer
        self._create_loss_optimizer()

    def _parse_parameters(self, architecture_path='architecture.json'):
        #####################################
        # Architecture and model definition #
        #####################################

        # Read in the architecture parameters, defined by the user
        print('Constructing architecture and model definition')
        with open(architecture_path) as fp:
            architecture_template = byteify(json.load(fp))

        self.architecture = {'Modules': {}, 'Scaffold': {}}
        self.architecture['Scaffold'] = architecture_template['Scaffold']
        self.architecture['Inputs'] = self.config['Options']['Inputs']
        self.architecture['Outputs'] = self.config['Options']['Outputs']

        for key in self.architecture['Inputs']+self.architecture['Outputs']:
            self.architecture['Modules'][key] = copy.deepcopy(architecture_template['Modules'])
            self.architecture['Modules'][key]['input_height'] = self.config['Tracks'][key]['input_height']
            self.architecture['Modules'][key]['Layer1']['filter_height'] = self.config['Tracks'][key]['input_height']
            # Parameters customized for specific tracks are read from self.architecture.json and updates the arch. dict.
            if key in architecture_template.keys():
                for key_key in architecture_template[key].keys():
                    sub_val = architecture_template[key][key_key]
                    if type(sub_val) == dict:
                        for key_key_key in sub_val:
                            self.architecture['Modules'][key][key_key][key_key_key] = sub_val[key_key_key]
                    else:
                        self.architecture['Modules'][key][key_key] = sub_val


    def _combine_representations(self, mode):
        """Concatenates tensors in representations list to either a convolution or fully connected representation
        Args:
            mode: convolution or fully_connected
        """
        if mode == 'convolution':
            self.combined_representation = tf.concat(self.representations, 1)
        elif mode == 'fully_connected':
            raise NotImplementedError
            self.combined_representation = tf.concat(self.representations, 0)
        else:
            raise NotImplementedError

    def initialize(self, restore_dirs=None):
        """Initialize the scaffold model either from saved checkpoints (pre-trained)
        or from scratch
        """
        self.sess = tf.Session() # Launch the session
        # Initializing the tensor flow variables
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
            # Modality-wise dropout
            self.net = tf.nn.dropout(tf.identity(self.net), self.keep_prob_input, noise_shape=[self.inp_size, len(self.representations), 1, 1])
            self.net = Conv2D(self.architecture['Scaffold']['Layer1']['number_of_filters'],
                                       [len(self.representations), self.architecture['Scaffold']['Layer1']['filter_width']],
                                       activation=self.architecture['Scaffold']['Layer1']['activation'],
                                       kernel_regularizer='l2', padding='valid', name='conv_combined')(self.net)
            self.net = AveragePooling2D([1, self.architecture['Scaffold']['Layer1']['pool_size']],
                                                       strides=[1, self.architecture['Scaffold']['Layer1']['pool_stride']],
                                                       padding='valid', name='AvgPool_combined')(self.net)
            if self.batch_norm:
                self.net = BatchNormalization()(self.net)
            self.net = Flatten()(self.net)
            self.scaffold_representation = Dense(self.architecture['Scaffold']['representation_width'],
                                                 activation='linear', name='representation')(self.net)

            self.predictions = {}
            for key in self.architecture['Outputs']:
                if self.config['Options']['Stranded']:
                    self.net = Dense(self.architecture['Modules'][key]['input_width'],
                                 activation='linear',
                                 name='final_FC')(self.scaffold_representation)
                else:
                    self.net = Dense(self.architecture['Modules'][key]['input_height'] *
                                     self.architecture['Modules'][key]['input_width'],
                                     activation='linear',
                                     name='final_FC')(self.scaffold_representation)

                if key == 'DNAseq':
                    self.net = tf.reshape(self.net, [-1, 4, self.architecture['Modules']['DNAseq']['input_width'], 1])
                    self.predictions[key] = multi_softmax(self.net, axis=1, name='multiSoftmax')

                else:
                    self.predictions[key] = tf.nn.softmax(self.net, name='softmax')

    def _create_track_module(self, key):
        with tf.variable_scope(key):
            net = Conv2D(self.architecture['Modules'][key]['Layer1']['number_of_filters'],
                         [self.architecture['Modules'][key]['Layer1']['filter_height'],
                         self.architecture['Modules'][key]['Layer1']['filter_width']],
                         activation=self.architecture['Modules'][key]['Layer1']['activation'],
                         kernel_regularizer='l2',
                         padding='valid',
                         name='conv_1')(self.inputs[key])
            net = AveragePooling2D((1, self.architecture['Modules'][key]['Layer1']['pool_size']),
                                    strides=(1, self.architecture['Modules'][key]['Layer1']['pool_stride']))(net)
            if self.batch_norm:
                net = BatchNormalization()(net)
            net = Conv2D(self.architecture['Modules'][key]['Layer2']['number_of_filters'],
                                  [self.architecture['Modules'][key]['Layer2']['filter_height'],
                                   self.architecture['Modules'][key]['Layer2']['filter_width']],
                                  activation=self.architecture['Modules'][key]['Layer2']['activation'],
                                  kernel_regularizer='l2',
                                  padding='valid',
                                  name='conv_2')(net)

            net = AveragePooling2D([1, self.architecture['Modules'][key]['Layer2']['pool_size']],
                                    strides=[1, self.architecture['Modules'][key]['Layer2']['pool_stride']],
                                    padding='valid',
                                    name='AvgPool_2')(net)
            if self.batch_norm:
                net = BatchNormalization()(net)
            net = Flatten()(net)
            net = Dense(self.architecture['Modules'][key]['representation_width'],
                        name='representation')(net)


            # seems that _scaffold_width is defaulted to representation width?
        if not hasattr(self, '_scaffold_width'):
            self._scaffold_width = self.architecture['Modules'][key]['representation_width']
        return net

    def _create_loss_optimizer(self):
        self.accuracy = {}
        self.cost = 0
        for key in self.architecture['Outputs']:
            if key != 'DNAseq':
                self.loss = kullback_leibler_divergence(self.output_tensor[key], self.predictions[key])
                # self.loss = KL_divergence(self.output_tensor[key], self.predictions[key])
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
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, global_step=self.global_step)

    def train(self, train_data, accuracy=None, inp_dropout=0.9, batch_size=128):
        """Trains model based on mini-batch of input data. Returns cost of mini-batch.
        """

        if train_data==[]:
            train_feed = {}
        else:
            train_feed = {self.outputs[key]: train_data[key] for key in self.architecture['Outputs']}
            train_feed.update({self.inputs[key]: train_data[key] for key in self.architecture['Inputs']})
            tmpKey = train_data.keys()[0]

        train_feed.update({self.dropout: self.architecture['Scaffold']['dropout'],
                           self.keep_prob_input: inp_dropout,
                           self.inp_size: batch_size,
                           K.learning_phase(): 1})

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
            self.test_feed.update({self.inputs[key]: validation_data[key] for key in self.architecture['Inputs']})
            self.test_feed.update({self.dropout: 1.,
                                   self.keep_prob_input: 1.,
                                   self.inp_size: validation_data.values()[0].shape[0],
                                   K.learning_phase(): 0})

        fetches = {'cost': self.cost}
        if accuracy is not None:
            fetches.update({'accuracy_' + key: val for key, val in self.accuracy.items()})
        return_dict = self._run(fetches, self.test_feed)
        return return_dict

    def predict(self, testInp):
        """Return the result of a flow based on mini-batch of input data.
        """
        self.test_feed = {self.dropout:1,
                          self.keep_prob_input:1.,
                          self.keep_prob_input:1.,
                          self.inp_size:testInp.values()[0].shape[0],
                          K.learning_phase():0}
        self.test_feed.update({self.inputs[key]: testInp[key] for key in self.architecture.keys()})
        return self.sess.run( self.net, feed_dict=self.test_feed)

    def summarize(self, step):
        summaryStr = self.sess.run(self.summary_op, feed_dict=self.test_feed)
        self.summaryWriter.add_summary(summaryStr, step)
        self.summaryWriter.flush()

    def create_monitor_variables(self, savePath):
        """Writes to results directory a summary of graph variables"""
        tf.summary.scalar('KL divergence', self.cost)
        for key, val in self.accuracy.items():
            tf.summary.scalar(key+'/Accuracy', val)
        self.summary_op = tf.summary.merge_all()
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
        # pdb.set_trace()
        if isinstance(fetches, dict):
            keys, values = fetches.keys(), list(fetches.values())

            ##################debugger########################################
            # from tensorflow.python import debug as tf_debug
            # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
            # self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            # ##################debugger########################################
            # pdb.set_trace()
            res = self.sess.run(values, feed_dict)
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
