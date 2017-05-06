"""Author: Umut Eser
Documentation: Dylan Marshall

'models.py' handles the creation and use of convolutional neural networks for FIDDLE.

Usage:
    To utilize methods and classes in models.py, place the following import
    command in the imports of a python file.

        > from models import *
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

###################################################################
# Running on a shared cluster specific tools
import sys
sys.path.append('/home/ue4/tfvenv/lib/python2.7/site-packages/')
##################################################################

import pdb, traceback, sys
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Lambda, Conv2D, concatenate, Reshape, AveragePooling2D, Flatten, BatchNormalization, MaxPooling2D
from keras.models import Model
from keras import backend as K
from keras.objectives import kullback_leibler_divergence
import json, six, copy, os
from visualization import put_kernels_on_grid, plot_prediction

################################################################################
#                              Global Variables                                #
################################################################################
global TRAIN_FETCHES
global VALIDATION_FETCHES
global PREDICTION_FETCHES

TRAIN_FETCHES = {}
VALIDATION_FETCHES = {}
PREDICTION_FETCHES ={}

################################################################################
#                             Auxiliary Methods                                #
################################################################################
def multi_softmax(target, axis = 1, name = None):
    """Applies softmax to a particular axis of an inputted tensor

    Args:
        :param target: (tf.tensor) tensor subject to axis dependent softmax function
        :param axis: (int, default = 1) axis along which softmax is applied
        :param name: (string, default = None) naming applied to resulting tf.tensor

    Returns:
        Tensor with softmax operation applied to a particular axis
    """

    with tf.name_scope(name):
        mx = tf.reduce_max(target, axis, keep_dims = True)
        Q = tf.exp(target - mx)
        Z = tf.reduce_sum(Q, axis, keep_dims = True)
    return Q / Z

def transform_track(track_ph, option = 'pdf'):
    """Calculates probability distribution - continuous, categorical, standardized - of inputted tensor, usually DNA region

    Args:
        :param track_ph: (tf.placeholder) initialized tensor of data to be transformed
        :param option: (string, default = pdf) type of probability distribution calculation desired on track_ph tensor

    Returns:
        Description of probability distribution of inputted tensor

    Todo:
        :param categorical: discretisizes the continuous input (To be implemented)
        :param standardize: zero mean, unit variance
    """

    if option == 'pdf':
        epsilon = K.epsilon() # default = 1e-07
        track_ph_axis_1 = track_ph.get_shape()[1] # usually one DNA strand
        track_ph_axis_2 = track_ph.get_shape()[2] # usually the other DNA strand
        output_tensor = tf.reshape(track_ph, [-1, (track_ph_axis_1 * track_ph_axis_2).value]) + epsilon
        output_tensor = tf.divide(output_tensor, tf.reduce_sum(output_tensor, 1, keep_dims = True))
    elif option == 'standardize': #TODO: implement categorical and standardized probability distribution outputs
        raise NotImplementedError
        from scipy import stats
        output_tensor = stats.zscore(output_tensor, axis=1)
    return output_tensor

def byteify(json_out):
    """Recursively reads in .json file content and converts to easily manipulable python dictionary format

    Args:
        :param json_out: (.json file) typically applied to config or arch parameter files

    Returns:
        Dictionary of nested key, value pairs describing inputted .json file
    """

    if isinstance(json_out, dict):
        return {
            byteify(key): byteify(value)
            for key, value in six.iteritems(json_out)
        }
    elif isinstance(json_out, list):
        return [byteify(element) for element in json_out]
    elif isinstance(json_out, unicode):
        return json_out.encode('utf-8')
    else:
        return json_out

class ArchitectureParsingError(Exception):
    pass

class ConfigurationParsingError(Exception):
    pass

################################################################################
#                               Model Classes                                  #
################################################################################
class Integrator(object):
    """Neural Network object"""

    def __init__(self,
                 config,
                 architecture_path='architecture.json',
                 learning_rate=0.01,
                 batch_norm=False,
                 model_path='../results/example'):
        """Initiates a decoder network with default values

        Args:
            :param config: parameters of data inputs and outputs [json file]
            :param architecture_path: (file name, default = 'architecture.json') parameters describing CNNs [json file]
            :param learning_rate: (floating point, default = 0.01) value correlated with training speed
            :param batch_norm: (boolean, default = False) whether or not batch normalization is implemented
            :param model_path: (directory name, default = '../results/example') directory where Integrator model will be saved
        """

        print('Sanity check for configurations.json "Options": "Strand":', config['Options']['Strand'])

        # define Integrator attributes
        self.config = config # copy in byteify-ied configurations.json file
        self.model_path = model_path # copy in directory destination for saved model
        self._parse_parameters(architecture_path) # parse in parameters defining architecture of neural network
        self.dropout = tf.placeholder(tf.float32) # create placeholder for dropout probability
        self.keep_prob_input = tf.placeholder(tf.float32) # create placeholder for 1 - dropout probability
        self.inp_size = tf.placeholder(tf.int32) # create placeholder for data input size
        self.learning_rate = learning_rate # copy in learning_rate floating point value
        self.representations = {}  # initializes representations dictionary
        self.tracks = {}  # initializes dictionary of key = input track, value = CNN Container
        self.inputs = {}  # initializes input dictionary of key = input track, value = inputs to corresponding CNN Container

        
        self.router = Router() # router object gathers the representations from encoders (prev. known as 
        # Convolutional Containers), then provides selected ones to the prediction decoders, 
        # (previously, known as Decoder)
        
        # stack representations of inputs into unified representation
        for track_name in self.architecture['Inputs']:
            with tf.variable_scope(track_name):
                self.tracks[track_name] = Encoder(track_name, self.architecture)
            self.inputs[track_name] = self.tracks[track_name].input
            self.router.stack_input(self.tracks[track_name].representation, track_name)



        self.decoders = {}
        for track_name in self.architecture['Outputs']:
            with tf.variable_scope(track_name):
                self.decoders[track_name] = Decoder(architecture=self.architecture,
                                                    dropout=self.dropout,
                                                    keep_prob_input=self.keep_prob_input,
                                                    inp_size=self.inp_size,
                                                    batch_norm=batch_norm,
                                                    strand=self.config['Options']['Strand'],
                                                    name=track_name)
                self.decoders[track_name].representations = self.router.route(block_list=[track_name])
                self.decoders[track_name].combine_representations() # combines representations into convolutional layer



        # define Integrator attributes
        self.output_tensor = {}  # initializes output_tensor
        self.outputs = {}  # initializes output dictionary
        self.cost_functions = {} # initializes cost functions dictionary

        # define output dictionary of key = output, values = tf.variables of prob representation
        for key in self.architecture['Outputs']:
            input_height = self.architecture['Modules'][key]["input_height"]
            input_width = self.architecture['Modules'][key]["input_width"]
            self.outputs[key] = tf.placeholder(tf.float32, [None, input_height, input_width, 1], name = 'output_' + key)
            if key != 'dnaseq':
                if self.config['Options']['Strand'] == 'Single':
                    self.positive_strand = tf.slice(self.outputs[key], [0, 0, 0, 0], [-1, 1, -1, -1])
                    self.output_tensor[key] = transform_track(self.positive_strand, option = 'pdf')
                else:
                    self.output_tensor[key] = transform_track(self.outputs[key], option = 'pdf')
                self.cost_functions[key] = kl_loss
            else:
                #TODO: incorporate topologically relevant dataset options
                self.output_tensor[key] = self.outputs[key]
                self.cost_functions[key] = multi_softmax_classification

        #TODO: self.freeze ... is boolean?
        self.freeze() # define whether model will be frozen at this point
        self._create_loss_optimizer() # Define loss function gradient optimizer

    def _parse_parameters(self, architecture_path = 'architecture.json'):
        """Read in architecture file, copy in parameters to self

        Args:
            :param architecture_path: (.json file, default = architecture.json) parameters describing CNNs
        """

        print('Constructing architecture and model definition')
        with open(architecture_path) as fp:
            architecture_template = byteify(json.load(fp))

        # define Integrator attributes
        if 'Inputs' in architecture_template.keys():
            self.architecture = architecture_template
        else:
            self.architecture = {'Modules': {}, 'Scaffold': {}}
            self.architecture['Scaffold'] = architecture_template['Scaffold']
            self.architecture['Inputs'] = self.config['Options']['Inputs']
            self.architecture['Outputs'] = self.config['Options']['Outputs']

            for key in self.architecture['Inputs'] + self.architecture['Outputs']:
                self.architecture['Modules'][key] = copy.deepcopy(architecture_template['Modules'])
                self.architecture['Modules'][key]['input_height'] = self.config['Tracks'][key]['input_height']
                self.architecture['Modules'][key]['Layer1']['filter_height'] = self.config['Tracks'][key]['input_height']
                if key in architecture_template.keys():
                    for key_key in architecture_template[key].keys():
                        sub_val = architecture_template[key][key_key]
                        if type(sub_val) == dict:
                            for key_key_key in sub_val:
                                self.architecture['Modules'][key][key_key][key_key_key] = sub_val[key_key_key]
                        else:
                            self.architecture['Modules'][key][key_key] = sub_val

    def initialize(self):
        """Initialize the decoder model either from scratch or from saved checkpoints (pre-trained)"""

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print('Session initialized.')
        #TODO: orient tf.Session() to allow if/else loading, is currently implemented no matter what
        #self._load()

    def _load(self):
        """Loads the pretrained model from the specified path"""

        #TODO: add frozen model loading option...
        #TODO: add partially pre-trained module loading option ...
        if not hasattr(self, 'sess'):
            self.sess = tf.Session()
        if ('all' in self.config['Options']['Reload']) or ('All' in self.config['Options']['Reload']):
            load_list = [track_name+'/encoder' for track_name in self.architecture['Inputs']]
            load_list += [track_name+'/decoder' for track_name in self.architecture['Outputs']]
        else:
            load_list = [track_name+'/encoder' for track_name in self.config['Options']['Reload']['Encoders']]
            load_list += [track_name+'/decoder' for track_name in self.config['Options']['Reload']['Decoders']]
        for scope in load_list:
            loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = scope ))
            # loader = tf.train.import_meta_graph(os.path.join(self.model_path, track_name + '_model.ckpt.meta'))
            track_name = scope.split('/')[0]+'_'+scope.split('/')[1]
            loader.restore(self.sess, os.path.join(self.model_path, track_name + '_model.ckpt'))
            print(track_name + ' model is loaded from pre-trained network')

    def freeze(self, freeze_list = []):
        """Filters tracks entering training

        Args:
            :param freeze_list: (list, default = empty) tracks not incorporated in training
        """
        freeze_list += [track_name + '/encoder' for track_name in self.config['Options']['Freeze']['Encoders']]
        freeze_list += [track_name + '/decoder' for track_name in self.config['Options']['Freeze']['Decoders']]
        self.trainables = []
        for key in self.architecture['Inputs']:
            scope = key + '/decoder'
            if scope not in freeze_list:
                vars = [y for y in [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if key in x.name] if
                        'encoder' in y.name]
                self.trainables += vars

        for key in self.architecture['Outputs']:
            scope = key + '/decoder'
            if scope not in freeze_list:
                vars = [y for y in [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if key in x.name] if
                        'decoder' in y.name]
                self.trainables += vars

    def _create_loss_optimizer(self):
        """Define loss function based on variational upper-bound and corresponding gradient optimizer"""

        # define Integrator attributes
        self.global_step = tf.Variable(0, name = 'globalStep', trainable = False)
        self.accuracy = {}
        self.cost = 0

        # define Integrator cost and loss
        for key in self.architecture['Outputs']:

            # apply cost function between output probability distribution and NN predictions
            cst = self.cost_functions[key](self.output_tensor[key], self.decoders[key].prediction)
            TRAIN_FETCHES.update({key + '_loss': cst})
            VALIDATION_FETCHES.update({key + '_loss': cst})
            self.cost += cst
            # determine % accuracy of sequencing peak recalls
            self.accuracy[key] = average_peak_distance(self.output_tensor[key], self.decoders[key].prediction)
            TRAIN_FETCHES.update({key+ '_average_peak_distance': self.accuracy[key]})
            VALIDATION_FETCHES.update({key+ 'Average_peak_distance': self.accuracy[key]})

            # self.performance[key] = self.performance_measures[key](self.output_tensor[key], self.decoders[key].prediction)

        # define Integrator gradient optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate). \
                         minimize(self.cost,
                                  global_step = self.global_step,
                                  var_list = self.trainables)
        TRAIN_FETCHES.update({'_':self.optimizer, 'cost':self.cost})
        VALIDATION_FETCHES.update({'cost': self.cost})

    # TODO: accuracy not utilized ... remove?
    def train(self, train_data, accuracy = None, inp_dropout = 0.1, batch_size = 128):
        """Trains model based on mini-batch of input data, calculates cost of mini-batch input

        Args:
            :param train_data: (io_tools.MultiModalData object) mini-batch from MultiModalData training data iterator
            :param accuracy: (boolean, default = None) ...?
            :param inp_dropout: (double, default = 0.1) probability of hidden unit dropout
            :param batch_size: (int, default = 128) number of inputted data units

        Returns:
            Cost of mini-batch of training data in dictionary format


        """

        # TODO: accuracy not utilized ... remove?
        if train_data == []:
            train_feed = {}
        else:
            train_feed = {self.outputs[key]: train_data[key] for key in self.architecture['Outputs']}
            train_feed.update({self.inputs[key]: train_data[key] for key in self.architecture['Inputs']})


        train_feed.update({
            self.dropout: self.architecture['Scaffold']['dropout'],
            self.keep_prob_input: (1 - inp_dropout),
            self.inp_size: batch_size,
            K.learning_phase(): 1 })

        TRAIN_FETCHES.update({'summary': self.summary_op})
        return_dict = self._run(TRAIN_FETCHES, train_feed)
        return return_dict

    def validate(self, validation_data, accuracy = None):
        """Tests model against validation data

        Args:
            :param validation_data: (io_tools.MultiModalData object) mini-batch from MultiModalData validation data iterator
            :param accuracy: (boolean, default = None) ...?

        Returns:
            Cost of employing validation data for prediction in dictionary format
        """

        # TODO: accuracy not utilized ... remove?
        if not hasattr(self, 'test_feed'):
            self.test_feed = {self.outputs[key]: validation_data[key] for key in self.architecture['Outputs']}
            self.test_feed.update({ self.inputs[key]: validation_data[key] for key in self.architecture['Inputs']})
            self.test_feed.update({
                self.dropout: 1.,
                self.keep_prob_input: 1.,
                self.inp_size: validation_data.values()[0].shape[0],
                K.learning_phase():0 })

        VALIDATION_FETCHES.update({'summary': self.summary_op})
        return_dict = self._run(VALIDATION_FETCHES, self.test_feed)
        return return_dict

    def predict(self, predict_data):
        """Tests model against predetermined indices

        Args:
            :param predict_data: (dictionary) keys = input for prediction, values = indices of data signals

        Returns:
            Cost of prediction in dictionary format
        """

        pred_feed = {}
        pred_feed.update({ self.inputs[key]: predict_data[key] for key in self.architecture['Inputs'] })
        pred_feed.update({
            self.dropout: 1.,
            self.keep_prob_input: 1.,
            self.inp_size: predict_data.values()[0].shape[0],
            K.learning_phase(): 0
        })

        PREDICTION_FETCHES.update({key: self.decoders[key].prediction for key in self.decoders.keys()})
        return_dict = self._run(PREDICTION_FETCHES, pred_feed)
        return return_dict

    def get_representations(self, predict_data):
        """Evaluates predictions at predetermined indices

        Args:
            :param predict_data: (dictionary) keys = input for prediction, values = indices of data signals

        Returns:
            Predictions in manipulable dictionary format
        """

        pred_feed = {}
        pred_feed.update({
            self.inputs[key]: predict_data[key]
            for key in self.architecture['Inputs']
        })

        pred_feed.update({
            self.dropout: 1.,
            self.keep_prob_input: 1.,
            self.inp_size: predict_data.values()[0].shape[0],
            K.learning_phase(): 0
        })

        fetches = {}
        fetches.update(self.router.representations)
        fetches.update({'decoder_'+key: self.decoders[key].decoder_representation for key in self.decoders.keys()})

        return_dict = self._run(fetches, pred_feed)
        return return_dict

    def summarize(self, train_summary, validation_summary, step):
        """Writes to results directory a summary of training and validation steps

        Args:
            :param train_summary: ...
            :param validation_summary: ...
        """

        self.summary_writer_train.add_summary(train_summary, step)
        self.summary_writer_valid.add_summary(validation_summary, step)
        self.summary_writer_train.flush()
        self.summary_writer_valid.flush()

    def create_monitor_variables(self, show_filters = True):
        """Writes to results directory a summary of graph variables

        Args:
            :param show_filters: (boolean, default = True) whether to also write filters to results dir
        """

        if show_filters:
            for track_name in self.inputs.keys():
                weights = [
                    v
                    for v in tf.get_collection(
                        tf.GraphKeys.GLOBAL_VARIABLES, scope = track_name)
                    if 'conv_1/kernel:' in v.name
                ]
                grid = put_kernels_on_grid(weights[0])
                tf.summary.image(track_name + '/conv1/features', grid)

        tf.summary.scalar('KL divergence', self.cost)
        for key, val in self.accuracy.items():
            tf.summary.scalar(key + '/Accuracy', val)
        self.summary_op = tf.summary.merge_all()
        self.summary_writer_train = tf.summary.FileWriter(self.model_path + '/training', self.sess.graph)
        self.summary_writer_valid = tf.summary.FileWriter(self.model_path + '/validation', self.sess.graph)

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
            keys, values = fetches.keys(), list(fetches.values())
            res = self.sess.run(values, feed_dict)
            return {key: value for key, value in zip(keys, res)}
        else:
            return self.sess.run(fetches, feed_dict)

    def profile(self):
        """Allows profiling of 'models.py' to evaluate bottlenecks in computational cost"""

        from tensorflow.python.client import timeline
        self.run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()
        self.sess.run(self.optimizer, options = self.run_options, run_metadata = self.run_metadata, feed_dict = self.test_feed)
        # create the timeline object, and write it to a json
        tl = timeline.Timeline(self.run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)

    def saver(self):
        """Allows saving of checkpoint versions of tf.graph"""
        self.savers_dict = {}
        for key in self.architecture['Inputs']:
            vars = [y for y in [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if key in x.name] if 'encoder' in y.name]
            self.savers_dict[key+'_encoder'] = tf.train.Saver(vars)

        for key in self.architecture['Outputs']:
            # pdb.set_trace()
            vars = [y for y in [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if key in x.name] if
                    'decoder' in y.name]
            self.savers_dict[key+'_decoder'] = tf.train.Saver(vars)



class BaseTrackContainer(object):
    #TODO: this?
    def __init__(self, track_name):
        pass

    def initialize(self):
        """Initialize the model
        """
        if self.sess is None:
            self.sess = tf.Session()  # Launch the session
        # Initializing the tensorflow variables
        init = tf.global_variables_initializer(
        )  # std out recommended this instead
        self.sess.run(init)
        print('Session initialized.')

    def load(self):
        pass

    def forward(self):
        pass

    def freeze(self):
        pass

    def save(self):
        pass

class Encoder(BaseTrackContainer):
    '''Previously known as ConvolutionalContainer'''

    def __init__(self, track_name, architecture, batch_norm=False):
        BaseTrackContainer.__init__(self, track_name)
        self.track_name = track_name
        self.architecture = architecture
        self.batch_norm = batch_norm

        self.input = tf.placeholder(
            tf.float32, [
                None,
                self.architecture['Modules'][self.track_name]["input_height"],
                self.architecture['Modules'][self.track_name]["input_width"], 1
            ],
            name=self.track_name + '_input')
        self._build()

    def _build(self):
        with tf.variable_scope('encoder'):
            net = Conv2D(
                self.architecture['Modules'][self.track_name]['Layer1']
                ['number_of_filters'], [
                    self.architecture['Modules'][self.track_name]['Layer1'][
                        'filter_height'], self.architecture['Modules'][
                            self.track_name]['Layer1']['filter_width']
                ],
                activation=self.architecture['Modules'][self.track_name][
                    'Layer1']['activation'],
                kernel_regularizer='l2',
                padding='valid',
                name='conv_1')(self.input)
            net = AveragePooling2D(
                (1, self.architecture['Modules'][self.track_name]['Layer1'][
                    'pool_size']),
                strides=(1, self.architecture['Modules'][self.track_name][
                    'Layer1']['pool_stride']))(net)
            if self.batch_norm:
                net = BatchNormalization()(net)
            net = Conv2D(
                self.architecture['Modules'][self.track_name]['Layer2']
                ['number_of_filters'], [
                    self.architecture['Modules'][self.track_name]['Layer2'][
                        'filter_height'], self.architecture['Modules'][
                            self.track_name]['Layer2']['filter_width']
                ],
                activation=self.architecture['Modules'][self.track_name][
                    'Layer2']['activation'],
                kernel_regularizer='l2',
                padding='valid',
                name='conv_2')(net)

            net = AveragePooling2D(
                [
                    1, self.architecture['Modules'][self.track_name]['Layer2'][
                        'pool_size']
                ],
                strides=[
                    1, self.architecture['Modules'][self.track_name]['Layer2'][
                        'pool_stride']
                ],
                padding='valid',
                name='AvgPool_2')(net)
            if self.batch_norm:
                net = BatchNormalization()(net)
            net = Flatten()(net)
            self.representation = Dense(
                self.architecture['Modules'][self.track_name][
                    'representation_width'],
                name='representation')(net)

class Decoder():

    def __init__(self, architecture, dropout=1.,
                 keep_prob_input=None,
                 inp_size=None,
                 batch_norm=False,
                 strand='Single',
                 unified=True,
                 name=None):

        self.architecture = architecture
        self.dropout = dropout
        self.keep_prob_input = keep_prob_input
        self.inp_size = inp_size
        self.batch_norm = batch_norm
        self.strand = strand
        self.unified = unified
        self.representations = {}
        self.scope = name



    def combine_representations(self):
        """Concatenates representation tensors """

        self.decoder_height = len(self.representations)
        self.decoder_width = self.architecture['Modules'].values()[0]['representation_width']
        repr_list = []
        activation_type = None

        with tf.variable_scope('decoder'):
            ix = 0
            ix_dict = {}
            for track_name, val in self.representations.items():
                ix_dict[track_name] = ix
                repr_list.append(val)
                ix += 1
            self.combined_representation = tf.reshape(tf.concat(repr_list, 1), shape = [-1, self.decoder_height, self.decoder_width, 1])
            self._encapsulate_models()

    def _encapsulate_models(self):
        """Semi private continuation of combine_representation method"""
            # Modality-wise dropout
            # net = tf.nn.dropout(tf.identity(self.combined_representation), self.keep_prob_input, noise_shape = [self.inp_size, self.decoder_height, 1, 1])
        net = tf.identity(self.combined_representation)
        if self.unified:
            conv_height = 1
            net = tf.reduce_sum(net, 1, keep_dims = True)
        else:
            conv_height = self.decoder_height

        # 2 dimensional convolutional operation
        numFilt = self.architecture['Scaffold']['Layer1']['number_of_filters']
        filterShape = [conv_height, self.architecture['Scaffold']['Layer1']['filter_width']]
        actFunc = self.architecture['Scaffold']['Layer1']['activation']
        net = Conv2D(numFilt, filterShape, activation = actFunc, kernel_regularizer = 'l2', padding = 'valid', name = 'conv_combined')(net)

        # 2 dimensional average pooling operation
        poolSize = self.architecture['Scaffold']['Layer1']['pool_size']
        poolStrides = [1, self.architecture['Scaffold']['Layer1']['pool_stride']]
        net = AveragePooling2D([1, poolSize], strides = poolStrides, padding = 'valid', name = 'AvgPool_combined')(net)

        # apply batch normalization if true
        if self.batch_norm:
            net = BatchNormalization()(net)

        # flatten and apply activation( input * kernel + bias )
        net = Flatten()(net)
        represenWidth = self.architecture['Scaffold']['representation_width']
        self.decoder_representation = Dense(represenWidth, activation = 'linear', name = 'representation')(net)

        # apply fully connected layer and softmax

        inpWidth = self.architecture['Modules'][self.scope]['input_width']

        # define fully connected layer
        if (self.strand == 'Single'):
            net = Dense(inpWidth, activation = 'linear', name = 'final_FC')(self.decoder_representation)
        elif (self.strand == 'Double'):
            inpHeight = self.architecture['Modules'][self.scope]['input_height']
            net = Dense(inpHeight * inpWidth, activation = 'linear', name = 'final_FC')(self.decoder_representation)
        else:
            #TODO: perhaps raise ConfigurationParsingError earlier in main.py?
            raise ConfigurationParsingError('Configuration file should have Strand field as either Single or Double')

        self.prediction = tf.nn.softmax(net, name = 'softmax')


class Router(object):
    def __init__(self):
        self.representations = {}

    def stack_input(self, new_representation, track_name):
        self.representations[track_name] = new_representation

    def route(self, block_list):
        return {key: val for key, val in self.representations.items() if key not in block_list}

################################################################################
#                     Loss Functions and Performance Measures                  #
################################################################################
def kl_loss(y_true, y_pred):
    """Calculates Kullback-Leibler divergence between prior and posterior distributions

    Args:
        :param y_true: (tf.tensor) true distribution of data
        :param y_pred: (tf.tensor) predicted distribution of data

    Returns:
        Average relative entropy between prior and posterior distributions
    """

    KLdiv = kullback_leibler_divergence(y_true, y_pred)
    # PREDICTION_FETCHES.update({'KL_divergence': KLdiv})
    KLloss = tf.reduce_mean(KLdiv)
    # DeltaKL = KLdiv - KLloss
    # TRAIN_FETCHES.update({'DeltaKL': DeltaKL})
    return KLloss

def per_bp_accuracy(y_true, y_pred):
    pass

def average_peak_distance(y_true, y_pred):
    """Calculates distance between maximum of predicted and actual data in defined region

    Args:
        :param y_true: (tf.tensor) true distribution of data
        :param y_pred: (tf.tensor) predicted distribution of data

    Returns:
        Ad hoc measure of accuracy of predicted data distribution w.r.t true data distribution
    """

    real_y_max = tf.argmax(y_true, dimension = 1)
    predicted_y_max = tf.argmax(y_pred, dimension = 1)
    average_peak_dist = tf.reduce_mean(tf.cast(tf.abs(real_y_max - predicted_y_max), tf.float32))
    return average_peak_dist
