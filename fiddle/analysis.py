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
import os
# from matplotlib import pylab as pl



flags = tf.app.flags
flags.DEFINE_string('runName', 'experiment', 'Running name.')
flags.DEFINE_string('resultsDir', '../results', 'Directory for results data')
flags.DEFINE_string('dataDir', '../data/hdf5datasets', 'Default data directory')

FLAGS = flags.FLAGS
config.FLAGS=FLAGS




def main(_):
    '''
    This code imports NNscaffold class from models.py module for training, testing etc.

    Usage with defaults:
    python analysis.py (options)

    '''
    project_directory = os.path.join(FLAGS.resultsDir, FLAGS.runName)


    with open(os.path.join(project_directory, 'configuration.json')) as fp:
        config = byteify(json.load(fp))

    test_h5_handle = h5py.File(os.path.join(FLAGS.dataDir, config['Options']['DataName'], 'test.h5'), 'r')

    model = NNscaffold(config=config,
                       architecture_path=os.path.join(project_directory, 'architecture.json'))

    model.load(load_list=['chipnexus','dnaseq'])
    model.freeze()




if __name__ == '__main__':
    tf.app.run()
