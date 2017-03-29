from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import h5py # alternatively, tables module can be used
from tqdm import tqdm as tq
import cPickle as pickle
from models import *
from auxilary import *
import pandas as pd
import os
# from matplotlib import pylab as pl



flags = tf.app.flags
flags.DEFINE_string('runName', 'experiment', 'Running name.')
flags.DEFINE_string('resultsDir', '../results', 'Directory for results data')
flags.DEFINE_string('dataDir', '../data/hdf5datasets', 'Default data directory')
flags.DEFINE_boolean('saveDataForLater', True, 'Save results as hdf5 format to use later')

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
    model.config['Options']['Reload'] = 'all'
    test_data = {key: test_h5_handle[key][:] for key in model.inputs}

    model.initialize()

    ### get representations for each tracks and scaffolds.
    repr_dict = model.get_representations(test_data)
    if FLAGS.saveDataForLater:
        repr_h5_handle = h5py.File(os.path.join(project_directory, 'representations.h5','w'))
        for key, val in repr_dict.items():
            f_ = repr_h5_handle.create_dataset(key, (repr_dict[key].shape))
            f_[:] = repr_dict[key][:]
        repr_h5_handle.close()

        #2.dimensionality reduction and visualization (t-SNE, PCA etc.)

    ### get predictions
    pred_dict = model.predict(test_data)
    if FLAGS.saveDataForLater:
        pred_h5_handle = h5py.File(os.path.join(project_directory, 'predictions.h5','w'))
        for key, val in pred_dict.items():
            f_ = pred_h5_handle.create_dataset(key, (pred_dict[key].shape))
            f_[:] = pred_dict[key][:]
        pred_h5_handle.close()

    ### filter visualization




if __name__ == '__main__':
    tf.app.run()
