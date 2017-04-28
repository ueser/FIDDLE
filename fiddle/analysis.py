"""Author: Umut Eser
Documentation: Dylan Marshall

'analysis.py' interfaces with output from main.py to create representations and
predictions hdf5 datasets and other interpretable mediums.

Example:
    Assuming instructions for directory setup and file installation on the
    github page have been followed, default flags and the following command will
    output the datasets to a results directory "FIDDLE/results/experiment/"

        $ python analysis.py

FLAGS:
    flag:                   default:                description:

    --runName               'experiment'            name of run
    --resultsDir            '../results'            directory where results from runName will be stored
    --dataDir               '../data/hdf5datasets'  directory where hdf5datasets are stored
    --saveDataForLater      True                    save results as hdf5 format to use later
    --configuration         'configurations.json'   parameters of data inputs and outputs [json file]

Todo:
    incorporate t-SNE, PCA, filter visualizations
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pdb, traceback, sys #
import tensorflow as tf
import numpy as np
import h5py
from tqdm import tqdm as tq
import cPickle as pickle
#from auxilary import * # not sure what for?
import pandas as pd
import os

### FIDDLE specific tools ###
from models import *
#############################

flags = tf.app.flags
flags.DEFINE_string('runName', 'experiment', 'name of run')
flags.DEFINE_string('resultsDir', '../results', 'directory where results from runName will be stored')
flags.DEFINE_string('dataDir', '../data/hdf5datasets', 'directory where hdf5datasets are stored')
flags.DEFINE_boolean('saveDataForLater', True, 'save results as hdf5 format to use later')
flags.DEFINE_string('configuration', 'configurations.json', 'parameters of data inputs and outputs [json file]')
FLAGS = flags.FLAGS

def main(_):
    """Read in data, get representations, get predictions, output to hdf5"""

    # read in data and configurations
    project_directory = os.path.join(FLAGS.resultsDir, FLAGS.runName)
    with open(os.path.join(project_directory, 'configuration.json')) as fp:
        config = byteify(json.load(fp))
    test_h5_handle = h5py.File(os.path.join(FLAGS.dataDir, config['Options']['DataName'], 'test.h5'), 'r')
    model = NNscaffold(config = config, architecture_path = os.path.join(project_directory, 'architecture.json'), model_path = project_directory)
    model.config['Options']['Reload'] = 'all'
    test_data = {key: test_h5_handle[key][:1000] for key in model.inputs}
    model.initialize()

    # get representations for each tracks and scaffolds.
    reprDict = model.get_representations(test_data)
    if FLAGS.saveDataForLater:
        repr_h5_handle = h5py.File(os.path.join(project_directory, 'representations.h5'),'w')
        for key, val in reprDict.items():
            f_ = repr_h5_handle.create_dataset(key, (reprDict[key].shape))
            f_[:] = reprDict[key][:]
        repr_h5_handle.close()

        #TODO: 2.dimensionality reduction and visualization (t-SNE, PCA etc.)

    # get predictions
    pred_dict = model.predict(test_data)
    if FLAGS.saveDataForLater:
        pred_h5_handle = h5py.File(os.path.join(project_directory, 'predictions.h5'), 'w')
        for key, val in pred_dict.items():
            f_ = pred_h5_handle.create_dataset(key, (pred_dict[key].shape))
            f_[:] = pred_dict[key][:]
        pred_h5_handle.close()
    #TODO: filter visualization
    model.sess.close()

def blah(self, predict_data):
    """
    """
    #TODO: what is this?
    pred_feed = {}
    pred_feed.update({ self.inputs[key]: predict_data[key] for key in self.architecture['Inputs'] })
    pred_feed.update({self.common_predictor.all_gates: np.ones((predict_data.values()[0].shape[0], len(self.architecture['Inputs']))) + 0.})
    pred_feed.update({
        self.dropout: 1.,
        self.keep_prob_input: 1.,
        self.inp_size: predict_data.values()[0].shape[0],
        K.learning_phase(): 0
    })
    PREDICTION_FETCHES.update({key: val for key, val in self.common_predictor.predictions.items()})
    return_dict = self._run(PREDICTION_FETCHES, pred_feed)
    return return_dict


if __name__ == '__main__':
    try:
        tf.app.run()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
