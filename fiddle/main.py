from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pdb, traceback, sys #
import tensorflow as tf
import numpy as np
from tqdm import tqdm as tq
from models import *
from io_tools import *
import six
from collections import Counter
import os
import h5py
import json

#testing

flags = tf.app.flags
flags.DEFINE_string('runName', 'experiment', 'Running name.')
flags.DEFINE_string('dataDir', '../data/hdf5datasets/CN2TS_500bp', 'Regions to train [bed or gff files]')
flags.DEFINE_string('configuration', 'configuration.json', 'configuration file [json file]')
flags.DEFINE_string('architecture', 'architecture.json', 'configuration file [json file]')
flags.DEFINE_boolean('predict', False, 'If true, tests for the data and prints statistics about data for unit testing.')
flags.DEFINE_boolean('restore', False, 'If true, restores models from the ../results/XXtrained/')
flags.DEFINE_string('restorePath', '../results/test', 'Regions to validate [bed or gff files]')
flags.DEFINE_integer('maxEpoch', 1000, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batchSize', 100, 'Batch size.')
flags.DEFINE_float('learningRate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.5, 'Keep probability for training dropout.')
flags.DEFINE_string('resultsDir', '../results', 'Directory for results data')
FLAGS = flags.FLAGS

################debugger#####################
# from tensorflow.python import debug as tf_debug
################debugger#####################

def main(_):
    train_h5_handle  = h5py.File(os.path.join(FLAGS.dataDir, 'train.h5'),'r')
    validation_h5_handle  = h5py.File(os.path.join(FLAGS.dataDir, 'validation.h5'),'r')

    FLAGS.savePath = FLAGS.resultsDir + '/' + FLAGS.runName
    if not tf.gfile.Exists(FLAGS.savePath):
        print('Results will be saved in ' + str(FLAGS.savePath))
        tf.gfile.MakeDirs(FLAGS.savePath)



    model = NNscaffold(configuration_path=FLAGS.configuration,
                       architecture_path=FLAGS.architecture,
                       learning_rate=FLAGS.learningRate)
    json.dump(model.architecture, open(FLAGS.savePath + "/architecture.json", 'w'))


    all_keys = list(set(model.architecture['Inputs'] + model.architecture['Outputs']))

    data = MultiModalData(train_h5_handle, batch_size=FLAGS.batchSize)
    batcher = data.batcher()
    print('Storing validation data to the memory')
    try:
        validation_data = {key: validation_h5_handle[key][:] for key in all_keys}
    except KeyError:
        print('Make sure that the configuration file contains the correct track names (keys), '
              'which should match the hdf5 keys')
    if FLAGS.restore:
        model.load(FLAGS.restorePath)
    else:
        model.initialize()

    print('Saving to results directory: ' + str(FLAGS.savePath))
    model.create_monitor_variables(FLAGS.savePath)

    ####################
    # Launch the graph #
    ####################

    print('Launch the graph')
    header_str = 'Loss'
    for key in model.architecture['Outputs']:
        header_str += '\t' + key + '_Accuracy'
    header_str += '\n'

    saver = tf.train.Saver()
    with open((FLAGS.savePath + "/" + "train.txt"), "w") as train_file:
        train_file.write(header_str)

    with open((FLAGS.savePath + "/" + "validation.txt"), "w") as validation_file:
        validation_file.write(header_str)

    print('Pre-train validation run:')
    return_dict = model.validate(validation_data, accuracy=True)
    print("Pre-train validation loss: " + str(return_dict['cost']))
    print("Pre-train validation accuracy (%): " + str(100. * return_dict['accuracy_' + key] / validation_data.values()[0].shape[0]))
    model.profile() # what does this do?

    # totIteration = int(len(train_regions) / FLAGS.batchSize) # size of train_regions needs fixing, probably valid size as well
    globalMinLoss = np.inf
    step = 0
#for it in range(FLAGS.maxEpoch * totIteration): # EDIT: change back
    for it in range(200):

        print('Iteration: ' + str(it))
        # ido_ = 0.8 + 0.2 * it / 10. if it <= 10 else 1.
        ido_=1.
        return_dict_train = Counter({})
        t_batcher, t_trainer = 0, 0
        for iterationNo in tq(range(10)):
            with Timer() as t:
                train_batch = batcher.next()
            t_batcher += t.secs
            with Timer() as t:
                return_dict = Counter(model.train(train_batch, accuracy=True, inp_dropout=ido_, batch_size=FLAGS.batchSize))
            t_trainer += t.secs

            return_dict_train += return_dict
            step += 1
        print('Batcher time: ' + str(t_batcher))
        print('Trainer time: ' + str(t_trainer))
        for key in return_dict_train.keys():
            return_dict_train[key] /= iterationNo
        return_dict_valid = model.validate(validation_data, accuracy=True)


        write_to_txt(return_dict_train)
        write_to_txt(return_dict_valid, batch_size=validation_data.values()[0].shape[0], case='validation')

        model.summarize(step)

        if return_dict_valid['cost'] < globalMinLoss:
            globalMinLoss = return_dict_valid['cost']
            save_path = saver.save(model.sess, FLAGS.savePath + "/model.ckpt")
            print("Model saved in file: %s" % FLAGS.savePath)

    model.sess.close()



def write_to_txt(return_dict, batch_size=FLAGS.batchSize, case='train', verbose=True):
    save_path = FLAGS.resultsDir + '/' + FLAGS.runName
    line_to_write = ''
    for key, val in return_dict.items():
        if key == 'cost':
            cur_line = str(return_dict['cost'])
            line_to_write += str(return_dict[key])
        elif key == '_':
            continue
        else:
            cur_line = str(100. * return_dict[key] / batch_size)
            line_to_write += '\t' + cur_line
        if verbose:
            print(case + '\t' + key + ': ' + cur_line)

    with open((save_path + "/" + case + ".txt"), "a") as fp:
        fp.write(line_to_write + '\n')


if __name__ == '__main__':
    try:
        tf.app.run()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
