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


flags = tf.app.flags
flags.DEFINE_string('runName', 'experiment', 'Running name.')
flags.DEFINE_string('chromSizes', '../data/sacCer3.chrom.sizes', 'Chromosome sizes files from UCSC.')
flags.DEFINE_string('trainRegions', '../data/regions/train_regions.bed', 'Regions to train [bed or gff files]')
flags.DEFINE_string('validRegions', '../data/regions/validation_regions.bed', 'Regions to validate [bed or gff files]')
flags.DEFINE_string('dataDir', '../data/hdf5datasets/CN2TS_500bp', 'Regions to train [bed or gff files]')
flags.DEFINE_string('configuration', 'configuration.json', 'configuration file [json file]')
flags.DEFINE_string('architecture', 'architecture.json', 'configuration file [json file]')
flags.DEFINE_boolean('predict', False, 'If true, tests for the data and prints statistics about data for unit testing.')
flags.DEFINE_boolean('restore', False, 'If true, restores models from the ../results/XXtrained/')
flags.DEFINE_string('restorePath', '../results/test', 'Regions to validate [bed or gff files]')
flags.DEFINE_integer('maxEpoch', 1000, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batchSize', 100, 'Batch size.')
flags.DEFINE_integer('filterWidth', 10, 'Filter width for convolutional network')
flags.DEFINE_integer('sampleSize', None, 'Sample size.')
flags.DEFINE_integer('testSize', None, 'Test size.')
flags.DEFINE_integer('trainSize', None, 'Train size.')
flags.DEFINE_boolean('overwrite', False, 'Overwrite the mmap files?')
flags.DEFINE_integer('chunkSize', 1000, 'Chunk size.')
flags.DEFINE_float('learningRate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('trainRatio', 0.8, 'Train data ratio')
flags.DEFINE_float('dropout', 0.5, 'Keep probability for training dropout.')
flags.DEFINE_string('resultsDir', '../results', 'Directory for results data')
FLAGS = flags.FLAGS

################debugger#####################
# from tensorflow.python import debug as tf_debug
################debugger#####################

def main(_):
    train_h5_handle  = h5py.File(os.path.join(FLAGS.dataDir, 'train.h5'),'r')
    validation_h5_handle  = h5py.File(os.path.join(FLAGS.dataDir, 'validation.h5'),'r')
    # Initialize results directory
    FLAGS.savePath = FLAGS.resultsDir + '/' + FLAGS.runName
    if not tf.gfile.Exists(FLAGS.savePath):
        tf.gfile.MakeDirs(FLAGS.savePath)

    model = NNscaffold(configuration_path=FLAGS.configuration, architecture_path=FLAGS.architecture, learning_rate=FLAGS.learningRate)
    json.dump(model.architecture, open(FLAGS.savePath + "/architecture.json", 'w'))

    #####################################
    # Train region and data definition #
    #####################################

    print('Creating multithread runner data object')
    data = MultiThreadRunner(train_h5_handle, model.inputs, model.outputs)

    all_keys = list(set(model.architecture['Inputs'] + model.architecture['Outputs']))
    print('Storing validation data to the memory')
    validation_data = {key: validation_h5_handle[key][:1000].reshape(1000,
                                                                     validation_h5_handle[key].shape[1],
                                                                     validation_h5_handle[key].shape[2],
                                                                     1) for key in all_keys}

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
    with open((FLAGS.savePath + "/" + "train.txt"), "w") as trainFile:
        trainFile.write(header_str)

    with open((FLAGS.savePath + "/" + "test.txt"), "w") as testFile:
        testFile.write(header_str)

    print('Pre-train test run:')
    return_dict = model.validate(validation_data, accuracy=True)
    print("Pre-train test loss: " + str(return_dict['cost']))
    print("Pre-train test accuracy (%): " + str(100. * return_dict['accuracy_' + key] / validation_data.values()[0].shape[0]))
    model.profile() # what does this do?

    # totIteration = int(len(train_regions) / FLAGS.batchSize) # size of train_regions needs fixing, probably valid size as well
    globalMinLoss = np.inf
    step = 0
#for it in range(FLAGS.maxEpoch * totIteration): # EDIT: change back
    data.start_threads(model.sess, n_threads=4)
    for it in range(20):
        print("it = " + str(it))
        ido_ = 0.8 + 0.2 * it / 10. if it <= 10 else 1.
        return_dict_train = Counter({})
        t_batcher, t_trainer = 0, 0
        for iterationNo in tq(range(10)):
            with Timer() as t:
                train_batch = data.get_batch()
                t_batcher += t.secs
            with Timer() as t:
                return_dict = Counter(model.train(train_batch, accuracy=True, inp_dropout=ido_))
                t_trainer += t.secs

            return_dict_train += return_dict
            step += 1
        print('Batcher time: ' + str(t_batcher))
        print('Trainer time: ' + str(t_trainer))
        for key in return_dict_train.keys():
            return_dict_train[key] /= iterationNo
        return_dict_valid = model.validate(validation_data, accuracy=True)

        print('Iteration: ' + str(it))
        write_to_txt(return_dict_train)
        write_to_txt(return_dict_valid, batch_size=len(valid_regions), case='validation')

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
            print(return_dict['cost'])
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



#flags.DEFINE_boolean('mmap', False, 'If true, Creates memory mapped objects for streamed training')

#    # 3) Construct data_tracks dictionary along with concurrent memory map or hdf5
# memory mapping to be tossed
#    data_tracks = {}
#    if FLAGS.mmap:
#        print('3) Extracting memory mapped metadata')
#        for key, vals in six.iteritems(config['Tracks']):
#            data_tracks[key] = {}
#            data_tracks[key]['path_to_mmap'] = vals['data_dir']
#            data_tracks[key]['metadata'] = {}
#            filetype = vals['orig_files']['type']
#            if filetype == 'fasta':
#                data_tracks[key]['caller'] = extract_fasta_to_file(vals['orig_files']['pos'], vals['data_dir'], FLAGS.overwrite)
#                data_tracks[key]['metadata']['input_height'] = 4
#            elif filetype == 'bigwig':
#                if 'neg' in vals['orig_files'].keys():
#                    data_tracks[key]['caller'] = extract_bigwig_to_file([vals['orig_files']['pos'], vals['orig_files']['neg']],
#                                                          vals['data_dir'], chrom_sizes, FLAGS.overwrite)
#                    data_tracks[key]['metadata']['input_height'] = 2
#                elif 'pos' in vals['orig_files'].keys():
#                    data_tracks[key]['caller'] = extract_bigwig_to_file([vals['orig_files']['pos']],
#                                                              vals['data_dir'], chrom_sizes, FLAGS.overwrite)
#                    data_tracks[key]['metadata']['input_height'] = 1
#            else:
#                raise NotImplementedError
