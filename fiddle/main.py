from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pdb, traceback, sys #
import tensorflow as tf
import numpy as np
from tqdm import tqdm as tq
from models import *
from io_tools import *
import json
import six
from pybedtools import BedTool
from collections import Counter
import os
import copy
import h5py
from bx.bbi.bigwig_file import BigWigFile
from pysam import FastaFile

flags = tf.app.flags
flags.DEFINE_string('runName', 'experiment', 'Running name.')
flags.DEFINE_string('chromSizes', '../data/sacCer3.chrom.sizes', 'Chromosome sizes files from UCSC.')
flags.DEFINE_string('trainRegions', '../data/regions/train_regions.bed', 'Regions to train [bed or gff files]')
flags.DEFINE_string('validRegions', '../data/regions/validation_regions.bed', 'Regions to validate [bed or gff files]')
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
flags.DEFINE_string('dataDir', '../data', 'Directory for input data')
flags.DEFINE_string('resultsDir', '../results', 'Directory for results data')
FLAGS = flags.FLAGS

################debugger#####################
from tensorflow.python import debug as tf_debug
################debugger#####################

def main(_):

    ##################################################
    # Input data tracks and train regions definition #
    ##################################################

    # Read in chromosome size file to directory format
    print('Reading in chromosome sizes file')
    with open(FLAGS.chromSizes, 'r') as f:
        chrom_sizes = {line.split('\t')[0]: (1, int(line.split('\t')[-1].split('\n')[0])) for line in f.readlines() if line.split('\t')[0] != 'chrM'}

    # Read in configurations json file to dictionary format
    print('Inputting configurations file')
    with open('configurations_test.json') as fp:
        config = byteify(json.load(fp))
    data_tracks = {}
    for key, vals in six.iteritems(config['Tracks']):
        data_tracks[key] = {}
        data_tracks[key]['path_to_hdf5'] = vals['data_dir']
        data_tracks[key]['metadata'] = {}
        filetype = vals['orig_files']['type']
        if filetype == 'fasta':
            data_tracks[key]['caller'] = [FastaFile(vals['orig_files']['pos']).fetch]
            data_tracks[key]['metadata']['input_height'] = 4
        elif filetype == 'bigwig':
            if 'neg' in vals['orig_files'].keys():
                data_tracks[key]['caller'] = [BigWigFile(open(vals['orig_files']['pos'], 'r')).get_as_array,
                                              BigWigFile(open(vals['orig_files']['neg'], 'r')).get_as_array]
                data_tracks[key]['metadata']['input_height'] = 2
            else:
                data_tracks[key]['caller'] = [BigWigFile(open(vals['orig_files']['pos'], 'r')).get_as_array]
                data_tracks[key]['metadata']['input_height'] = 1
        else:
            raise NotImplementedError

    # Read in configurations json file to dictionary format
    print('Creating training and validation region objects')
    train_regions = BedTool(FLAGS.trainRegions).set_chromsizes(chrom_sizes)
    valid_regions = BedTool(FLAGS.validRegions).set_chromsizes(chrom_sizes)

    print('Preparing training and validation hdf5 data')
    output_dir = '../data/hdf5datasets/' + FLAGS.runName
    filename_train = 'train_' + FLAGS.runName
    if not os.path.isfile(os.path.join(output_dir, filename_train + '.h5')) or FLAGS.overwrite:
        save_for_fast_training_hdf5(output_dir, data_tracks, train_regions, filename_train)
    filename_validation = 'validation_' + FLAGS.runName
    if not os.path.isfile(os.path.join(output_dir, filename_validation + '.h5')) or FLAGS.overwrite:
        save_for_fast_training_hdf5(output_dir, data_tracks, valid_regions, filename_validation)

    train_h5_handle = h5py.File(os.path.join(output_dir, filename_train + '.h5'), 'r')
    validation_h5_handle = h5py.File(os.path.join(output_dir, filename_validation + '.h5'), 'r')

    #####################################
    # Architecture and model definition #
    #####################################

    # Read in the architecture parameters, defined by the user
    print('Constructing architecture and model definition')
    with open('architecture.json') as fp:
        architecture_template = byteify(json.load(fp))
    architecture = {'Modules': {}, 'Scaffold': {}}
    for key in data_tracks.keys():
        architecture['Modules'][key] = copy.deepcopy(architecture_template['Modules'])
        architecture['Modules'][key]['input_height'] = data_tracks[key]['metadata']['input_height']
        architecture['Modules'][key]['Layer1']['filter_height'] = architecture['Modules'][key]['input_height']
        # Parameters customized for specific tracks are read from architecture.json and updates the arch. dict.
        if key in architecture_template.keys():
            for key_key in architecture_template[key].keys():
                sub_val = architecture_template[key][key_key]
                if type(sub_val) == dict:
                    for key_key_key in sub_val:
                        architecture['Modules'][key][key_key][key_key_key] = sub_val[key_key_key]
                else:
                    architecture['Modules'][key][key_key] = sub_val
    architecture['Scaffold'] = architecture_template['Scaffold']
    architecture['Inputs'] = config['Options']['Inputs']
    architecture['Outputs'] = config['Options']['Outputs']

    # Initialize results directory
    FLAGS.savePath = FLAGS.resultsDir + '/' + FLAGS.runName
    if not tf.gfile.Exists(FLAGS.savePath):
        tf.gfile.MakeDirs(FLAGS.savePath)

    json.dump(architecture, open(FLAGS.savePath + "/architecture.json", 'w'))
    model = NNscaffold(architecture, FLAGS.learningRate)

    #####################################
    # Train region and data definition #
    #####################################

    print('Creating multithread runner data object')
    data = MultiThreadRunner(train_h5_handle, model.inputs, model.outputs, architecture)
    print('Setting batcher')
    batcher = data._batcher() # batcher = data.batcher(train_regions, batch_size=FLAGS.batchSize)
    print('Storing validation data to the memory')
    validation_data = {key: val[:] for key, val in validation_h5_handle.items()}

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
    for key in architecture['Outputs']:
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
    print("Pre-train test accuracy (%): " + str(100. * return_dict['accuracy_' + key] / len(valid_regions)))
    model.profile() # what does this do?

    totIteration = int(len(train_regions) / FLAGS.batchSize) # size of train_regions needs fixing, probably valid size as well
    globalMinLoss = np.inf
    step = 0
#for it in range(FLAGS.maxEpoch * totIteration): # EDIT: change back
    for it in range(20):
        print("it = " + str(it))
        ido_ = 0.8 + 0.2 * it / 10. if it <= 10 else 1.
        return_dict_train = Counter({})
        t_batcher, t_trainer = 0, 0
        for iterationNo in tq(range(10)):
            with Timer() as t:
                train_batch = next(batcher)
            t_batcher += t.secs
            with Timer() as t:
                return_dict = Counter(model.train(train_batch, accuracy=True, inp_dropout=ido_))
            t_trainer += t.secs
            # print(iterationNo, return_dict)
            # if np.isnan(return_dict['cost']):
            # print({key:val[:5,:,:,0].sum(axis=2) for key, val in train_batch.items()})
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
