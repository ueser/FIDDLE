
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import cPickle as pickle
import copy
import config


def get_network_architecture():
    """
    Builds the network architecture from config.FLAGS or from pre-trained network
    Input dictionary values should match with the hdf5 data track names
    """

    inputDict = {'NS':'NETseq',
                'MS':'MNaseseq',
                'RS':'RNAseq',
                'DS':'DNAseq',
                'CS':'ChIPseq',
                'TS':'TSSseq'}

    inputList=[]
    for kys in config.FLAGS.inputs.split('_'):
        inputList.append(inputDict[kys])

    outputList=[]
    for kys in config.FLAGS.outputs.split('_'):
        outputList.append(inputDict[kys])


    network_architecture={}
    if config.FLAGS.restore:

        network_architecture = pickle.load(open(config.FLAGS.savePath+'/network_architecture.pkl','rb'))
        # restore_dirs={inputDict[key]:'../results/'+key+'trained/' for key in config.FLAGS.inputs.split('_')}
        # for inputName in inputList:
        #     network_architecture.update( pickle.load(open(restore_dirs[inputName]+'network_architecture.pkl','rb')))

    else:
        restore_dirs = None
        inputHeights = {'DNAseq':4,'NETseq':2,'ChIPseq':2,'MNaseseq':2,'RNAseq':1}
        default_arch ={
        "inputShape": [4,500,1],
        "outputWidth": [500],
        "numberOfFilters":[80,80],
        "filterSize":[[2,5],[1,5]],
        "pool_size":[1,2],
        "pool_stride":[1,2],
        "FCwidth":1024,
        "dropout":0.5
        }
        for inputName in inputList:
            network_architecture[inputName] = copy.deepcopy(default_arch)
            network_architecture[inputName]['inputShape'][0]=inputHeights[inputName]
            network_architecture[inputName]['filterSize'][0][0]=inputHeights[inputName]

    return network_architecture, outputList
