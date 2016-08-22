'''
This script contains the functions for post-training analysis
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import h5py # alternatively, tables module can be used
from tqdm import tqdm as tq

import config
from dataClass import *
from models import *


def getModel():

def probeParams(layer=0,representation='heatmap'):
    
