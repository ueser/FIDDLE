from bx.bbi.bigwig_file import BigWigFile
import json
import os
import itertools
import numpy as np
import six
import bcolz
from pysam import FastaFile
import roman as rm
import pdb
#import pyBigWig
import numpy as np
import math

print('')
print('another test ...')
print('')

bigwigs = ['../data/bigwigs/netseq_pos.bw']
bw = [BigWigFile(open(bigwig, 'r')) for bigwig in bigwigs]
chrom = 'chrII'
size = (1, 813184)
data = np.empty(size[1])
print(data)
data = bw[0].get_as_array(chrom, 0, size[1])
print(type(data))
print(data)

numNan = 0
numNotNan = 0
for i in range(len(data)):
    value = data[i:i+1]
    if math.isnan(value):
        numNan += 1
    else:
        numNotNan += 1

print('numNan = ' + str(numNan))
print('numNotNan = ' + str(numNotNan))
print('num unique things = ' + str(len(list(set(data)))))
