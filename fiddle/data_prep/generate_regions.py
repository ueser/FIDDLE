import numpy as np
from tqdm import tqdm as tq
import sys
sys.path.append('/Users/umut/Projects/ClassifySpecies/analysis/')
from genericFunctions import *
from optparse import OptionParser

usage = 'usage: %prog <out_file_name>'
parser = OptionParser(usage)
parser.add_option('-e', dest='width', type='int', default=500, help='Extend all sequences to this length [Default: %default]')
parser.add_option('-r', dest='stride', default=20, type='int', help='Stride sequences [Default: %default]')

(options,args) = parser.parse_args()
f_name = args[0]

print('Reading annotation...')
annotSC=getAnnotation('YJ168')

print('Saving regions for plus strand')

starts = np.array([annotSC[annotSC['strand']=='+']['tss'].values-ix for ix in range(0,500,50)]).flatten()
chroms = np.array([annotSC[annotSC['strand']=='+']['chr'].values for ix in range(0,500,50)]).flatten()
bin_size = 500
step_size = 500
with open(f_name + '.pos.bed','w') as fp:
    for start, ch in tq(zip(starts, chroms)):
        chname = 'chr'+toRoman(ch)
        fp.write('{0}\t{1}\t{2}\n'.format(chname, int(start), int(start+bin_size)))


print('Saving regions for minus strand')

starts = np.array([annotSC[annotSC['strand']=='-']['tss'].values-ix for ix in range(0,500,50)]).flatten()
chroms = np.array([annotSC[annotSC['strand']=='-']['chr'].values for ix in range(0,500,50)]).flatten()

bin_size = 500
step_size = 500
with open(f_name + '.neg.bed','w') as fp:
    for start, ch in tq(zip(starts, chroms)):
        chname = 'chr'+toRoman(ch)
        fp.write('{0}\t{1}\t{2}\n'.format(chname, int(start-bin_size), int(start)))

