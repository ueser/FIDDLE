import numpy as np
from tqdm import tqdm as tq
import sys
sys.path.append('/Users/umut/Projects/ClassifySpecies/analysis/')
from genericFunctions import *
from optparse import OptionParser
import h5py

def one_hot_encode_sequence(seq):
    # the order of the letters is not arbitrary.
    # Flip the matrix up-down and left-right for reverse compliment
    seq = seq.lower()
    letterdict = {'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1],
               'n': [0.25, 0.25, 0.25, 0.25]}
    result = np.array([letterdict[x] for x in seq])
    return result.T

def get_fasta(file_path):
    seqs=[]
    with open(file_path, 'r') as fr:
        while True:
            line = fr.readline()
            if '>' in line:
                seqs.append(fr.readline().split('\n')[0])
            if line=='':
                break
    return seqs




seqs = get_fasta('sense.fa')
seqs_array = np.array(map(one_hot_encode_sequence, seqs))
smpl = 'Dia_Cnt'
print('Reading ' + smpl + ' sense')
tmp_ts = np.genfromtxt('Dia_Cnt.ts.sense_asense.txt')
tmp_cn = np.genfromtxt('Dia_Cnt.cn.sense_asense.txt')

tf1 = np.sum(np.isnan(tmp_ts),axis=1)>0
tf2 = np.sum(np.isnan(tmp_cn),axis=1)>0
tf3 = ~(tf1|tf2)
tssseq_se = tmp_ts[tf3, :500]
tssseq_as = tmp_ts[tf3, 500:]
chipnexus_se = tmp_cn[tf3, :500]
chipnexus_as = tmp_cn[tf3, 500:]
dnaseq = seqs_array[tf3, :, :]

print('Reading ' + smpl + ' antisense')

seqs = get_fasta('asense_tbf.fa')
seqs_array = np.array(map(one_hot_encode_sequence, seqs))

tmp_ts = np.fliplr(np.genfromtxt('Dia_Cnt.ts.asense_sense_tbf.txt'))
tmp_cn = np.fliplr(np.genfromtxt('Dia_Cnt.cn.asense_sense_tbf.txt'))
idx = np.sort(np.unique(np.r_[np.where(np.sum(np.isnan(tmp_ts), axis=1) == 0)[0],
                              np.where(np.sum(np.isnan(tmp_cn), axis=1) == 0)[0]]))

tf1 = np.sum(np.isnan(tmp_ts),axis=1)>0
tf2 = np.sum(np.isnan(tmp_cn),axis=1)>0
tf3 = ~(tf1|tf2)

tssseq_se = np.r_[tssseq_se, tmp_ts[tf3, :500]]
tssseq_as = np.r_[tssseq_as, tmp_ts[tf3, 500:]]
chipnexus_se = np.r_[chipnexus_se, tmp_cn[tf3, :500]]
chipnexus_as = np.r_[chipnexus_as, tmp_cn[tf3, 500:]]

tmparr =seqs_array[tf3,:,:]
for ii in range(tmparr.shape[0]):
    tmparr[ii] = np.flipud(np.fliplr(tmparr[ii, :, :]))

dnaseq = np.r_[dnaseq, tmparr]

print('data were read...')


print('creating h5 files...')
validation_ratio = 0.05
test_ratio=0.1

idx = np.arange(tssseq_se.shape[0])
np.random.shuffle(idx)
validation_size = int(len(idx)*validation_ratio)
test_size = int(len(idx)*test_ratio)
train_size = len(idx)-validation_size-test_size

directory = '/Users/umut/Projects/FIDDLE/data/hdf5datasets/CN2TS_WT_500bp_tss/'
if not os.path.exists(directory):
    os.makedirs(directory)

train_h5 = h5py.File(os.path.join(directory,'train.h5'),'w')
validation_h5 = h5py.File(os.path.join(directory,'validation.h5'),'w')
test_h5 = h5py.File(os.path.join(directory,'test.h5'),'w')


print('creating h5 files... DNA sequence')

train=train_h5.create_dataset('dnaseq',((train_size,4, dnaseq.shape[2], 1)))
validation=validation_h5.create_dataset('dnaseq',((validation_size, 4, dnaseq.shape[2], 1)))
test=test_h5.create_dataset('dnaseq',((test_size, 4, dnaseq.shape[2], 1)))
train[:,:,:,0] = dnaseq[idx[:train_size]]
validation[:,:,:,0] = dnaseq[idx[train_size:(train_size+validation_size)]]
test[:,:,:,0] = dnaseq[idx[(train_size+validation_size):]]


print('creating h5 files... TSSseq')

train=train_h5.create_dataset('tssseq',((train_size, 2,tssseq_se.shape[1],1 )))
validation=validation_h5.create_dataset('tssseq',((validation_size, 2,tssseq_se.shape[1],1 )))
test=test_h5.create_dataset('tssseq',((test_size, 2,tssseq_se.shape[1],1 )))
train[:,0,:,0] = tssseq_se[idx[:train_size]]
train[:,1,:,0] = tssseq_as[idx[:train_size]]
validation[:,0,:,0] = tssseq_se[idx[train_size:(train_size+validation_size)]]
validation[:,1,:,0] = tssseq_as[idx[train_size:(train_size+validation_size)]]
test[:,0,:,0] = tssseq_se[idx[(train_size+validation_size):]]
test[:,1,:,0] = tssseq_as[idx[(train_size+validation_size):]]


print('creating h5 files... ChIPnexus')

train=train_h5.create_dataset('chipnexus',((train_size, 2,chipnexus_se.shape[1],1 )))
validation=validation_h5.create_dataset('chipnexus',((validation_size, 2,chipnexus_se.shape[1],1 )))
test=test_h5.create_dataset('chipnexus',((test_size, 2,chipnexus_se.shape[1],1 )))
train[:,0,:,0] = chipnexus_se[idx[:train_size]]
train[:,1,:,0] = chipnexus_as[idx[:train_size]]
validation[:,0,:,0] = chipnexus_se[idx[train_size:(train_size+validation_size)]]
validation[:,1,:,0] = chipnexus_as[idx[train_size:(train_size+validation_size)]]
test[:,0,:,0] = chipnexus_se[idx[(train_size+validation_size):]]
test[:,1,:,0] = chipnexus_as[idx[(train_size+validation_size):]]

train_h5.close()
validation_h5.close()
test_h5.close()
print('Done...')
