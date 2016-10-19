#!/usr/bin/env python
import os
import sys
sys.path.append('/Users/umut/Projects/genome/python/lib')
import genome.db
from optparse import OptionParser
import h5py
import pandas as pd
import numpy as np
from numpy import linalg as LA
from scipy.stats import norm
import numpy.random as npr


################################################################################
# data4training.py
#
# Make an HDF5 file for Torch input using a pyTable wrapper called genome: https://github.com/gmcvicker/genome
#
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <assembly> <annotation_file> <out_file>'
    parser = OptionParser(usage)
    parser.add_option('-b', dest='chunkSize', default=1000, type='int', help='Align sizes with batch size')
    parser.add_option('-e', dest='width', type='int', default=500, help='Extend all sequences to this length [Default: %default]')
    parser.add_option('-r', dest='stride', default=20, type='int', help='Stride sequences [Default: %default]')
    parser.add_option('-u', dest='upstream', default=500, type='int', help='Stride sequences [Default: %default]')
    parser.add_option('-d', dest='downstream', default=500, type='int', help='Stride sequences [Default: %default]')

    (options,args) = parser.parse_args()

    if len(args) !=3 :
        print(args)
        print(options)
        print(len(args))
        parser.error('Must provide assembly, annotation file and an output name')
    else:
        assembly = args[0]
        annot_file = args[1]
        out_file = args[2]

    # read in the annotation file
    annot = pd.read_table(annot_file,sep=',')

    # Make directory for the project
    directory = "../data/hdf5datasets/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    x1 = options.upstream
    x2 = options.downstream

    # open the hdf5 file to write
    f = h5py.File(os.path.join(directory,out_file), "w")
    dataSize = np.floor(len(annot)*(x1+x2-options.width)/options.stride)
    dataSize -= dataSize % options.chunkSize

    print >> sys.stderr, '%d data size\n' % (dataSize)

    inputList = ['DNAseq','NETseq','ChIPseq','MNaseseq','RNAseq','TSSseq']

    # note that we have 1 channel and 4xoptions.width matrices for dna sequence.
    NSdata = f.create_dataset("NETseq", (dataSize,2,options.width))
    MSdata = f.create_dataset("MNaseseq", (dataSize,2,options.width))
    DSdata = f.create_dataset("DNAseq", (dataSize,4,options.width))
    RSdata = f.create_dataset("RNAseq", (dataSize,1,options.width))
    CSdata = f.create_dataset("ChIPseq", (dataSize,2,options.width))
    TSdata = f.create_dataset("TSSseq", (dataSize,2,options.width))
    # PCdata = f.create_dataset("PromoterClass", (dataSize,1,options.width))
    infodata = f.create_dataset("info", (dataSize,4))


    chrRange = annot['chr'].unique()
    # Use 4 channel and 1xoptions.width
    NStmp = np.zeros([options.chunkSize,2,options.width])
    MStmp = np.zeros([options.chunkSize,2,options.width])
    DStmp = np.zeros([options.chunkSize,4,options.width])
    RStmp = np.zeros([options.chunkSize,1,options.width])
    CStmp = np.zeros([options.chunkSize,2,options.width])
    TStmp = np.zeros([options.chunkSize,2,options.width])
    # PCtmp = np.zeros([options.chunkSize,1,options.width])

    infotmp = np.zeros([options.chunkSize,4])

    print assembly
    gdb = genome.db.GenomeDB(path='/Users/umut/Projects/genome/data/share/genome_db',assembly=assembly)

    NSpos = gdb.open_track('NSpos')
    NSneg = gdb.open_track('NSneg')
    MSpos = gdb.open_track('MSpos')
    MSneg = gdb.open_track('MSneg')
    CSpos = gdb.open_track('TFpos')
    CSneg = gdb.open_track('TFneg')
    RS = gdb.open_track('RS')
    TSpos = gdb.open_track('TSpos')
    TSneg = gdb.open_track('TSneg')

    seq = gdb.open_track("seq")

    qq=0;
    cc=0;
    ps =0;
    nestVar = 0;
    debugMode = True
    for chname in chrRange:
        if nestVar:
                break
        cc +=1
        tf = annot.chr==chname
        print 'doing %s' % (chname)

        for i in range(sum(tf)):
            if nestVar:
                break
            tss = annot[tf].tss.iloc[i]

            if annot[tf].strand.iloc[i]=="-":
                xran = range(tss-x2,tss+x1-options.width,options.stride)

            else:
                if tss<1000:
                    continue
                xran = range(tss-x1,tss+x2-options.width,options.stride)

            annotIdx = annot[tf].index[i]

            for pos in xran:
                if nestVar:
                    break
                seqVec = seq.get_seq_str(chname,pos+1,(pos+options.width))
                dsdata = vectorizeSequence(seqVec.lower())

                nsP = NSpos.get_nparray(chname,pos+1,(pos+options.width))
                nsN = NSneg.get_nparray(chname,pos+1,(pos+options.width))
                msP = MSpos.get_nparray(chname,pos+1,(pos+options.width))
                msN = MSneg.get_nparray(chname,pos+1,(pos+options.width))
                tfP = CSpos.get_nparray(chname,pos+1,(pos+options.width))
                tfN = CSneg.get_nparray(chname,pos+1,(pos+options.width))
                rs = RS.get_nparray(chname,pos+1,(pos+options.width))
                tsP = TSpos.get_nparray(chname,pos+1,(pos+options.width))
                tsN = TSneg.get_nparray(chname,pos+1,(pos+options.width))


                if debugMode:
                    if not checkData(np.r_[nsP,nsN,msP,msN,rs,tsP,tsN,tfP,tfN]):
                        print('NaN detected in chr' + chname + ' and at the position:' + str(pos))
                        # print nsmstsrsdata
                        # nestVar = 1;
                        continue

                if annot[tf].strand.iloc[i]=="+":
                    NStmp[qq,0,:] = nsP.T
                    NStmp[qq,1,:] = nsN.T
                    MStmp[qq,0,:] =msP.T
                    MStmp[qq,1,:] = msN.T
                    DStmp[qq,:,:] = dsdata.T
                    RStmp[qq,0,:] = rs.T
                    CStmp[qq,0,:] =tfP.T
                    CStmp[qq,1,:] = tfN.T
                    TStmp[qq,0,:] =tsP.T
                    TStmp[qq,1,:] =tsN.T

                    infotmp[qq,:] = [cc, 1,annotIdx,pos]
                else:
                    NStmp[qq,0,:] = np.flipud(nsN).T
                    NStmp[qq,1,:] = np.flipud(nsP).T
                    MStmp[qq,0,:] = np.flipud(msN).T
                    MStmp[qq,1,:] = np.flipud(msP).T
                    RStmp[qq,0,:] = np.flipud(rs).T
                    DStmp[qq,:,:] = np.flipud(np.fliplr(dsdata)).T
                    CStmp[qq,0,:] = np.flipud(tfN).T
                    CStmp[qq,1,:] = np.flipud(tfP).T
                    TStmp[qq,0,:] =np.flipud(tsN).T
                    TStmp[qq,1,:] =np.flipud(tsP).T
                    infotmp[qq,:] = [cc, -1,annotIdx,pos]

                qq+=1

                if (qq>=options.chunkSize):
                    stp = options.chunkSize
                    NSdata[range(ps,ps+stp),:,:] = NStmp
                    MSdata[range(ps,ps+stp),:,:] = MStmp
                    DSdata[range(ps,ps+stp),:,:] = DStmp
                    RSdata[range(ps,ps+stp),:,:] = RStmp
                    CSdata[range(ps,ps+stp),:,:] = CStmp
                    TSdata[range(ps,ps+stp),:,:] = TStmp
                    infodata[range(ps,ps+stp),:] = infotmp
                    NStmp= np.zeros([options.chunkSize,2,options.width])
                    MStmp = np.zeros([options.chunkSize,2,options.width])
                    DStmp = np.zeros([options.chunkSize,4,options.width])
                    RStmp = np.zeros([options.chunkSize,1,options.width])
                    TStmp = np.zeros([options.chunkSize,2,options.width])
                    CStmp = np.zeros([options.chunkSize,2,options.width])

                    infotmp = np.zeros([options.chunkSize,4])
                    ps+=stp
                    qq=0
                    print >> sys.stderr, '%d  data chunk saved ' % ps

                if ps >=(dataSize):
                    nestVar = 1;
                    break

    print('Total number of samples: '+str(ps))

    f.close()
    NSpos.close()
    NSneg.close()
    MSpos.close()
    MSneg.close()
    RS.close()
    CSpos.close()
    CSneg.close()
    TSpos.close()
    TSneg.close()
    seq.close()
    
    print('Verifying the saved database...')
    verifyHDF5(inputList,directory,out_file)
    print('Passed...')
def vectorizeSequence(seq):
    # the order of the letters is not arbitrary.
    # Flip the matrix up-down and left-right for reverse compliment
    ltrdict = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1],'n':[0.25,0.25,0.25,0.25]}
    return np.array([ltrdict[x] for x in seq])


def normalizeData(data):
    data = np.subtract(data,np.mean(data,axis=0))
    l2norm = LA.norm(data,axis=0)+ abs(7./3 - 4./3 -1)
    return np.divide(data,l2norm)

def checkData(Xdata):
    if np.sum(np.isnan(Xdata).flatten())>0:
        return False
    else:
        return True

def verifyHDF5(inputList,directory,out_file):
    f = h5py.File(os.path.join(directory,out_file), "r")
    for dataName in inputList:
        tmp = f.get(dataName)
        assert tmp[-1].sum() >0,'Last entries are null'
    f.close()
################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
