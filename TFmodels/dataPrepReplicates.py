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



    TSdata = f.create_dataset("TSSseq1", (dataSize,1,options.width))
    TSdata2 = f.create_dataset("TSSseq2", (dataSize,1,options.width))
    TSdata3 = f.create_dataset("TSSseq3", (dataSize,1,options.width))
    # PCdata = f.create_dataset("PromoterClass", (dataSize,1,options.width))
    infodata = f.create_dataset("info", (dataSize,4))


    chrRange = annot['chr'].unique()

    TStmp = np.zeros([options.chunkSize,1,options.width])
    TStmp2 = np.zeros([options.chunkSize,1,options.width])
    TStmp3 = np.zeros([options.chunkSize,1,options.width])

    infotmp = np.zeros([options.chunkSize,4])

    print assembly
    gdb = genome.db.GenomeDB(path='/Users/umut/Projects/genome/data/share/genome_db',assembly=assembly)


    TSpos = gdb.open_track('TSpos')
    TSneg = gdb.open_track('TSneg')

    TSpos2 = gdb.open_track('TSpos2')
    TSneg2 = gdb.open_track('TSneg2')

    TSpos3 = gdb.open_track('TSpos3')
    TSneg3 = gdb.open_track('TSneg3')


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


                tsP = TSpos.get_nparray(chname,pos+1,(pos+options.width))
                tsN = TSneg.get_nparray(chname,pos+1,(pos+options.width))

                tsP2 = TSpos2.get_nparray(chname,pos+1,(pos+options.width))
                tsN2 = TSneg2.get_nparray(chname,pos+1,(pos+options.width))

                tsP3 = TSpos3.get_nparray(chname,pos+1,(pos+options.width))
                tsN3 = TSneg3.get_nparray(chname,pos+1,(pos+options.width))


                if annot[tf].strand.iloc[i]=="+":

                    if sum(tsP)==0:
                        tsP = tsP + 1/np.float(options.width)
                    else:
                        tsP = tsP/sum(tsP+1e-5)
                    if sum(tsP2)==0:
                        tsP2 = tsP2 + 1/np.float(options.width)
                    else:
                        tsP2 = tsP2/sum(tsP2+1e-5)
                    if sum(tsP3)==0:
                        tsP3 = tsP3 + 1/np.float(options.width)
                    else:
                        tsP3 = tsP3/sum(tsP3+1e-5)

                    TStmp[qq,0,:] =tsP.T
                    TStmp2[qq,0,:] =tsP2.T
                    TStmp3[qq,0,:] =tsP3.T
                    infotmp[qq,:] = [cc, 1,annotIdx,pos]
                else:

                    if sum(tsN)==0:
                        tsN = tsN + 1/np.float(options.width)
                    else:
                        tsN = tsN/sum(tsN+1e-5)
                    if sum(tsN2)==0:
                        tsN2 = tsN2 + 1/np.float(options.width)
                    else:
                        tsN2 = tsN2/sum(tsN2+1e-5)
                    if sum(tsN3)==0:
                        tsN3 = tsN3 + 1/np.float(options.width)
                    else:
                        tsN3 = tsN3/sum(tsN3+1e-5)
                    TStmp[qq,0,:] =np.flipud(tsN).T
                    TStmp2[qq,0,:] =np.flipud(tsN2).T
                    TStmp3[qq,0,:] =np.flipud(tsN3).T
                    infotmp[qq,:] = [cc, -1,annotIdx,pos]

                qq+=1

                if (qq>=options.chunkSize):
                    stp = options.chunkSize

                    TSdata[range(ps,ps+stp),:,:] = TStmp
                    TSdata2[range(ps,ps+stp),:,:] = TStmp2
                    TSdata3[range(ps,ps+stp),:,:] = TStmp3
                    infodata[range(ps,ps+stp),:] = infotmp
                    infotmp = np.zeros([options.chunkSize,4])
                    ps+=stp
                    qq=0
                    print >> sys.stderr, '%d  data chunk saved ' % ps

                if ps >=(dataSize):
                    nestVar = 1;
                    break

    print ps

    f.close()

    TSpos.close()
    TSneg.close()
    TSpos2.close()
    TSneg2.close()
    TSpos3.close()
    TSneg3.close()

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

def verifyHDF5():
    f = h5py.File(os.path.join(directory,out_file), "w")
    for dataName in inputList:
        tmp = f.get(dataName)
        assert(tmp[-1].sum() >0,'Last entries are null')

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
