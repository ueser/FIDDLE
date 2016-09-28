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
    parser.add_option('-d', dest='rootDir', type='str', default='.', help='Root directory of the project [Default: %default]')
    parser.add_option('-b', dest='chunkSize', default=1000, type='int', help='Align sizes with batch size')
    parser.add_option('-e', dest='width', type='int', default=500, help='Extend all sequences to this length [Default: %default]')
    parser.add_option('-r', dest='stride', default=20, type='int', help='Stride sequences [Default: %default]')

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
    directory = "../../data/hdf5datasets/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Parameters
    x1 = 500  #upstream
    x2 = 500  #downstream

    # open the hdf5 file to write
    f = h5py.File(os.path.join(directory,out_file), "w")
    trainSize = np.floor(0.90*len(annot)*(x1+x2-options.width)/options.stride)
    testSize = np.floor(0.05*len(annot)*(x1+x2-options.width)/options.stride)
    # make sure that the sizes are integer multiple of chunk size
    trainSize -= (trainSize % options.chunkSize)
    testSize -= (testSize % options.chunkSize)
    trainSize = int(trainSize)
    testSize = int(testSize)
    print >> sys.stderr, '%d training sequences\n %d test sequences ' % (trainSize,testSize)

    # note that we have 1 channel and 4xoptions.width matrices for dna sequence.
    NStrainData = f.create_dataset("NStrainInp", (trainSize,2,1,options.width))
    MStrainData = f.create_dataset("MStrainInp", (trainSize,2,1,options.width))
    DStrainData = f.create_dataset("DStrainInp", (trainSize,4,1,options.width))
    RStrainData = f.create_dataset("RStrainInp", (trainSize,1,1,options.width))
    TFtrainData = f.create_dataset("TFtrainInp", (trainSize,2,1,options.width))

    trainTarget = f.create_dataset("trainOut", (trainSize,1,1,options.width))

    # note that we have 1 channel and 4xoptions.width matrices for dna sequence.
    NStestData = f.create_dataset("NStestInp", (testSize,2,1,options.width))
    MStestData = f.create_dataset("MStestInp", (testSize,2,1,options.width))
    DStestData = f.create_dataset("DStestInp", (testSize,4,1,options.width))
    RStestData = f.create_dataset("RStestInp", (testSize,1,1,options.width))
    TFtestData = f.create_dataset("TFtestInp", (testSize,2,1,options.width))

    testTarget = f.create_dataset("testOut", (testSize,1,1,options.width))

    info = f.create_dataset("info", (trainSize+testSize,4)) # chromosome no,  strand, index of the annotation, genomic position

    chrRange = annot['chr'].unique()
    # Use 4 channel and 1xoptions.width
    NSdata = np.zeros([options.chunkSize,2,1,options.width])
    MSdata = np.zeros([options.chunkSize,2,1,options.width])
    DSdata = np.zeros([options.chunkSize,4,1,options.width])
    RSdata = np.zeros([options.chunkSize,1,1,options.width])
    TFdata = np.zeros([options.chunkSize,2,1,options.width])
    target = np.zeros([options.chunkSize,1,1,options.width])

    infodata = np.zeros([options.chunkSize,4])

    qq=0;
    cc=0;
    ps =0;
    nestVar = 0;
    debugMode = True

    # if options.species in ['YJ167','YJ168','YJ169','Scer','YSC001']:
    #     assembly = 'sacCer3'
    # elif options.species in ['YJ160']:
    #     assembly = 'Klla'
    # elif options.species in ['YJ177']:
    #     assembly = 'DeHa2'
    # else:
    #     raise('unknown species')

    print assembly
    gdb = genome.db.GenomeDB(path='/Users/umut/Projects/genome/data/share/genome_db',assembly=assembly)

    NSpos = gdb.open_track('NSpos')
    NSneg = gdb.open_track('NSneg')
    MSpos = gdb.open_track('MSpos')
    MSneg = gdb.open_track('MSneg')
    TFpos = gdb.open_track('TFpos')
    TFneg = gdb.open_track('TFneg')
    RS = gdb.open_track('RS')
    TSpos = gdb.open_track('TSpos')
    TSneg = gdb.open_track('TSneg')
    seq = gdb.open_track("seq")

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
                tfP = TFpos.get_nparray(chname,pos+1,(pos+options.width))
                tfN = TFneg.get_nparray(chname,pos+1,(pos+options.width))
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
                    NSdata[qq,0,0,:] = nsP.T
                    NSdata[qq,1,0,:] = nsN.T
                    MSdata[qq,0,0,:] =msP.T
                    MSdata[qq,1,0,:] = msN.T
                    DSdata[qq,:,0,:] = dsdata.T
                    RSdata[qq,0,0,:] = rs.T
                    TFdata[qq,0,0,:] =tfP.T
                    TFdata[qq,1,0,:] = tfN.T
                    if sum(tsP)==0:
                        tsP = tsP + 1/np.float(options.width)
                    else:
                        tsP = tsP/sum(tsP+1e-5)
                    target[qq,0,0,:] =tsP.T
                    infodata[qq,:] = [cc, 1,annotIdx,pos]
                else:
                    NSdata[qq,0,0,:] = np.flipud(nsN).T
                    NSdata[qq,1,0,:] = np.flipud(nsP).T
                    MSdata[qq,0,0,:] = np.flipud(msN).T
                    MSdata[qq,1,0,:] = np.flipud(msP).T
                    RSdata[qq,0,0,:] = np.flipud(rs).T
                    DSdata[qq,:,0,:] = np.flipud(np.fliplr(dsdata)).T
                    TFdata[qq,0,0,:] = np.flipud(tfN).T
                    TFdata[qq,1,0,:] = np.flipud(tfP).T
                    if sum(tsN)==0:
                        tsN = tsN + 1/np.float(options.width)
                    else:
                        tsN = tsN/sum(tsN+1e-5)
                    target[qq,0,0,:] =np.flipud(tsN).T
                    infodata[qq,:] = [cc, -1,annotIdx,pos]

                qq+=1

                if ((ps+options.chunkSize) <= trainSize) and (qq>=options.chunkSize):
                    stp = options.chunkSize
                    NStrainData[range(ps,ps+stp),:,:,:] = NSdata
                    MStrainData[range(ps,ps+stp),:,:,:] = MSdata
                    DStrainData[range(ps,ps+stp),:,:,:] = DSdata
                    RStrainData[range(ps,ps+stp),:,:,:] = RSdata
                    TFtrainData[range(ps,ps+stp),:,:,:] = TFdata
                    trainTarget[range(ps,ps+stp),:,:,:] = target
                    info[range(ps,ps+stp),:] = infodata
                    NSdata = np.zeros([options.chunkSize,2,1,options.width])
                    MSdata = np.zeros([options.chunkSize,2,1,options.width])
                    DSdata = np.zeros([options.chunkSize,4,1,options.width])
                    RSdata = np.zeros([options.chunkSize,1,1,options.width])
                    infodata = np.zeros([options.chunkSize,4])
                    ps+=stp
                    qq=0
                    print >> sys.stderr, '%d  training chunk saved ' % ps
                if (ps >= trainSize) & (ps < (trainSize + testSize)) and (qq>=options.chunkSize):
                    NStestData[range(ps-trainSize,ps-trainSize+stp),:,:,:] = NSdata
                    MStestData[range(ps-trainSize,ps-trainSize+stp),:,:,:] = MSdata
                    DStestData[range(ps-trainSize,ps-trainSize+stp),:,:,:] = DSdata
                    RStestData[range(ps-trainSize,ps-trainSize+stp),:,:,:] = RSdata
                    TFtestData[range(ps-trainSize,ps-trainSize+stp),:,:,:] = TFdata
                    testTarget[range(ps-trainSize,ps-trainSize+stp),:,:,:] = target
                    info[range(ps,ps+stp),:] = infodata
                    rt = ps-trainSize
                    ps+=stp
                    qq=0
                    print >> sys.stderr, '%d  testing chunk saved ' % rt
                if ps >=(trainSize+testSize):
                    nestVar = 1;
                    break

    print ps
    f.close()
    NSpos.close()
    NSneg.close()
    MSpos.close()
    MSneg.close()
    RS.close()
    TFpos.close()
    TFneg.close()
    TSpos.close()
    TSneg.close()
    seq.close()

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


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
