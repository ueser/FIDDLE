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
# data4prediction.py
#
# Make an HDF5 file for Torch input using a pyTable wrapper called genome: https://github.com/gmcvicker/genome
#
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <annotation_file> <out_file>'
    parser = OptionParser(usage)
    parser.add_option('-d', dest='rootDir', type='str', default='.', help='Root directory of the project [Default: %default]')
    parser.add_option('-p', dest='species', type='str', default='Scer', help='Species? [Default: %default]')
    parser.add_option('-b', dest='chunkSize', default=100, type='int', help='Align sizes with batch size')
    parser.add_option('-e', dest='width', type='int', default=500, help='Extend all sequences to this length [Default: %default]')
    parser.add_option('-r', dest='stride', default=20, type='int', help='Stride sequences [Default: %default]')

    (options,args) = parser.parse_args()

    if len(args) !=2 :
        print(args)
        print(options)
        print(len(args))
        parser.error('Must provide annotation file and an output name')
    else:
        annot_file = args[0]
        out_file = args[1]


    #### <-- to be generalized ---> ####
    annot = pd.read_table(annot_file,sep="\t",header=None)
    # annot.drop(annot.columns[[range(6,11)]],axis=1,inplace=True)
    annot.columns = ['chr','start','end','strand']
    # switch start and end positions into tss for negative strand
    annot['tss'] = np.zeros([annot.shape[0]],dtype=int)
    tf = (annot.strand=='+')
    annot.loc[tf,'tss'] = annot[tf].start
    tf = (annot.strand=='-')
    annot.loc[tf,'tss'] = annot[tf].end
    #### --> to be generalized <--- ####

    # Make directory for the project
    directory = "../../data/hdf5datasets/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Parameters
    x1 = 750  #upstream
    x2 = 250  #downstream

    # open the hdf5 file to write
    f = h5py.File(os.path.join(directory,out_file), "w")
    trainSize = len(annot)*(x1+x2-options.width)/options.stride

    print >> sys.stderr, '%d training sequences ' % trainSize

    # note that we have 1 channel and 4xoptions.width matrices for dna sequence.
    NStrainData = f.create_dataset("NStrainInp", (trainSize,2,1,options.width))
    MStrainData = f.create_dataset("MStrainInp", (trainSize,2,1,options.width))
    DStrainData = f.create_dataset("DStrainInp", (trainSize,4,1,options.width))
    RStrainData = f.create_dataset("RStrainInp", (trainSize,1,1,options.width))
    # TFtrainData = f.create_dataset("TFtrainInp", (trainSize,2,1,options.width))

    info = f.create_dataset("info", (trainSize,4)) # chromosome no,  strand, index of the annotation, genomic position

    chrRange = annot['chr'].unique()
    # Use 4 channel and 1xoptions.width
    NSdata = np.zeros([options.chunkSize,2,1,options.width])
    MSdata = np.zeros([options.chunkSize,2,1,options.width])
    DSdata = np.zeros([options.chunkSize,4,1,options.width])
    RSdata = np.zeros([options.chunkSize,1,1,options.width])
    # TFdata = np.zeros([options.chunkSize,2,1,options.width])
    infodata = np.zeros([options.chunkSize,4])

    qq=0;
    cc=0;
    ps =0;
    nestVar = 0;
    debugMode = True

    if options.species in ['YJ167','YJ168','YJ169','Scer','YSC001']:
        assembly = 'sacCer3'
    elif options.species in ['YJ160']:
        assembly = 'Klla'
    elif options.species in ['YJ177']:
        assembly = 'DeHa2'
    else:
        raise('unknown species')

    print assembly
    gdb = genome.db.GenomeDB(path='/Users/umut/Projects/genome/data/share/genome_db',assembly=assembly)

    NSpos = gdb.open_track('NSpos')
    NSneg = gdb.open_track('NSneg')
    MSpos = gdb.open_track('MSpos')
    MSneg = gdb.open_track('MSneg')
    RS = gdb.open_track('RS')
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

                seqVec = seq.get_seq_str(chname,pos+1,(pos+options.width))
                dsdata = vectorizeSequence(seqVec.lower())

                nsP = NSpos.get_nparray(chname,pos+1,(pos+options.width))
                nsN = NSneg.get_nparray(chname,pos+1,(pos+options.width))
                msP = MSpos.get_nparray(chname,pos+1,(pos+options.width))
                msN = MSneg.get_nparray(chname,pos+1,(pos+options.width))
                rs = RS.get_nparray(chname,pos+1,(pos+options.width))


                if debugMode:
                    if not checkData(np.r_[nsP,nsN,msP,msN,rs]):
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
                    infodata[qq,:] = [cc, 1,annotIdx,pos]
                else:
                    NSdata[qq,0,0,:] = np.flipud(nsN).T
                    NSdata[qq,1,0,:] = np.flipud(nsP).T
                    MSdata[qq,0,0,:] = np.flipud(msN).T
                    MSdata[qq,1,0,:] = np.flipud(msP).T
                    RSdata[qq,0,0,:] = np.flipud(rs).T
                    DSdata[qq,:,0,:] = np.flipud(np.fliplr(dsdata)).T
                    infodata[qq,:] = [cc, -1,annotIdx,pos]

                qq+=1

                if (ps < trainSize) and (qq>=options.chunkSize):
                    stp = options.chunkSize
                    NStrainData[range(ps,ps+stp),:,:,:] = NSdata
                    MStrainData[range(ps,ps+stp),:,:,:] = MSdata
                    DStrainData[range(ps,ps+stp),:,:,:] = DSdata
                    RStrainData[range(ps,ps+stp),:,:,:] = RSdata
                    info[range(ps,ps+stp),:] = infodata
                    NSdata = np.zeros([options.chunkSize,2,1,options.width])
                    MSdata = np.zeros([options.chunkSize,2,1,options.width])
                    DSdata = np.zeros([options.chunkSize,4,1,options.width])
                    RSdata = np.zeros([options.chunkSize,1,1,options.width])
                    infodata = np.zeros([options.chunkSize,4])
                    ps+=stp
                    qq=0
                    print >> sys.stderr, '%d  training chunk saved ' % ps

                if ps >=(trainSize):
                    nestVar = 1;
                    break

    print ps
    f.close()
    NSpos.close()
    NSneg.close()
    MSpos.close()
    MSneg.close()
    RS.close()
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
