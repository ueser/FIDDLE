#!/usr/bin/env python
import os
import sys
sys.path.append('/Users/umut/Projects/genome/python/lib')
import genome.db
from optparse import OptionParser
import h5py
import pandas as pd
import numpy as np


################################################################################
# data4predictionYAC.py
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
        parser.error('Must provide annotation file, and an output prefix')
    else:
        annot_file = args[0]
        out_file = args[1]

    #### <-- to be generalized ---> ####
    annot = pd.read_table(annot_file,sep="\t")
    # annot.drop(annot.columns[[range(6,11)]],axis=1,inplace=True)
    print annot.head()
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

    totLen=0
    chrRange = annot['chr'].unique()
    for chname in chrRange:
        st = min(annot[annot['chr']==chname].start)
        en = max(annot[annot['chr']==chname].end-options.width)
        totLen += (en-st-options.width)/options.stride

    trainSize = 2*np.ceil(totLen+options.chunkSize)

    print >> sys.stderr, '%d training sequences ' % trainSize

    print os.path.join(directory,out_file)
    # open the hdf5 file to write
    f = h5py.File(os.path.join(directory,out_file), "w")
    # note that we have 1 channel and 4xoptions.width matrices for dna sequence.
    NStrainData = f.create_dataset("NStrainInp", (trainSize,2,1,options.width))
    MStrainData = f.create_dataset("MStrainInp", (trainSize,2,1,options.width))
    DStrainData = f.create_dataset("DStrainInp", (trainSize,4,1,options.width))
    RStrainData = f.create_dataset("RStrainInp", (trainSize,1,1,options.width))
    TFtrainData = f.create_dataset("TFtrainInp", (trainSize,2,1,options.width))

    info = f.create_dataset("info", (trainSize,4)) # chromosome no,  strand, index of the annotation, genomic position

    # Use 4 channel and 1xoptions.width
    NSdata = np.zeros([options.chunkSize,2,1,options.width])
    MSdata = np.zeros([options.chunkSize,2,1,options.width])
    DSdata = np.zeros([options.chunkSize,4,1,options.width])
    RSdata = np.zeros([options.chunkSize,1,1,options.width])
    TFdata = np.zeros([options.chunkSize,2,1,options.width])
    infodata = np.zeros([options.chunkSize,4])

    if options.species in ['Scer','YSC001']:
        assembly = 'sacCer3'
    elif options.species in ['YJ160']:
        assembly = 'Klla'
    elif options.species in ['YJ177']:
        assembly = 'DeHa2'
    elif options.species in ['YJ167','YJ168','YJ169']:
        assembly = 'KllaYAC'
    elif options.species in ['YJ170','YJ71']:
        assembly = 'DehaYAC'
    else:
        raise('unknown species')

    print assembly
    gdb = genome.db.GenomeDB(path='/Users/umut/Projects/genome/data/share/genome_db',assembly=assembly)

    NSpos = gdb.open_track('NSpos')
    NSneg = gdb.open_track('NSneg')
    MSpos = gdb.open_track('MSpos')
    MSneg = gdb.open_track('MSneg')
    TFpos = gdb.open_track('TFpos')
    TFneg = gdb.open_track('TFneg')
    RS = gdb.open_track('RS')
    seq = gdb.open_track("seq")

    qq=0;
    cc=0;
    ps =0;
    debugMode = True
    for chname in chrRange:
        cc+=1
        st = min(annot[annot['chr']==chname].start)
        en = max(annot[annot['chr']==chname].end)-options.width

        xran = np.arange(st,en,options.stride)
        for pos in xran:

            seqVec = seq.get_seq_str(chname,pos+1,(pos+options.width))
            dsdata = vectorizeSequence(seqVec.lower())

            nsP = NSpos.get_nparray(chname,pos+1,(pos+options.width))
            nsN = NSneg.get_nparray(chname,pos+1,(pos+options.width))
            msP = MSpos.get_nparray(chname,pos+1,(pos+options.width))
            msN = MSneg.get_nparray(chname,pos+1,(pos+options.width))
            tfP = TFpos.get_nparray(chname,pos+1,(pos+options.width))
            tfN = TFneg.get_nparray(chname,pos+1,(pos+options.width))
            rs = RS.get_nparray(chname,pos+1,(pos+options.width))

            if debugMode:
                if not checkData(np.r_[nsP,nsN,msP,msN,rs,tfP,tfN]):
                    print('NaN detected in chr' + chname + ' and at the position:' + str(pos))
                    continue


            NSdata[qq,0,0,:] = nsP.T
            NSdata[qq,1,0,:] = nsN.T

            MSdata[qq,0,0,:] = msP.T
            MSdata[qq,1,0,:] = msN.T

            TFdata[qq,0,0,:] = tfP.T
            TFdata[qq,1,0,:] = tfN.T

            DSdata[qq,:,0,:] = dsdata.T
            RSdata[qq,0,0,:] = rs.T

            infodata[qq,:] = [cc, 1,0,pos]
            qq+=1

            NSdata[qq,0,0,:] = np.flipud(nsN).T
            NSdata[qq,1,0,:] = np.flipud(nsP).T

            MSdata[qq,0,0,:] = np.flipud(msN).T
            MSdata[qq,1,0,:] = np.flipud(msP).T

            TFdata[qq,0,0,:] = np.flipud(tfN).T
            TFdata[qq,1,0,:] = np.flipud(tfP).T

            RSdata[qq,0,0,:] = np.flipud(rs).T
            DSdata[qq,:,0,:] = np.flipud(np.fliplr(dsdata)).T

            infodata[qq,:] = [cc, -1,0,pos]
            qq+=1

            if (ps < trainSize) and (qq>=options.chunkSize):
                stp = options.chunkSize
                NStrainData[range(ps,ps+stp),:,:,:] = NSdata
                MStrainData[range(ps,ps+stp),:,:,:] = MSdata
                TFtrainData[range(ps,ps+stp),:,:,:] = TFdata
                DStrainData[range(ps,ps+stp),:,:,:] = DSdata
                RStrainData[range(ps,ps+stp),:,:,:] = RSdata
                info[range(ps,ps+stp),:] = infodata
                # Use 4 channel and 1xoptions.width
                NSdata = np.zeros([options.chunkSize,2,1,options.width])
                MSdata = np.zeros([options.chunkSize,2,1,options.width])
                TFdata = np.zeros([options.chunkSize,2,1,options.width])
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
    TFpos.close()
    TFneg.close()
    RS.close()
    seq.close()

def vectorizeSequence(seq):
    # the order of the letters is not arbitrary.
    # Flip the matrix up-down and left-right for reverse compliment
    ltrdict = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1],'n':[0.25,0.25,0.25,0.25]}
    return np.array([ltrdict[x] for x in seq])

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
