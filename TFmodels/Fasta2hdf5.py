import numpy as np
from Bio import SeqIO
import h5py
from optparse import OptionParser
from tqdm import tqdm as tq
usage = 'usage: %prog [options] <fasta file path> <output file path>'
parser = OptionParser(usage)
parser.add_option('-b', dest='chunkSize', default=1000, type='int', help='Align sizes with batch size')
parser.add_option('-w', dest='width', default=1000, type='int', help='Length of each sequence')


(options,args) = parser.parse_args()

def bufcount(filename):
    f = open(filename)
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.read # loop optimization

    buf = read_f(buf_size)
    while buf:
        lines += buf.count('>')
        buf = read_f(buf_size)

    return lines


def vectorizeSequence(seq):
    # the order of the letters is not arbitrary.
    # Flip the matrix up-down and left-right for reverse compliment
    ltrdict = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0]}
    return np.array([ltrdict[x] for x in seq])



records = list(SeqIO.parse(args[0], "fasta"))
totlen = bufcount(args[0])
counter_ = 0
current_index = 0
with h5py.File(args[1], 'w') as hf:
    seqPointer = hf.create_dataset('hg38_promoters',(totlen,4,options.width))

    for fasta in tq(records):
        # get the fasta files.
        name, sequence = fasta.id, fasta.seq.tostring()
        # Write the chromosome name
        new_file.write(name)
        # one hot encode...
        data[counter_]=vectorizeSequence(sequence.lower())
        counter_+=1
        if counter_ == options.chunkSize:
            seqPointer[current_index:(current_index+options.chunkSize),:,:] = data[:]
            current_index += options.chunkSize
            counter_ = 0

    assert (current_index+counter_) == totlen, 'Total length does not match!'
    seqPointer[current_index:(current_index+counter_),:,:] = data[:counter_]
