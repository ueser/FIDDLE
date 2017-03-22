import sys
from optparse import OptionParser
import os
import numpy as np
from parse_gff3 import parseGFF3
from tqdm import tqdm as tq


def main():
    usage = 'usage: %prog [options] <chr_sizes> <annotation_file_path> <out_file_name>'
    parser = OptionParser(usage)
    parser.add_option('-t', dest='loci_of_interest', default='CDS', type='str', help='Loci of interest e.g. TSS, SNP, CDS etc. [Default: %default]')
    parser.add_option('-e', dest='width', type='int', default=500, help='Extend all sequences to this length [Default: %default]')
    parser.add_option('-r', dest='stride', default=20, type='int', help='Stride size for moving window [Default: %default]')
    parser.add_option('-u', dest='upstream', default=500, type='int', help='Upstream distance to locus of interest to include [Default: %default]')
    parser.add_option('-d', dest='downstream', default=500, type='int', help='Upstream distance to locus of interest to include [Default: %default]')
    parser.add_option('-s', dest='split', default=2, type='int', help='Split into validation and test sets i.e. 0: only train, 1:train and test, 2: train test and validation [Default: %default]')
    (options, args) = parser.parse_args()

    # Make directory for the project, establish indexing bounds
    directory = "../data/regions/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(args[0], 'r') as f:
        chr_sizes = {line.split('\t')[0]: (1, int(line.split('\t')[-1].split('\n')[0])) for line in f.readlines()}
    save_path = os.path.join(directory, args[2])
    xran_pos = np.arange(-options.upstream, options.downstream, options.stride)
    xran_neg = np.arange(-options.downstream, options.upstream, options.stride)

    # Read in gff3 annotation file
    with open(save_path, 'w') as fw:
        for record in tq(parseGFF3(args[1])):
            if (record['type'] != options.loci_of_interest) or \
                    (record['source'] != 'ensembl') or \
                    (record['seqid'] == 'Mito') or \
                    ((record['end'] - max(options.upstream, options.downstream)) > chr_sizes['chr' + record['seqid']]):
                continue
            starts = record['start'] + xran_pos if record['strand'] == '+' else record['end'] + xran_neg
            starts = starts[starts > 0]
            list_to_write = ['chr' + record['seqid'] + '\t' + str(st) + '\t' + str(st + options.width) + '\t.\t.\t' + record['strand']
                             for st in starts]
            fw.write('\n'.join(list_to_write)+'\n')

    # Construct train, test, and validation regions
    with open(save_path, 'r') as fr:
        _ = fr.readline() # discard first line, no information
        all_lines = fr.readlines()
    np.random.shuffle(all_lines)
    with open(os.path.join(directory, 'train_regions.bed'), 'w') as fw:
        fw.write(''.join(all_lines[4000:]))
    if options.split > 0:
        with open(os.path.join(directory, 'test_regions.bed'), 'w') as fw:
            fw.write(''.join(all_lines[2000:4000]))
    if options.split > 1:
        with open(os.path.join(directory, 'validation_regions.bed'), 'w') as fw:
            fw.write(''.join(all_lines[:2000]))

if __name__ == '__main__':
    main()
