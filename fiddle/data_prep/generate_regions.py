import sys
from optparse import OptionParser
import os
import numpy as np
from tqdm import tqdm as tq
import pdb

def main():
    usage = 'usage: %prog [options] <chr_size> <out_file_name> <annotation_file_path> '
    parser = OptionParser(usage)
    parser.add_option('-t', dest='loci_of_interest', default='CDS', type='str', help='Loci of interest e.g. TSS, SNP, CDS etc. [Default: %default]')
    parser.add_option('-e', dest='width', type='int', default=500, help='Extend all sequences to this length [Default: %default]')
    parser.add_option('-r', dest='stride', default=20, type='int', help='Stride size for moving window [Default: %default]')
    parser.add_option('-u', dest='upstream', default=500, type='int', help='Upstream distance to locus of interest to include [Default: %default]')
    parser.add_option('-d', dest='downstream', default=500, type='int', help='Upstream distance to locus of interest to include [Default: %default]')
    parser.add_option('-s', dest='split', default=2, type='int', help='Split into validation and test sets i.e. 0: only train, 1:train and test, 2: train test and validation [Default: %default]')
    (options, args) = parser.parse_args()

    # Make directory for the project, establish indexing bounds
    directory = "~/Projects/FIDDLE/data/regions/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    save_path = os.path.join(directory, args[1])
    xran_pos = np.arange(-options.upstream, options.downstream, options.stride)
    xran_neg = np.arange(-options.downstream, options.upstream, options.stride)


    if 'gff' in args[2][-4:]:
        from parse_gff3 import parseGFF3
        with open(args[0], 'r') as f:
            chr_sizes = {line.split('\t')[0]: (1, int(line.split('\t')[-1].split('\n')[0])) for line in f.readlines()}
        # Read in gff3 annotation file
        with open(save_path, 'w') as fw:
            for record in tq(parseGFF3(args[2])):
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
    elif 'bed' in args[2][-4:]:
        import pandas as pd

        df = pd.read_csv(args[2],sep='\t', header=None)
        df.drop(df.columns[3:], axis=1, inplace=True)
        df.columns=['chr', 'start', 'end']

        print('Random striding ...')
        # pdb.set_trace()

        shp = df.shape[0]
        starts = ((df['start'].values+df['end'].values)/2).astype(int) - 500
        chrs = df['chr'].values


        df_new = pd.DataFrame({'chr':chrs, 'start': starts, 'end': starts + 500})


        for ix in tq(range(10)):
            rands = np.random.randint(1000, size=shp)
            new_starts = starts + rands
            df_new = df_new.append(pd.DataFrame({'chr':chrs, 'start': new_starts, 'end': new_starts + 500}))

        df_new.to_csv(save_path, sep='\t', header=False, index=False, columns=['chr', 'start', 'end'])
        print('Saved to: ' + save_path)
if __name__ == '__main__':
    main()
