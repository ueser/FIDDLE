import h5py
import numpy as np

def main():
    h5pnt = h5py.File('/Users/umut/Projects/FIDDLE/data/hdf5datasets/CN2TS_DIAandWT_500bp.h5', 'w')
    max_size = 50000
    dnaseq = h5pnt.create_dataset('dnaseq', (max_size, 4, 500, 1))
    tssseq = h5pnt.create_dataset('tssseq', (max_size, 2, 500, 1))
    chipnexus = h5pnt.create_dataset('chipnexus', (max_size, 2, 500, 1))
    quality = h5pnt.create_dataset('quality', (max_size,))

    qq = 0
    qq2 = 0
    qual = ['HQ', 'MQ', 'LQ']
    for ix in range(3):
        if qq2 == max_size:
            break
        seqs = get_fasta('/Users/umut/Projects/intragenicTranscription/data/extracts/' + qual[ix] + '.fa')
        seqs_array = np.array(map(one_hot_encode_sequence, seqs))
        for smpl in ['Dia', 'Dia_Cnt']:
            print('Doing ' + smpl + ' ' + qual[ix])
            tmp_ts = np.genfromtxt('/Users/umut/Projects/intragenicTranscription/data/extracts/' + qual[ix] + '_' + smpl + '.ts.pos_neg.txt')
            tmp_cn = np.genfromtxt('/Users/umut/Projects/intragenicTranscription/data/extracts/' + qual[ix] + '_' + smpl + '.cn.pos_neg.txt')
            idx = np.sort(np.unique(np.r_[np.where(np.sum(np.isnan(tmp_ts), axis=1) == 0)[0],
                                          np.where(np.sum(np.isnan(tmp_cn), axis=1) == 0)[0]]))
            qq2 = (qq + len(idx))
            if (qq + len(idx)) > max_size:
                qq2 = max_size
            tssseq[qq:qq2, 0, :, 0] = tmp_ts[idx[:(qq2 - qq)], :500]
            tssseq[qq:qq2, 1, :, 0] = tmp_ts[idx[:(qq2 - qq)], 500:]
            chipnexus[qq:qq2, 0, :, 0] = tmp_cn[idx[:(qq2 - qq)], :500]
            chipnexus[qq:qq2, 1, :, 0] = tmp_cn[idx[:(qq2 - qq)], 500:]
            dnaseq[qq:qq2, :, :, 0] = seqs_array[idx[:(qq2 - qq)], :, :]
            quality[qq:qq2] = ix * np.ones((qq2 - qq))
            qq += len(idx)

            if qq2 == max_size:
                break


    h5pnt.close()

    print('all data saved to hdf5')
#!mkdir -p /Users/umut/Projects/FIDDLE/data/hdf5datasets/CN2TS_DIAandWT_500bp
    validation_ratio = 0.05
    test_ratio = 0.1

    idx = np.arange(h5pnt.values()[0].shape[0])
    np.random.shuffle(idx)
    validation_size = int(len(idx) * validation_ratio)
    test_size = int(len(idx) * test_ratio)
    train_size = len(idx) - validation_size - test_size

    train_h5 = h5py.File('/Users/umut/Projects/FIDDLE/data/hdf5datasets/CN2TS_DIAandWT_500bp/train.h5', 'w')
    test_h5 = h5py.File('/Users/umut/Projects/FIDDLE/data/hdf5datasets/CN2TS_DIAandWT_500bp/test.h5', 'w')
    validation_h5 = h5py.File('/Users/umut/Projects/FIDDLE/data/hdf5datasets/CN2TS_DIAandWT_500bp/validation.h5', 'w')

    train = {}
    validation = {}
    test = {}
    for key in tq(h5pnt.keys()):

        train[key] = train_h5.create_dataset(key, ((train_size,) + h5pnt[key].shape[1:]))
        validation[key] = validation_h5.create_dataset(key, ((validation_size,) + h5pnt[key].shape[1:]))
        test[key] = test_h5.create_dataset(key, ((test_size,) + h5pnt[key].shape[1:]))
        tmp = h5pnt.get(key)[:]
        train[key][:] = tmp[idx[:train_size]]
        validation[key][:] = tmp[idx[train_size:(train_size + validation_size)]]
        test[key][:] = tmp[idx[(train_size + validation_size):]]

    train_h5.close()
    test_h5.close()
    validation_h5.close()



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


if __name__=='__main__':
    main()