from matplotlib import pylab as pl
import numpy as np
import h5py
from matplotlib.backends.backend_pdf import PdfPages
import os
import pdb

def plot_prediction(pred_vec, orig_vec=None, save_dir='../results/', name='profile_prediction', strand='Single'):
    pl.ioff()
    pp = PdfPages(os.path.join(save_dir,name+'.pdf'))


    if strand=='Double':
        to_size = pred_vec.values()[0].shape[1]/2
        for ix in range(pred_vec.values()[0].shape[0]):
            for key in pred_vec.keys():
                fig = pl.figure()
                if orig_vec is not None:
                    # pdb.set_trace()
                    pl.plot(orig_vec[key][ix, 0, :]/np.sum(orig_vec[key][ix,:,:]+ 1e-7), label=key+'_Original')
                    pl.plot(-orig_vec[key][ix, 1, :]/np.sum(orig_vec[key][ix,:,:]+ 1e-7))
                pl.plot(pred_vec[key][ix, :to_size], label=key+'_Prediction')
                pl.plot(-pred_vec[key][ix, to_size:])
                pl.legend()
                pl.title(key+'_'+str(ix))
                pp.savefig()
                pl.close(fig)
    else:
        for ix in range(pred_vec.values()[0].shape[0]):
            for key in pred_vec.keys():
                fig = pl.figure()
                if orig_vec is not None:
                    pl.plot(orig_vec[key][ix, 0, :]/np.sum(orig_vec[key][ix,0,:]+ 1e-7), label=key+'_Original')
                pl.plot(pred_vec[key][ix, :], label=key+'_Prediction')
                pl.legend()
                pl.title(key+'_'+str(ix))
                pp.savefig()
                pl.close(fig)
    pp.close()


def visualize_filters():
    raise NotImplementedError

def main():
    raise NotImplementedError


if __name__=='__main__':
    main()