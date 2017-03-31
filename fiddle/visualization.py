from matplotlib import pylab as pl
import numpy as np
import h5py
import os, io, sys
import pdb
from math import sqrt
import tensorflow as tf
from tqdm import tqdm as tq
import cPickle as pickle
import tensorflow as tf


### FIDDLE specific tools ###
sys.path.append('../dev/')
from viz_sequence import *
#############################


################################################################################
# Main
################################################################################
def main():
    vizflags = tf.app.flags
    vizflags.DEFINE_string('runName', 'experiment', 'Running name.')
    vizflags.DEFINE_string('resultsDir', '../results', 'Directory for results data')
    vizflags.DEFINE_boolean('makeGif', True, 'Make gif from png files')
    vizflags.DEFINE_boolean('makePng', True, 'Make png from saved prediction pickles')
    vizflags.DEFINE_string('vizType', 'dnaseq', 'data type to be vizualized')

    vizFLAGS = vizflags.FLAGS

    save_dir = os.path.join(FLAGS.resultsDir,FLAGS.runName)

    if FLAGS.makePng:
        pckl_files = [fname for fname in os.listdir(save_dir) if 'pred_viz' in fname]
        orig_file = [fname for fname in os.listdir(save_dir) if 'originals.pck' in fname]
        pred_dict = pickle.load(open(os.path.join(save_dir, pckl_files[0]), 'r'))

        if ('dna_before_softmax' in pred_dict.keys()):

            qq=0
            for f_ in tq(pckl_files):
                pred_dict = pickle.load(open(os.path.join(save_dir, f_),'r'))
                iter_no  = int(f_.split('.')[0].split('_')[-1])
                qq+=1
                print('\nplotting {} of {}'.format(qq, len(pckl_files)))
                weights = pred_dict['dna_before_softmax']
                pred_vec = pred_dict['prediction']
                visualize_dna(weights, pred_vec,
                          name='iteration_{}'.format(iter_no),
                          save_dir=save_dir, verbose=False)

        elif vizFLAGS.vizType == 'tssseq':
            qq = 0
            orig_output = pickle.load(open(os.path.join(save_dir, orig_file), 'r'))
            strand = 'Single'
            for f_ in tq(pckl_files):
                pred_dict = pickle.load(open(os.path.join(save_dir, f_), 'r'))
                iter_no = int(f_.split('.')[0].split('_')[-1])
                qq += 1
                print('\nplotting {} of {}'.format(qq, len(pckl_files)))

                if (qq==1) and (pred_dict.values()[0].shape[1]==2):
                    strand = 'Double'
                plot_prediction(predicted_dict, orig_output,
                            name='iteration_{}'.format(iter_no),
                            save_dir=save_dir,
                            strand=strand)

        else:
            raise NotImplementedError

    if vizFLAGS.makeGif:
        print('Making gif animation ... ')
        import imageio
        images = []
        png_files = [fname for fname in os.listdir(save_dir) if '.png' in fname]
        sorted_idx = np.argsort([int(f_.split('.')[0].split('_')[-1]) for f_ in png_files])

        for ix in tq(sorted_idx):
            filename = os.path.join(save_dir,png_files[ix])
            images.append(imageio.imread(filename))
        imageio.mimsave(os.path.join(save_dir,'prediction_viz.gif'), images)




################################################################################
# Auxilary Functions
################################################################################

def plot_prediction(pred_vec, orig_vec=None, save_dir='../results/', name='profile_prediction', strand='Single'):
    pl.ioff()
    fig, axarr = pl.subplots(pred_vec.values()[0].shape[0],len(pred_vec))
    if strand=='Double':
        to_size = pred_vec.values()[0].shape[1]/2
        for ix in range(pred_vec.values()[0].shape[0]):
            for jx, key in enumerate(pred_vec.keys()):
                if orig_vec is not None:
                    # pdb.set_trace()
                    axarr[ix, jx].plot(orig_vec[key][ix, 0, :]/np.sum(orig_vec[key][ix,:,:]+ 1e-7), label=key+'_Original', color='g')
                    axarr[ix, jx].plot(-orig_vec[key][ix, 1, :]/np.sum(orig_vec[key][ix,:,:]+ 1e-7), color='g')
                axarr[ix, jx].plot(pred_vec[key][ix, :to_size], label=key+'_Prediction', color='r')
                axarr[ix, jx].plot(-pred_vec[key][ix, to_size:], color='r')
                axarr[ix, jx].axis('off')
        axarr[0, 0].set_title(pred_vec.keys()[0])
        axarr[0, 1].set_title(pred_vec.keys()[1])


    else:
        for ix in range(pred_vec.values()[0].shape[0]):
            for jx, key in enumerate(pred_vec.keys()):
                if orig_vec is not None:
                    # pdb.set_trace()
                    axarr[ix, jx].plot(orig_vec[key][ix,0, :] / np.max(orig_vec[key][ix,0, :] + 1e-7),
                                       label=key + '_Original', color='g')
                axarr[ix, jx].plot(pred_vec[key][ix, :]/np.max(pred_vec[key][ix, :]), label=key + '_Prediction', color='r')
                axarr[ix, jx].axis('off')

        axarr[0, 1].set_title(pred_vec.keys()[0])
        axarr[0, 1].set_title(pred_vec.keys()[1])

    pl.savefig(os.path.join(save_dir,name+'.png'),format='png')
    pl.close(fig)



def put_kernels_on_grid(kernel, pad = 1):

    ''' modified from @kukuruza: https://gist.github.com/kukuruza/03731dc494603ceab0c5
    Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)
    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''
    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of filters')
                return (i, int(n / i))
    (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
    print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scaling to [0, 255] is not necessary for tensorboard
    return x7


def visualize_filters():

    raise NotImplementedError

def plot_weights(array,
                 height_padding_factor=0.2,
                 length_padding=1.0,
                 subticks_frequency=1.0,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs,
                 highlight={},
                 ax=[]):
    # fig = plt.figure(figsize=(20,2))
    # ax = fig.add_subplot(111)
    plot_weights_given_ax(ax=ax, array=array,
        height_padding_factor=height_padding_factor,
        length_padding=length_padding,
        subticks_frequency=subticks_frequency,
        colors=colors,
        plot_funcs=plot_funcs,
        highlight=highlight)

def visualize_dna(weigths, pred_vec, save_dir='../results/', name='dna_prediction', verbose=True):
    pl.ioff()
    fig = pl.figure(figsize=(20,20))
    for ix in tq(range(pred_vec.shape[0])):
        if verbose:
            print('\nsubplotting {} of {}'.format(ix, pred_vec.shape[0]))
        ax = fig.add_subplot(pred_vec.shape[0], 1, ix+1)
        H = abs((.25 * np.log2(.25 + 1e-7) - pred_vec[ix, :, :, 0] * np.log2(pred_vec[ix, :,:,0] + 1e-7)).sum(axis=0))
        H = np.tile(H, 4).reshape(4, pred_vec.shape[2], 1)
        plot_weights(weigths[ix] * H,
                     height_padding_factor=0.2,
                     length_padding=1.0,
                     colors=default_colors,
                     subticks_frequency=pred_vec.shape[2]/2,
                     plot_funcs=default_plot_funcs,
                     highlight={},
                     ax=ax)
    pl.savefig(os.path.join(save_dir, name + '.png'), format='png')
    pl.close(fig)



if __name__=='__main__':
    main()