from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


from keras.layers import Input, Dense, Lambda, Convolution2D, concatenate, Reshape,AveragePooling2D, Flatten
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import History
import h5py


def KL_loss(y_true, y_pred):
    y_true = Reshape((-1,1000), input_shape=(None,2,500))(y_true)

    return objectives.kullback_leibler_divergence(y_true, y_pred)


h5pnt = h5py.File('../../intragenicTranscription/data/hdf5datasets/CN2TS_DIAandWT_500bp.h5','r')
print(h5pnt['tssseq'])
representation = {}

input={'dnaseq':Input(shape=(4,500),  name='dnaseq'), 'chipnexus':Input(shape=(2,500),  name='chipnexus')}
h1 = Reshape((4,500,1),input_shape=(None, 4, 500))(input['dnaseq'])
h1 = Convolution2D(64, 4, 10,
                 activation='relu', dim_ordering='tf')(h1)

h1 = AveragePooling2D(pool_size=(1, 5), strides=(1,5), border_mode='valid')(h1)

h1 = Convolution2D(64, 1, 10,
                 activation='relu',dim_ordering='tf')(h1)
h1 = AveragePooling2D(pool_size=(1, 5), strides=(1,5), border_mode='valid')(h1)
h1 = Flatten()(h1)
representation['dnaseq'] = Dense(500, activation='linear')(h1)

h1 = Reshape((2,500,1),input_shape=(None, 2, 500))(input['chipnexus'])
h1 = Convolution2D(64, 2, 10,
                 activation='relu',dim_ordering='tf')(h1)

h1 = AveragePooling2D(pool_size=(1, 5), strides=(1,5), border_mode='valid')(h1)

h1 = Convolution2D(64, 1, 10,
                 activation='relu',dim_ordering='tf')(h1)
h1 = AveragePooling2D(pool_size=(1, 5), strides=(1,5), border_mode='valid')(h1)
h1 = Flatten()(h1)
representation['chipnexus'] = Dense(500, activation='linear')(h1)

h2 = concatenate([representation['dnaseq'], representation['chipnexus']], axis=-1)
h2 = Reshape((2,500,1))(h2)
h2 = Convolution2D(64, 2, 10,
                 activation='relu',dim_ordering='tf')(h2)

h2 = AveragePooling2D(pool_size=(1, 5), strides=(1,5), border_mode='valid')(h2)
h2 = Flatten()(h2)
pred = Dense(2*500, activation='softmax')(h2)


fiddle = Model([input['dnaseq'],input['chipnexus']], pred)
print(fiddle.summary())
fiddle.compile(optimizer='adam', loss=KL_loss)

fiddle.fit([h5pnt['dnaseq'], h5pnt['chipnexus']],
           h5pnt['tssseq'],
           shuffle=True,
           validation_split=0.1,
           batch_size=100,
           epoch=50)

