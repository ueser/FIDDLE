from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import h5py # alternatively, tables module can be used
from tqdm import tqdm as tq

from matplotlib import pylab as pl


f = h5py.File('../data/hdf5datasets/NSMSDSRSCSTS.hdf5','r')
NSinp = f.get('NETseq')[:50000]
TSout = f.get('TSSseq')[:50000]

idx = np.random.choice(50000,50000,replace=False)
NSinp = NSinp[idx].reshape(50000,2,500,1)
TSout = TSout[idx,0,:]



X = tf.placeholder(tf.float32, shape=[None,2,500,1],name='NETseq')
Y = tf.placeholder(tf.float32, shape=[None,500],name='TSSseq')

X1 = tf.contrib.layers.convolution2d(inputs=X,num_outputs=40,kernel_size = [2,10],\
padding='VALID',activation_fn=tf.nn.relu,normalizer_fn=tf.contrib.layers.batch_norm )
X1 = tf.contrib.layers.max_pool2d(X1,stride=[1,3],kernel_size=[1,3])
X1 = tf.contrib.layers.flatten(X1)
X1 = tf.contrib.layers.fully_connected(X1,500)
X1_sm = tf.nn.softmax(X1)

loss = tf.nn.softmax_cross_entropy_with_logits(X1_sm,Y)


target = tf.floor(10.*tf.cast(tf.argmax(Y,dimension=1),tf.float32)/500)
pred = tf.floor(10.*tf.cast(tf.argmax(X1_sm,dimension=1),tf.float32)/500)
accuracy = tf.reduce_sum(tf.cast(tf.equal(pred,target),tf.int32))


meanLoss = tf.reduce_mean(loss)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(meanLoss)


nn=0
fig = pl.figure()
pl.plot(TSout[45,:])
pl.savefig('testOutput_'+str(nn))
pl.close(fig)
fig = pl.figure()
pl.plot(NSinp[45,0,:])
pl.plot(-NSinp[45,1,:])
pl.savefig('testInput_'+str(nn))
pl.close(fig)
fig = pl.figure()
pl.plot(TSout[18,:])
pl.savefig('testOutputB_'+str(nn))
pl.close(fig)
fig = pl.figure()
pl.plot(NSinp[18,0,:])
pl.plot(-NSinp[18,1,:])
pl.savefig('testInputB_'+str(nn))
pl.close(fig)




sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
totalTrainTssLoss=0
totalTrainAcc =0
testDict ={X:NSinp[40000:],Y:TSout[40000:]}
normTrain = len(range(0,40000,100))
for epoch in tq(range(50)):
    for iterationNo in tq(range(0,40000,100)):
        _, tssLoss,acc = sess.run([train_op, meanLoss,accuracy],feed_dict={X:NSinp[iterationNo:(iterationNo+100)],Y:TSout[iterationNo:(iterationNo+100)]})
        totalTrainTssLoss+=tssLoss
        totalTrainAcc +=acc



    meanTrainTssLoss = totalTrainTssLoss/(normTrain+0.)
    meanTrainAcc = totalTrainAcc/(normTrain+0.)
    tssLoss, testAcc,X1sm = sess.run([meanLoss,accuracy,X1_sm],feed_dict=testDict)
    print("Training step " + str(epoch) + " : \n Train loss: " + str(meanTrainTssLoss) +"\n Train accuracy(%): "+ str(100.*meanTrainAcc/100))
    print("Test loss: " +  str(tssLoss)+\
    "\n Test accuracy(%): "+ str(100.*testAcc/10000))
    totalTrainAcc = 0
    totalTrainTssLoss = 0
    fig = pl.figure()
    pl.plot(X1sm[45,:])
    pl.savefig('A_'+str(nn))
    pl.close(fig)
    fig = pl.figure()
    pl.plot(X1sm[18,:])
    pl.savefig('B_'+str(nn))
    pl.close(fig)
    nn+=1
