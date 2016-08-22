from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf





def inference(inputList, inputPlaceholders, conf):
  """Build the multi modal model up to where it may be used for inference.
  Args:
    inputList: A list that contains the names of the input datasets, must match with the hdf5 records
    inputPlaceholders: List of placeholders that matches with the inputList.
    conf: Configuration dictionary. Default values are set in config.py
  Returns:
    logits: Output tensor with the computed logits.
  """
  # Concatenate the outputs of the sub models and add a convolutional layer on top of it
  if len(inputList)>1:
      print('Sub models are being created...')
      modelList=[]
      for i,inputName in enumerate(inputList):
          with tf.name_scope(inputName):
              weights,biases = getConvNetParams(inputName,conf)
              fc1, shp = makeSubModel(inputPlaceholders[i],conf[inputName],weights,biases)
              modelList.append(fc1)
      combinedLayer = tf.concat(1,modelList)
      print('Done')

      combinedLayer = tf.reshape(combinedLayer, shape=[-1,len(inputList), shp, 1])
      weightsComb,biasComb, combShape = getCombinedConvParams(conf,len(inputList))
      convComb = conv2d(combinedLayer,weightsComb['Filters'],biasComb['BiasConv'])
      fc2 = tf.reshape(convComb, [-1,combShape*1*conf['combinedFilterAmount']])
      fc2 = tf.add(tf.matmul(fc2, weightsComb['FC']), biasComb['BiasFC'])
      fc2 = tf.nn.relu(fc2)
  else:
      inputName = inputList[0]
      print(conf)
      with tf.name_scope(inputName):
          weights,biases = getConvNetParams(inputName,conf)
          fc1,shp = makeSubModel(inputPlaceholders[0],conf[inputName],weights,biases)
  wdOut = tf.Variable(tf.random_normal([shp,conf['outputShape'][0]]),name='outFC')
  bdOut = tf.Variable(tf.zeros([conf['outputShape'][0]]),name='outBias')
  fc1 = tf.nn.dropout(fc1,conf['keepProb'])
  logits = tf.add(tf.matmul(fc1, wdOut), bdOut)
  return logits



def evaluation(logits,target,topK=50):

    correct = tf.nn.in_top_k(logits, target, topK)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.float32))

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, 1, k, 1], strides=[1,1, k, 1],
                          padding='VALID')


# Create model
def makeSubModel(x, conf, weights, biases):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, conf['height'],  conf['width'],  conf['channel']])
    # Convolution Layer
####3
    x = tf.nn.tanh(x)
#####
    conv1 = conv2d(x, weights['Filters1'], biases['BiasConv1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool(conv1, k=conf['poolSize'])

    # Convolution Layer
    # conv2 = conv2d(conv1, weights['Filters2'], biases['BiasConv2'])
    # Max Pooling (down-sampling)
    # conv2 = maxpool(conv2, k=conf['poolSize'])

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    w_shape = conf['width']-conf['filterWidth']+1
    w_shape = int((w_shape - conf['poolSize'])/conf['poolStride'] +1)
    # w_shape = int(w_shape)-conf['filterWidth']+1
    # w_shape = int((w_shape - conf['poolSize'])/conf['poolStride'] +1)

    fc1 = tf.reshape(conv1, [-1,w_shape*1*conf['filterAmount'][0]])
    # fc1 = tf.add(tf.matmul(fc1, weights['FC1']), biases['BiasFC1'])
    # fc1 = tf.nn.relu(fc1)

    return fc1, w_shape*1*conf['filterAmount'][0]


def getConvNetParams(inputName,conf):
    with tf.name_scope(inputName):
        w_shape = conf[inputName]['width']-conf[inputName]['filterWidth']+1
        w_shape = (w_shape - conf[inputName]['poolSize'])/conf[inputName]['poolStride'] +1
        w_shape = int(w_shape)-conf[inputName]['filterWidth']+1
        w_shape = int((w_shape - conf[inputName]['poolSize'])/conf[inputName]['poolStride'] +1)

        wc1 = tf.Variable(tf.random_normal([conf[inputName]['filterHeight'], conf[inputName]['filterWidth'], 1, conf[inputName]['filterAmount'][0]]),\
                name='Filters1')
        wc2 = tf.Variable(tf.random_normal([1, conf[inputName]['filterWidth'],conf[inputName]['filterAmount'][0], conf[inputName]['filterAmount'][1]]),\
        name='Filters2')
        wd1 = tf.Variable(tf.random_normal([w_shape*1*conf[inputName]['filterAmount'][1], conf['FC1width']]),\
        name='FC1')

        bc1 = tf.Variable(tf.zeros([conf[inputName]['filterAmount'][0]]),name='BiasConv1')
        bc2 = tf.Variable(tf.zeros([conf[inputName]['filterAmount'][1]]),name='BiasConv2')
        bd1 = tf.Variable(tf.zeros([ conf['FC1width']]),name='BiasFC1')

    weights = { 'Filters1':wc1,
                'Filters2':wc2,
                'FC1':wd1}
    biases = {  'BiasConv1':bc1,
                'BiasConv2':bc2,
                'BiasFC1':bd1}
    return weights, biases

def getCombinedConvParams(conf,numberOfInputs):

    wcComb= tf.Variable(tf.random_normal([numberOfInputs, conf['combinedFilterWidth'], 1, conf['combinedFilterAmount']]),name='CombinedFilters')
    bcComb = tf.Variable(tf.random_normal([conf['combinedFilterAmount']]),name='CombinedBiasConv')
    combShape = conf['FC1width']- conf['combinedFilterWidth'] +1

    wdComb = tf.Variable(tf.random_normal([combShape*1*conf['combinedFilterAmount'], 1024]),name='CombinedFC')
    bdComb = tf.Variable(tf.zeros([1024]),name='CombinedBiasFC')


    weights = { 'Filters':wcComb,
                'FC':wdComb}
    biases = {  'BiasConv':bcComb,
                'BiasFC':bdComb}

    return weights, biases, combShape
