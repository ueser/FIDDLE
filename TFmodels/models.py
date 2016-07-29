import tensorflow as tf
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
    conv1 = conv2d(x, weights['Filters1'], biases['BiasConv1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool(conv1, k=conf['poolSize'])

    # Convolution Layer
    conv2 = conv2d(conv1, weights['Filters2'], biases['BiasConv2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool(conv2, k=conf['poolSize'])

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    w_shape = conf['width']-conf['filterWidth']+1
    w_shape = (w_shape - conf['poolSize'])/conf['poolStride'] +1
    w_shape = w_shape-conf['filterWidth']+1
    w_shape = (w_shape - conf['poolSize'])/conf['poolStride'] +1

    fc1 = tf.reshape(conv2, [-1,w_shape*1*conf['filterAmount'][1]])
    fc1 = tf.add(tf.matmul(fc1, weights['FC1']), biases['BiasFC1'])
    fc1 = tf.nn.relu(fc1)

    return fc1


def getConvNetParams(inputName,conf):
    with tf.name_scope(inputName):
        w_shape = conf[inputName]['width']-conf[inputName]['filterWidth']+1
        w_shape = (w_shape - conf[inputName]['poolSize'])/conf[inputName]['poolStride'] +1
        w_shape = w_shape-conf[inputName]['filterWidth']+1
        w_shape = (w_shape - conf[inputName]['poolSize'])/conf[inputName]['poolStride'] +1

        wc1 = tf.Variable(tf.random_normal([conf[inputName]['filterHeight'], conf[inputName]['filterWidth'], 1, conf[inputName]['filterAmount'][0]]),name='Filters1')
        wc2 = tf.Variable(tf.random_normal([1, conf[inputName]['filterWidth'],conf[inputName]['filterAmount'][0], conf[inputName]['filterAmount'][1]]),name='Filters2')
        wd1 = tf.Variable(tf.random_normal([w_shape*1*conf[inputName]['filterAmount'][1], conf['FC1width']]),name='FC1')

        bc1 = tf.Variable(tf.random_normal([conf[inputName]['filterAmount'][0]]),name='BiasConv1')
        bc2 = tf.Variable(tf.random_normal([conf[inputName]['filterAmount'][1]]),name='BiasConv2')
        bd1 = tf.Variable(tf.random_normal([ conf['FC1width']]),name='BiasFC1')

    weights = { 'Filters1':wc1,
                'Filters2':wc2,
                'FC1':wd1}
    biases = {  'BiasConv1':bc1,
                'BiasConv2':bc2,
                'BiasFC1':bd1}
    return weights, biases
