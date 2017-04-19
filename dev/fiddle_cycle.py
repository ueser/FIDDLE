

cs_pred = tf.nn.softmax(DS_container_from_pred.representation, name='ds_softmax')

self.loss_estimator = kullback_leibler_divergence(self.inputs['chipseq'],cs_pred, name='KL_from_pred')

self.cost = tf.reduce_mean(self.loss_estimator)

self.optimizer = \
    tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost,
                                                                      global_step=self.global_step,
                                                                      var_list=self.trainables)


