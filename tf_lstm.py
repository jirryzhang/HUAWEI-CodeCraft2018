# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from sklearn import preprocessing

input_size = 1
hidden_unit = 100
output_size = 1
alpha = 0.006
time_step = 7
standardlization = 1
x_train, y_train = [], []
x_test, y_test = [], []

tf.reset_default_graph()

class LSTM:
    
    def __init__(self, data):
        self.data = np.array(data)
        self.w_in = tf.Variable(tf.random_normal([input_size, hidden_unit]))
        self.w_out = tf.Variable(tf.random_normal([hidden_unit, output_size]))
        
        self.b_in = tf.Variable(tf.constant(0.1, shape=[hidden_unit, ]))
        self.b_out = tf.Variable(tf.constant(0.1, shape=[output_size, ]))
        
    def data_preprocess(self, batch_size=50, train_begin=0, train_end=50):
        global x_train, y_train
        global x_test, y_test
        
        for i in range(train_begin, train_end-time_step):
            x = self.data[i:i+time_step, None]
            y = self.data[i+time_step:i+time_step+1, None]
            x_train.append(x.tolist())
            y_train.append(y.tolist())
        for i in range(train_end-time_step,len(self.data)-time_step):
            x = self.data[i:i+time_step, None]
            y = self.data[i+time_step:i+time_step+1, None]
            x_test.append(x.tolist())
            y_test.append(y.tolist())
        
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
    
    
    def model(self):
        global x_train, y_train
        global x_test, y_test
        
        X = tf.placeholder(tf.float32, [None, time_step, input_size])
        y = tf.placeholder(tf.float32, [None, 1, output_size])
        
        # layer_in_x = tf.reshape(X, [-1, input_size])
        # layer_in_y = tf.matmul(self.w_in, layer_in_x) + self.b_in
        # layer_in_y = tf.reshape(layer_in_y, [-1, time_step, input_size])
        
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_unit,
                                            activation=tf.nn.sigmoid,
                                            forget_bias=1,
                                            state_is_tuple=True)
        
        rnn_output, final_state = tf.nn.dynamic_rnn(cell,
                                                    X,
                                                    dtype=tf.float32)
        
        stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden_unit])
        stacked_output = tf.layers.dense(stacked_rnn_output, output_size)
        output = tf.reshape(stacked_output, [-1, time_step, output_size])
        
        loss = tf.reduce_sum(tf.square(output[:, -1, :] - y))
        optimizer = tf.train.AdamOptimizer(learning_rate=alpha)
        training_op = optimizer.minimize(loss)
        
        init = tf.global_variables_initializer()
        
        epochs = 1000
        with tf.Session() as sess:
            init.run()
            for ep in range(epochs):
                sess.run(training_op, feed_dict={X: x_train, y: y_train})
                if ep % 100 == 0:
                    mse = loss.eval(feed_dict={X: x_train, y: y_train})
                    print(ep, '\tMSE: ', mse)
            y_pred = sess.run(output, feed_dict = {X: x_train})
            print(sum(y_pred) / standardlization)

    