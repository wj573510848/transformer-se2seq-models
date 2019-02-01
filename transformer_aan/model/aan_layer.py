#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:20:36 2019

@author: wj
"""
import tensorflow as tf
from model import model_utils

def linear(inputs, output_size, bias, concat=True, dtype=None, scope=None):
    """
    Linear layer
    :param inputs: A Tensor or a list of Tensors with shape [batch, input_size]
    :param output_size: An integer specify the output size
    :param bias: a boolean value indicate whether to use bias term
    :param concat: a boolean value indicate whether to concatenate all inputs
    :param dtype: an instance of tf.DType, the default value is ``tf.float32''
    :param scope: the scope of this layer, the default value is ``linear''
    :returns: a Tensor with shape [batch, output_size]
    :raises RuntimeError: raises ``RuntimeError'' when input sizes do not
                          compatible with each other
    """

    with tf.variable_scope(scope, default_name="linear", values=[inputs]):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        input_size = [item.get_shape()[-1].value for item in inputs]

        if len(inputs) != len(input_size):
            raise RuntimeError("inputs and input_size unmatched!")

        output_shape = tf.concat([tf.shape(inputs[0])[:-1], [output_size]],
                                 axis=0)
        # Flatten to 2D
        inputs = [tf.reshape(inp, [-1, inp.shape[-1].value]) for inp in inputs]

        results = []

        if concat:
            input_size = sum(input_size)
            inputs = tf.concat(inputs, 1)

            shape = [input_size, output_size]
            matrix = tf.get_variable("matrix", shape, dtype=dtype)
            results.append(tf.matmul(inputs, matrix))
        else:
            for i in range(len(input_size)):
                shape = [input_size[i], output_size]
                name = "matrix_%d" % i
                matrix = tf.get_variable(name, shape, dtype=dtype)
                results.append(tf.matmul(inputs[i], matrix))

        output = tf.add_n(results)

        if bias:
            shape = [output_size]
            bias = tf.get_variable("bias", shape, dtype=dtype)
            output = tf.nn.bias_add(output, bias)

        output = tf.reshape(output, output_shape)

        return output
class AAN(tf.layers.Layer):
    def __init__(self,hidden_size, dropout, train):
        super(AAN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.train = train
        self.linear_layer = tf.layers.Dense(2*hidden_size, use_bias=True,
                                              name="ann_linear")
    def call(self,decoder_inputs,target_ids=None,cache=None):
        '''
        训练的时候，decoder_inputs是全部输入
        解码的时候，decoder_inputs是一个一个的输入，需要cache将每次的id记录下来，更新权重
        '''
        #cumulative-average operation
        ids=target_ids
        inputs=decoder_inputs
        if cache is not None:
            cache['k']=tf.concat([cache['k'],decoder_inputs],axis=1) #batch_size,length,hidden_size      
            #print(cache)
            cache['w']=cache['w']+tf.ones_like(cache['w']) #(1,)
        if ids is None:
            mask=cache['w']-tf.ones_like(cache['w']) 
            mask = tf.where(tf.less_equal(mask, 0.), tf.ones_like(mask), mask)
            mask = tf.cast(mask, tf.float32)
            y=tf.reduce_sum(cache['k'],axis=1)/mask[0]
            y=tf.expand_dims(y,axis=1)
        else:
            weights=model_utils.get_aan_weights(ids)
            y = tf.matmul(weights, inputs)
        
        # Gating layer
        z=self.linear_layer(tf.concat([inputs,y],axis=-1))
        i, f = tf.split(z, [self.hidden_size, self.hidden_size], axis=-1)
        y = tf.sigmoid(i) * inputs + tf.sigmoid(f) * y
        
        if self.train:
            y=tf.nn.dropout(y,1-self.dropout)
        return inputs+y

        
    