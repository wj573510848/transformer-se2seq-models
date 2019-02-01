#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:05:12 2019

@author: wj
"""
import tensorflow as tf
import logging
import models
import data_process
import os
import pickle

class config:
    def __init__(self):
        self.raw_data_dir='./raw_data'
        self.model_save_dir='./out'
        self.log_dir='./log'
        self.tokenized_data_dir='./data'
        self.max_source_length=64
        self.max_target_length=66
        
        self.source_language_type='cn'
        self.target_language_type='cn'
        self.special_symbol=True #是否有特殊符号"<[a-zA-Z]+>"作为特殊符号
        self.source_language_lower=True #是否将英文统一为小写
        self.target_language_lower=True
        self.vocab_remains=['<pad>','<unk>','<s>','</s>'] 
        
        self.num_epochs=200
        
        
        self.vocab_size=''
        self.hidden_size=256
        self.keep_prob=0.7
        self.num_encoder_layers=3
        self.num_decoder_layers=2
        self.beam_width=4
        self.length_penalty_weight=0   
        self.tgt_sos_id=2
        self.tgt_eos_id=3
        self.max_decoder_length=66
        self.batch_size=32
        
        self.learning_rate = 2.0
        self.learning_rate_decay_rate = 1.0
        self.learning_rate_warmup_steps = 10000
        
        # Optimizer params
        self.optimizer_adam_beta1 = 0.9
        self.optimizer_adam_beta2 = 0.997
        self.optimizer_adam_epsilon = 1e-09
        
        self.num_sampled_softmax=50000
        
def build_model(source_ids,target_ids,source_mask,target_mask,is_training,config):
    #使用共享embedding
    with tf.device("cpu:0"):
        embedding_table = tf.get_variable(
          "embedding_weights", [config.vocab_size, config.hidden_size],
          initializer=tf.random_normal_initializer(
              0., config.hidden_size ** -0.5))
    encoder_inputs=tf.nn.embedding_lookup(
            embedding_table, source_ids)
    encoder_inputs *= config.hidden_size ** 0.5
    encoder_seq_length=tf.reduce_sum(source_mask,axis=-1)
    encoder_outputs,encoder_state=models.encoder_cells(
            config.hidden_size,config.keep_prob,is_training,encoder_inputs,encoder_seq_length,
         config.num_encoder_layers,logger=logging)
    if is_training:
        target_input_ids=target_ids[:,:-1]
        target_input_mask=target_mask[:,:-1]
        target_output_ids=target_ids[:,1:]
        target_output_mask=target_mask[:,1:]
        #target_input_ids=target_ids
        #target_input_mask=target_mask
        #target_output_ids=target_ids
        #target_output_mask=target_mask
        decoder_inputs=tf.nn.embedding_lookup(
            embedding_table, target_input_ids)
        decoder_seq_length=tf.reduce_sum(target_input_mask,axis=-1)
        decoder_seq_length=tf.cast(decoder_seq_length,dtype=tf.int32)
        logits,decoder_cell_outputs,output_layer=models.decoder_cells(encoder_outputs,encoder_state,encoder_seq_length,
                      config.num_decoder_layers,is_training,config.hidden_size,
                      config.beam_width,config.length_penalty_weight,
                      config.batch_size,decoder_inputs,decoder_seq_length,
                      config.vocab_size,config.tgt_sos_id,config.tgt_eos_id,
                      embedding_table,config.max_decoder_length,config.keep_prob)
        target_output_ids=target_output_ids[:,:tf.reduce_max(decoder_seq_length)]
        
        
        if config.num_sampled_softmax > 0:
            is_sequence = (decoder_cell_outputs.shape.ndims == 3)
            if is_sequence:
                labels = tf.reshape(target_output_ids, [-1, 1])
                inputs = tf.reshape(decoder_cell_outputs, [-1, config.hidden_size])

            crossent = tf.nn.sampled_softmax_loss(
                  weights=tf.transpose(output_layer.kernel),
                  biases=output_layer.bias or tf.zeros([config.vocab_size]),
                  labels=labels,
                  inputs=inputs,
                  num_sampled=config.num_sampled_softmax,
                  num_classes=config.vocab_size,
                  partition_strategy="div",)

            if is_sequence:
                crossent = tf.reshape(crossent, [config.batch_size, -1])
        
        #loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output_ids, logits=logits)
        loss = loss = tf.reduce_sum(crossent ) / tf.to_float(config.batch_size)
        return loss
    else:
        outputs=models.decoder_cells(encoder_outputs,encoder_state,encoder_seq_length,
                      config.num_decoder_layers,False,config.hidden_size,
                      config.beam_width,config.length_penalty_weight,
                      config.batch_size,None,None,
                      config.vocab_size,config.tgt_sos_id,config.tgt_eos_id,
                      embedding_table,config.max_decoder_length,config.keep_prob)
        return outputs
def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps):
    """Calculate learning rate with linear warmup and rsqrt decay."""
    with tf.name_scope("learning_rate"):
        warmup_steps = tf.to_float(learning_rate_warmup_steps)
        step = tf.to_float(tf.train.get_or_create_global_step())

        learning_rate *= (hidden_size ** -0.5)
        # Apply linear warmup
        learning_rate *= tf.minimum(1.0, step / warmup_steps)
        # Apply rsqrt decay
        learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))        
        return learning_rate
def get_train_op(loss, config):
    """Generate training operation that updates variables based on loss."""
    with tf.variable_scope("get_train_op"):
        learning_rate = get_learning_rate(
                config.learning_rate, config.hidden_size,
                config.learning_rate_warmup_steps)

        # Create optimizer. Use LazyAdamOptimizer from TF contrib, which is faster
        # than the TF core Adam optimizer.
        optimizer = tf.contrib.opt.LazyAdamOptimizer(
                learning_rate,
                beta1=config.optimizer_adam_beta1,
                beta2=config.optimizer_adam_beta2,
                epsilon=config.optimizer_adam_epsilon)

        # Calculate and apply gradients using LazyAdamOptimizer.
        global_step = tf.train.get_global_step()
        tvars = tf.trainable_variables()
        gradients = optimizer.compute_gradients(
                loss, tvars, colocate_gradients_with_ops=True)
        train_op = optimizer.apply_gradients(
                gradients, global_step=global_step, name="train")
    return train_op
def model_fn_builder(config):
    def model_fn(features, labels, mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            print(features)
            print(labels)
            source_ids=features['q_ids']
            target_ids=features['a_ids']
            source_mask=features['q_mask']
            target_mask=features['a_mask']
            loss = build_model(source_ids,target_ids,source_mask,target_mask,True,config)
            train_op= get_train_op(loss, config)
            for var in tf.trainable_variables():
                tf.logging.info(var)
            output_spec=tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
        return output_spec
    return model_fn
def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    
    options=config()
    
    tf.gfile.MakeDirs(options.model_save_dir)
    tf.gfile.MakeDirs(options.tokenized_data_dir)
    tf.gfile.MakeDirs(options.log_dir)
    
    data_model=data_process.create_train_data(options.raw_data_dir,options)
    options.vocab_size=data_model.vocab_size
    options.tgt_eos_id=data_model.eos_id
    options.tgt_sos_id=data_model.sos_id
    log_steps=200
    do_train=True
    
    
    session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    run_config=tf.estimator.RunConfig(
            model_dir=options.model_save_dir,
            log_step_count_steps=log_steps,
            session_config=session_config)
    
    model_fn=model_fn_builder(options)
    estimator=tf.estimator.Estimator(model_fn=model_fn,model_dir=options.model_save_dir,config=run_config)
    if do_train:
        option_file=os.path.join(options.model_save_dir,'options.pkl')
        with open(option_file,'wb') as f:
            pickle.dump(options.__dict__,f,-1)
        tf.logging.info("*** options ***")
        for key in options.__dict__:
            tf.logging.info("\t{}:{}".format(key,options.__dict__[key]))
        estimator.train(input_fn=data_model.tf_dateset())
if __name__=="__main__":
    main()