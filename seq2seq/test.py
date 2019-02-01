#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 16:04:49 2019

@author: wj
"""
import tensorflow as tf
import train
import numpy as np
import pickle
import data_process
import logging
import utils
import time
import traceback
class test_model_old:
    def __init__(self):
        self.options=train.config()
        self.build()
    
    def build(self):
        source_ids=tf.placeholder(tf.int32,[2,64])
        target_ids=tf.placeholder(tf.int32,[2,66])
        source_mask=tf.placeholder(tf.int32,[2,64])
        target_mask=tf.placeholder(tf.int32,[2,66])
        logits=train.build_model(source_ids,target_ids,source_mask,target_mask,True,self.options)
        sess=tf.Session()
        sess.run(tf.initialize_all_variables())
        a=np.ones([2,64],np.int32)
        a[0][32:]=0
        b=np.ones([2,66],np.int32)
        b[0][20:]=0
        b[1][30:]=0
        l=sess.run(logits,{source_ids:a,
                           target_ids:b,
                           source_mask:a,
                           target_mask:b})
        print(l)
        print(l.shape)
def get_assign_map(tf_vars_dict):
   
    a=['embedding_weights:0',
       'bidirectional_rnn/fw/lstm_cell/kernel:0',
       'bidirectional_rnn/fw/lstm_cell/bias:0',
       'bidirectional_rnn/bw/lstm_cell/kernel:0',
       'bidirectional_rnn/bw/lstm_cell/bias:0',
       'rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0',
       'rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0',
       'rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:0',
       'rnn/multi_rnn_cell/cell_1/lstm_cell/bias:0',
       'memory_layer/kernel:0',
       'decoder/multi_rnn_cell/cell_0_attention/attention/lstm_cell/kernel:0',
       'decoder/multi_rnn_cell/cell_0_attention/attention/lstm_cell/bias:0',
       'decoder/multi_rnn_cell/cell_1/lstm_cell/kernel:0',
       'decoder/multi_rnn_cell/cell_1/lstm_cell/bias:0']
    assignment_map={i[:-2]:tf_vars_dict[i] for i in a}
    assignment_map.update({
            'dense/kernel':tf_vars_dict['decoder/dense/kernel:0'],
            'dense/bias':tf_vars_dict['decoder/dense/bias:0']
            })
    return assignment_map
class test_model:
    def __init__(self):
        self._pre_process()
        self.build()
    def build(self):
        self.model_file=tf.train.get_checkpoint_state('./out').model_checkpoint_path
        graph=tf.Graph()
        with graph.as_default():
            self.source_ids=tf.placeholder(tf.int32,[self.options.batch_size,self.options.max_source_length])
            self.target_ids=tf.placeholder(tf.int32,[self.options.batch_size,self.options.max_target_length])
            self.source_mask=tf.placeholder(tf.int32,[self.options.batch_size,self.options.max_source_length])
            self.target_mask=tf.placeholder(tf.int32,[self.options.batch_size,self.options.max_target_length])
            self.ouputs=train.build_model(self.source_ids,self.target_ids,self.source_mask,self.target_mask,False,self.options)
            sess_config=tf.ConfigProto()
            sess_config.gpu_options.allow_growth=True
            self.sess=tf.Session(graph=graph,config=sess_config)
            tf_vars_dict={i.name:i for i in tf.trainable_variables()}
            a_map=get_assign_map(tf_vars_dict)
            tf.train.init_from_checkpoint(self.model_file,a_map)
            self.sess.run(tf.global_variables_initializer())
            #a=np.ones([self.options.batch_size,self.options.max_source_length],np.int32,)
            #a=a*5
            #a=np.array(a,np.int32)
            #res=self.sess.run(self.ouputs,{self.source_ids:a,
            #                               self.source_mask:a})
            #print(res[0])
            #print(res.predicted_ids.shape)
    def predict_one_sentence(self,sentence):
        sentence_split=self.tokenizer.tokenize(sentence,self.options.source_language_type,self.options.source_language_lower)
        s_ids,s_mask=self.data_tools.encode(sentence_split)
        if len(s_ids)>self.options.max_source_length:
            s_ids=s_ids[:self.options.max_source_length]
            s_mask=s_mask[:self.options.max_source_length]
        while len(s_ids)<self.options.max_source_length:
            s_ids.append(0)
            s_mask.append(0)
        feed_dict={self.source_ids:[s_ids],
                   self.source_mask:[s_mask]}
        res=self.sess.run(self.ouputs,feed_dict)
        res=res.predicted_ids
        res=res[0]
        res=res.transpose()
        word_lists=[]
        for ids in res:
            word_lists.append(" ".join(self.decode_single(ids)))
        #word_list=self.decode_single(decode_ids)
        #return " ".join(word_list)
        return word_lists
    def test_interactive(self):
        while 1:
            q=input("Q:")
            if q:
                try:
                    t1=time.time()
                    word_list=self.predict_one_sentence(q)
                    for i,sent in enumerate(word_list):
                        print("A_{}:{}".format(i,sent))
                    t2=time.time()
                    print("cost time:{}".format(t2-t1))
                except:
                    traceback.print_exc()
                    break
            else:
                break
    def decode_single(self,decode_ids):
        no_pad_ids=[]
        for i,id_ in enumerate(decode_ids):
            if i==0:
                if id_==self.options.tgt_sos_id:
                    continue
            if id_==self.options.tgt_eos_id:
                break
            no_pad_ids.append(id_)
        word_list=[self.data_tools.id2word[i] for i in no_pad_ids]
        return word_list
    def _pre_process(self):
        self.options=train.config()
        with open('./out/options.pkl','rb') as f:
            opt=pickle.load(f)
            self.options.__dict__.update(opt)
            self.options.batch_size=1
        vocab_file='./data/vocab.txt'
        self.data_tools=data_process.Data(vocab_file,None,self.options,logging)
        self.tokenizer=utils.Tokenizer(logging)
if __name__=="__main__":
    model=test_model()
    model.test_interactive()
'''

'''