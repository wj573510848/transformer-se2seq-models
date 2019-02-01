#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 10:20:27 2019

@author: wj
"""
import collections
import tensorflow as tf
import os
import glob
import re
from utils import read_lines,write_lines,create_vocabulary,load_vocabulary

def parse_raw_file(file):
    '''
    将原始文件解析为qa对
    原始文件格式:
    Q: hello
    Q: hi
    A: hello
    A: hi
    
    Q: ...
    A: ...
    yield: [q,a]
    '''
    qa={'q':set(),'a':set()}
    for line in read_lines(file,yield_null=True):
        if not line:
            if len(qa['q'])>0 and len(qa['a'])>0:
                for q in qa['q']:
                    for a in qa['a']:
                        yield [q,a]
            qa={'q':set(),'a':set()}
        elif line[0]=='Q':
            if line[3:]:
                qa['q'].add(line[3:])
        elif line[0]=='A':
            if line[3:]:
                qa['a'].add(line[3:])
    if len(qa['q'])>0 and len(qa['a'])>0:
        for q in qa['q']:
            for a in qa['a']:
                yield [q,a]
    qa={'q':set(),'a':set()}
def tokenize_one_line(sentence,cut_fun,specical_symbol,mode,lower):
    raw_sentence=sentence
    tokenized_sentence=[]
    if specical_symbol:
        sentence=re.split("(<[a-zA-Z]+>)",sentence)
        for sub_sent in sentence:
            if re.search("^<[a-zA-Z]+>$",sub_sent):
                tokenized_sentence.append(sub_sent)
            else:
                if sub_sent:
                    tokenized_sentence.extend(cut_fun(sub_sent,mode,lower))
    else:
        tokenized_sentence=cut_fun(raw_sentence,mode,lower)
    return cut_white_space(" ".join(tokenized_sentence))
def cut_white_space(sentence):
    return " ".join(sentence.split())
class Data:
    def __init__(self,vocab_file,sample_file,config,logger):
        self.logger=logger
        self.config=config
        self.sample_file=sample_file
        self.word2id,self.id2word=load_vocabulary(vocab_file)
        self.tf_record_file=os.path.join(self.config.tokenized_data_dir,'sample.tf_record')
        self.pad_id=self.word2id['<pad>']
        self.unk_id=self.word2id['<unk>']
    def tf_dateset(self):
        self.create_tf_record_file(self.sample_file)
        name_to_features = {
                "q_ids": tf.FixedLenFeature([self.config.max_source_length], tf.int64),
                "a_ids": tf.FixedLenFeature([self.config.max_target_length], tf.int64),
                "q_mask": tf.FixedLenFeature([self.config.max_source_length], tf.int64),
                "a_mask": tf.FixedLenFeature([self.config.max_target_length], tf.int64),
                }
        def _decode_record(record, name_to_features):
            """Decodes a record to a TensorFlow example."""
            example = tf.parse_single_example(record, name_to_features)
            return example
        def input_fn():
            """The actual input function."""
            d = tf.data.TFRecordDataset(self.tf_record_file)
            d=d.map(lambda record: _decode_record(record, name_to_features))
            d = d.repeat(self.config.num_epochs)
            d = d.shuffle(buffer_size=10000)
            d = d.batch(self.config.batch_size)
            return d
        #test
        #iterator = d.make_one_shot_iterator()
        #features = iterator.get_next()
        return input_fn
    def create_tf_record_file(self,sample_file):
        '''
        将qa转化为id，并且a添加<s></s>
        '''
        save_file=self.tf_record_file
        if os.path.isfile(save_file):
            self.logger.info('tf record file "{}" existed!'.format(save_file))
            return
        tf_writer = tf.python_io.TFRecordWriter(save_file)
        self.logger.info("Writing example ...")
        num=0
        for line in read_lines(sample_file):
            q_line,a_line=line.split('\t')
            q_words=q_line.split()
            a_words=a_line.split()
            if len(q_words)>self.config.max_source_length:
                q_words=q_words[:self.config.max_source_length]
            if len(a_words)>self.config.max_target_length-2:
                a_words=a_words[:self.config.max_target_length-2]
            a_words=['<s>']+a_words+['</s>']
            q_ids,q_mask=self.encode(q_words)
            a_ids,a_mask=self.encode(a_words)
            while len(q_ids)<self.config.max_source_length:
                q_ids.append(self.pad_id)
                q_mask.append(0)
            while len(a_ids)<self.config.max_target_length:
                a_ids.append(self.pad_id)
                a_mask.append(0)
            #print(a_words)
            #print(q_words)
            assert len(q_ids)==self.config.max_source_length
            assert len(a_ids)==self.config.max_target_length
            assert len(q_mask)==self.config.max_source_length
            assert len(a_mask)==self.config.max_target_length
            features = collections.OrderedDict()
            features["q_ids"] = self.create_int_feature(q_ids)
            features['q_mask']= self.create_int_feature(q_mask)
            features["a_ids"] = self.create_int_feature(a_ids)
            features["a_mask"] = self.create_int_feature(a_mask)
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            tf_writer.write(tf_example.SerializeToString())
            num+=1
            if num<=5:
                self.logger.info("*** example {} ***".format(num))
                self.logger.info("source words:{}".format(q_words))
                self.logger.info("source ids:{}".format(q_ids))
                self.logger.info("source mask:{}".format(q_mask))
                self.logger.info("target words:{}".format(a_words))
                self.logger.info("target ids:{}".format(a_ids))
                self.logger.info("target mask:{}".format(a_mask))
            if num%100000==0:
                self.logger.info("write sample:{}".format(num))
        self.logger.info("Done! Total examples:{}".format(num))
    def create_int_feature(self,values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f
    def encode(self,word_list):
        ids=[self.word2id.get(i,self.unk_id) for i in word_list]
        mask=[1]*len(ids)
        if self.unk_id in ids:
            self.logger.warn("unknown word in {}".format(word_list))
        return ids,mask
    @property
    def vocab_size(self):
        return len(self.word2id)
    @property
    def eos_id(self):
        return self.word2id['</s>']
    @property
    def sos_id(self):
        return self.word2id['<s>']
    @staticmethod
    def pre_process_data(raw_data,tokenizer,config,logger):
        '''
        raw_data: dir or a specific file
        '''
        vocab_file=os.path.join(config.tokenized_data_dir,'vocab.txt')
        sample_file=os.path.join(config.tokenized_data_dir,'samples.txt')
        if os.path.isfile(vocab_file) and os.path.isfile(sample_file):
            logger.info("vocab file and sample file already existed!")
            return Data(vocab_file,sample_file,config,logger)
        else:
            logger.info("Genarate vocabulary and tokenized samples.")
            if os.path.isfile(raw_data):
                raw_data=[raw_data]
            else:
                raw_data=glob.glob(os.path.join(raw_data,'*'))
            samples=set()
            for file in raw_data:
                for qa in parse_raw_file(file):
                    q=qa[0]
                    a=qa[1]
                    tokenized_q=tokenize_one_line(
                            sentence=q,
                            cut_fun=tokenizer.tokenize,
                            specical_symbol=config.special_symbol,
                            mode=config.source_language_type,
                            lower=config.source_language_lower)
                    tokenized_a=tokenize_one_line(
                            sentence=a,
                            cut_fun=tokenizer.tokenize,
                            specical_symbol=config.special_symbol,
                            mode=config.target_language_type,
                            lower=config.target_language_lower)
                    samples.add(tokenized_q+"\t"+tokenized_a)
            logger.info('sample size:{}'.format(len(samples)))
            logger.info("save samples in '{}'".format(sample_file))
            write_lines(sample_file,samples)
            source_vocab,target_vocab,special_vocab=create_vocabulary(samples,config.special_symbol)
            source_vocab=set(list(source_vocab.keys()))
            for s_symbol in config.vocab_remains:
                if s_symbol in source_vocab:
                    source_vocab.discard(s_symbol)
                if s_symbol in target_vocab:
                    target_vocab.discard(s_symbol)
                if s_symbol in special_vocab:
                    special_vocab.discard(s_symbol)
            logger.info('vocab size:{}'.format(len(source_vocab)+len(target_vocab)+len(special_vocab)+len(config.vocab_remains)))
            logger.info('save vocabulary in "{}"'.format(vocab_file))
            with open(vocab_file,'w',encoding='utf8') as f:
                for line in config.vocab_remains:
                    f.write(line+'\n')
                for line in special_vocab:
                    f.write(line+'\n')
                for line in source_vocab|target_vocab:
                    f.write(line+'\n')
            return Data(vocab_file,sample_file,config,logger)
def create_train_data(data_dir,config):
    from utils import Tokenizer,get_logger
    logger=get_logger('log','./log/log.txt')
    t=Tokenizer(logger)
    model=Data.pre_process_data(data_dir,t,config,logger)
    model.create_tf_record_file(model.sample_file)
    return model
    #data=d()
    #num=0
    #i=data.make_initializable_iterator()
    #while i.get_next():
    #    num+=1
    #    if num%100000==0:
    #        print(num)
    #print(num)
if __name__=="__main__":
    import basic_model
    config=basic_model.config()
    create_train_data("/home/data/aoshuo/dnlu_4/corpora/text/zh-cn/faq/train/super_checked",config)
            