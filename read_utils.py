import os
import re
import codecs
import numpy as np
import pickle
import random


def load_origin_data(x_file, y_file):
    train_x = codecs.open(x_file, encoding='utf-8').read()
    train_x = re.split('<qid_.*?\n', train_x)[:-1]
    train_x = ['\n'.join([l.split('||| ')[1] for l in re.split('\n+', t) if l.split('||| ')[0]]) for t in train_x]
    # train_x = [split_data(l) for l in train_x]

    train_y = codecs.open(y_file, encoding='utf-8').read()
    train_y = train_y.split('\n')[:-1]
    train_y = [l.split('||| ')[1] for l in train_y]
    return train_x, train_y

class TextConverter(object):
    def __init__(self, text=None, save_pkl=None, vec_size=128):
        if os.path.exists(save_pkl):
            with open(save_pkl, 'rb') as f:
                self.id2word, self.word2id, self.embedding_array = pickle.load(f)
        else:
            self.word_to_vec(text, vec_size, save_pkl)

    def split_data(self, text):
        words = re.split('[ \n]+', text)
        idx = words.index('XXXXX')
        return words[:idx], words[idx + 1:]

    def word_to_vec(self, text, vec_size, save_pkl):
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        from gensim.models import Word2Vec
        # print('正在对添加语料进行分词...')
        # additional = codecs.open('../additional.txt', encoding='utf-8').read().split('\n') #自行从网上爬的童话语料
        # additional = map(lambda s: jieba.lcut(s, HMM=False), additional)

        text = [self.split_data(l) for l in text]
        class data_for_word2vec:  # 用迭代器
            def __iter__(self):
                for x in text:
                    yield x[0]
                    yield x[1]
        if not os.path.exists('temp/word2vec.model'):
            word2vec = Word2Vec(data_for_word2vec(), size=vec_size, min_count=2, sg=2, negative=10, iter=10)
            word2vec.save('temp/word2vec.model')
        else:
            word2vec = Word2Vec.load('temp/word2vec.model')
        from collections import defaultdict
        self.id2word = {i + 1: j for i, j in enumerate(word2vec.wv.index2word)}
        self.word2id = defaultdict(int, {j: i for i, j in self.id2word.items()})
        self.embedding_array = np.array([word2vec[self.id2word[i + 1]] for i in range(len(self.id2word))])
        pickle.dump([self.id2word, self.word2id, self.embedding_array], open(save_pkl, 'wb'))

    @property
    def vocab_size(self):
        return len(self.id2word)

    def word_to_int(self, word):
        if word in self.word2id:
            return self.word2id[word]
        else:
            return 0

    def xy_to_number(self, text_x, text_y):
        lens = len(text_x)
        text_y = text_y[:lens]
        text_x = [self.split_data(l) for l in text_x]
        xy_numbers = [([self.word_to_int(w) for w in text_x[j][0]], [self.word_to_int(w) for w in text_x[j][1]], self.word_to_int(text_y[j])) for j in range(lens)]
        return xy_numbers

    def batch_generator(self, xy_numbers, batchsize):
        '''产生训练batch样本'''
        n = len(xy_numbers)
        while True:
            random.shuffle(xy_numbers)  # 打乱顺序
            for i in range(0, n, batchsize):
                batch_samples = xy_numbers[i:i + batchsize]

                len_l = np.array([len(xy[0]) for xy in batch_samples])
                len_r = np.array([len(xy[1]) for xy in batch_samples])
                query_l = np.array([xy[0] + [0] * (max(len_l) - len(xy[0])) for xy in batch_samples])
                query_r = np.array([xy[1] + [0] * (max(len_r) - len(xy[1])) for xy in batch_samples])
                y = np.array([[xy[2]] for xy in batch_samples])
                y = (np.hstack([query_l, query_r]) == y).astype(np.float32)
                yield query_l,len_l,query_r,len_r,y

    def generate_valid_samples(self,xy_numbers, batchsize):
        n = len(xy_numbers)
        val_g = []
        for i in range(0, n, batchsize):
            batch_samples = xy_numbers[i:i + batchsize]

            len_l = np.array([len(xy[0]) for xy in batch_samples])
            len_r = np.array([len(xy[1]) for xy in batch_samples])
            query_l = np.array([xy[0] + [0] * (max(len_l) - len(xy[0])) for xy in batch_samples])
            query_r = np.array([xy[1] + [0] * (max(len_r) - len(xy[1])) for xy in batch_samples])
            y_index = np.array([[xy[2]] for xy in batch_samples])
            y = (np.hstack([query_l, query_r]) == y_index).astype(np.float32)
            val_g.append( (query_l, len_l, query_r, len_r, y, y_index))
        return val_g

