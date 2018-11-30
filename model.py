
"""

"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os


class Config(object):
    """RNN配置参数"""
    file_name = 'rnn'  # 保存模型文件

    use_embedding = True  # 是否用词向量，否则one_hot
    embedding_dim = 60  # 词向量维度

    num_layers= 1           # 隐藏层层数
    hidden_dim = 64  # 隐藏层神经元

    train_keep_prob = 0.8  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 60  # 每批训练大小
    max_steps = 20000  # 总迭代batch数

    log_every_n = 10  # 每多少轮输出一次结果
    save_every_n = 50  # 每多少轮校验模型并保存


class Model(object):
    def __init__(self, config, vocab_size, embedding_array):
        self.config = config
        self.vocab_size = vocab_size
        self.embedding = embedding_array

        # 待输入的数据
        self.query_seqs = tf.placeholder(tf.int32, [None, None], name='query')
        self.query_length = tf.placeholder(tf.int32, [None], name='query_length')

        self.response_seqs = tf.placeholder(tf.int32, [None, None], name='response')
        self.response_length = tf.placeholder(tf.int32, [None], name='response_length')

        self.targets = tf.placeholder(tf.float32, shape=[None, None], name='targets')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # 两个全局变量
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.global_loss = tf.Variable(0, dtype=tf.float32, trainable=False, name="global_loss")

        # Ann模型
        self.rnn()

        # 初始化session
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())


    def rnn(self):
        """rnn模型"""
        def get_mul_cell(hidden_dim, num_layers):# 创建多层lstm
            def get_en_cell(hidden_dim):# 创建单个lstm
                enc_base_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, forget_bias=1.0)
                return enc_base_cell
            return tf.nn.rnn_cell.MultiRNNCell([get_en_cell(hidden_dim) for _ in range(num_layers)])

        def bilstm(hidm, seq, seq_len, num_layers=1):
            cell_fw = get_mul_cell(hidm, num_layers)
            cell_bw = get_mul_cell(hidm, num_layers)
            (output_fw, output_bw), state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, seq,sequence_length=seq_len,dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            return output, state

        # 词嵌入层
        if self.config.use_embedding is False:
            self.lstm_query_seqs = tf.one_hot(self.query_seqs,depth=self.vocab_size)  # 独热编码[1,2,3] depth=5 --> [[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0]]，此时的输入节点个数为num_classes
            self.lstm_response_seqs = tf.one_hot(self.response_seqs, depth=self.vocab_size)
        else:
            embedding = tf.get_variable('embedding', [self.vocab_size, self.config.embedding_dim])  # 可训练的词向量
            # embedding = tf.constant(self.embedding, dtype=tf.float32)  # 用自定义词向量
            embedding_zero = tf.constant(0, dtype=tf.float32, shape=[1, self.config.embedding_dim])
            embedding = tf.concat([embedding, embedding_zero], axis=0)  # 增加一行0向量，代表padding向量值
            self.lstm_query_seqs = tf.nn.embedding_lookup(embedding,self.query_seqs)  # 词嵌入[1,2,3] --> [[3,...,4],[0.7,...,-3],[6,...,9]],embeding[depth*embedding_size]=[[0.2,...,6],[3,...,4],[0.7,...,-3],[6,...,9],[8,...,-0.7]]，此时的输入节点个数为embedding_size
            self.lstm_response_seqs = tf.nn.embedding_lookup(embedding, self.response_seqs)
            self.lstm_query_seqs = tf.nn.dropout(self.lstm_query_seqs, keep_prob=self.keep_prob)
            self.lstm_response_seqs = tf.nn.dropout(self.lstm_response_seqs, keep_prob=self.keep_prob)

        with tf.variable_scope("lstm_layer1") as scope:
            # 第一层bilstm网络
            query_output, self.query_state = bilstm(self.config.hidden_dim, self.lstm_query_seqs, self.query_length, self.config.num_layers)
            scope.reuse_variables()  # bilstm共享
            response_output, self.response_state = bilstm(self.config.hidden_dim, self.lstm_response_seqs, self.response_length, self.config.num_layers)
        with tf.variable_scope("lstm_layer2") as scope:
            # 第二层bilstm网络，第一层的输出作为第二层的输入
            query_output, self.query_state = bilstm(self.config.hidden_dim, query_output, self.query_length, self.config.num_layers)
            scope.reuse_variables()  # bilstm共享，但和第一层lstm_layer1不共享。
            response_output, self.response_state = bilstm(self.config.hidden_dim, response_output, self.response_length, self.config.num_layers)


            query_c_fw, query_h_fw = self.query_state[0][-1]  # 前向最后一层c/h
            query_c_bw, query_h_bw = self.query_state[1][-1]  # 后向最后一层c/h
            response_c_fw, response_h_fw = self.response_state[0][-1]
            response_c_bw, response_h_bw = self.response_state[1][-1]

            self.query_h_state = tf.concat([query_h_fw, query_h_bw], axis=1)
            self.response_h_state = tf.concat([response_h_fw, response_h_bw], axis=1)

            outputs = tf.concat([query_output, response_output], 1)
            final_state = (self.query_h_state + self.response_h_state) / 2  # 双向拼接、上下文取平均，得到encode向量

            L_context_mask = (1 - tf.cast(tf.sequence_mask(self.query_length), tf.float32)) * (
            -1e12)  # 对填充位置进行mask，注意这里是softmax之前的mask，所以mask不是乘以0，而是减去1e12
            R_context_mask = (1 - tf.cast(tf.sequence_mask(self.response_length), tf.float32)) * (-1e12)
            context_mask = tf.concat([L_context_mask, R_context_mask], 1)
            attention = context_mask + tf.matmul(outputs, tf.expand_dims(final_state, 2))[:, :,0]  # encode向量与每个时间步状态向量做内积，然后mask，然后softmax


        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            self.losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.targets, logits=attention)
            self.mean_loss = tf.reduce_mean(self.losses, name="mean_loss")  # batch样本的平均损失
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.mean_loss,
                                                                                                  global_step=self.global_step)

        with tf.name_scope("score"):
            self.y_pre = tf.nn.softmax(attention)
            # self.y_cos = self.logits
            # self.y_pre = tf.to_int32((tf.sign(self.y_cos * 2 - 1) + 1) / 2)

    def load(self, checkpoint):
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))

    def train(self, batch_train_g, model_path, val_g):
        with self.session as sess:
            for q, q_len, r, r_len, y in batch_train_g:
                start = time.time()
                feed = {self.query_seqs: q,
                        self.query_length: q_len,
                        self.response_seqs: r,
                        self.response_length: r_len,
                        self.targets: y,
                        self.keep_prob: self.config.train_keep_prob}
                batch_loss, _, query_h_state,lstm_query_seqs= sess.run([self.mean_loss, self.optim,self.query_h_state,self.lstm_query_seqs],feed_dict=feed)

                end = time.time()

                # control the print lines
                if self.global_step.eval() % self.config.log_every_n == 0:
                    print('step: {}/{}... '.format(self.global_step.eval(), self.config.max_steps),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))

                if (self.global_step.eval() % self.config.save_every_n == 0):
                    rk = 0.
                    lens_all = 0.
                    def cumsum_proba(x, y):  # 对相同项的概率进行合并
                        tmp = {}
                        for i, j in zip(x, y):
                            if i in tmp:
                                tmp[i] += j
                            else:
                                tmp[i] = j
                        return list(tmp.keys())[np.argmax(list(tmp.values()))]

                    for q, q_len, r, r_len, y ,y_index in val_g:
                        feed = {self.query_seqs: q,
                                self.query_length: q_len,
                                self.response_seqs: r,
                                self.response_length: r_len,
                                self.targets: y,
                                self.keep_prob: 1}

                        p = sess.run(self.y_pre, feed_dict=feed)
                        w = np.hstack([q, r])
                        rk += (np.array([cumsum_proba(s, t) for s, t in zip(w, p)]) == y_index.reshape(-1)).sum()
                        lens_all += len(y)
                    acc = rk / lens_all

                    # 计算预测准确率
                    print('val len:', lens_all)
                    print("accuracy:{:.2f}%.".format(acc * 100),
                          'best:{:.2f}%'.format(self.global_loss.eval()* 100))

                    if acc > self.global_loss.eval():
                        print('save best model...')
                        update = tf.assign(self.global_loss, acc)  # 更新最优值
                        sess.run(update)
                        self.saver.save(sess, os.path.join(model_path, 'model'), global_step=self.global_step)

                if self.global_step.eval() >= self.config.max_steps:
                    break



