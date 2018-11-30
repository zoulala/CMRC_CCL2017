import os
import tensorflow as tf
from read_utils import TextConverter
# from model_cnn import Model,Config
from model import Model,Config
from read_utils import load_origin_data

def main(_):

    model_path = os.path.join('models', Config.file_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)

    vocab_file = os.path.join(model_path, 'vocab_label.pkl')

    # 获取原始数据
    train_x_file = 'data/train/train.doc_query.part.0'
    train_y_file = 'data/train/train.answer'
    valid_x_file = 'data/validation/cloze.valid.doc_query'
    valid_y_file = 'data/validation/cloze.valid.answer'

    # 分配训练和验证数据集
    print('正在读取训练和验证语料...')
    train_x, train_y = load_origin_data(train_x_file,train_y_file)
    valid_x, valid_y = load_origin_data(valid_x_file, valid_y_file)

    # 数据处理
    converter = TextConverter(train_x+valid_x, vocab_file)
    print('vocab size:',converter.vocab_size)


    # 产生训练样本
    train_xy = converter.xy_to_number(train_x,train_y)
    train_g = converter.batch_generator(train_xy, Config.batch_size)

    # 产生验证样本
    valid_xy = converter.xy_to_number(valid_x, valid_y)
    val_g = converter.generate_valid_samples(valid_xy, Config.batch_size)

    # 加载上一次保存的模型
    model = Model(Config,converter.vocab_size, converter.embedding_array)
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    if checkpoint_path:
        model.load(checkpoint_path)

    print('start to training...')
    model.train(train_g, model_path, val_g)



if __name__ == '__main__':
    tf.app.run()
