import argparse
import json
import os
import sys

current_folder = os.path.dirname(os.path.abspath(__file__))
# 添加 configs 文件夹的路径到 Python 路径
configs_folder = os.path.join(current_folder, "..")  # 需要根据实际目录结构调整
sys.path.append(configs_folder)

import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from configs import config
from CNN_Model import CNN_Model
from DataProcessor import DataProcessor
from utils.evaluate_util import get_metrics
from utils.show_util import showToolName, showTrainParams
import datetime



class Trainer:
    def __int__(self):
        pass

    def train(self, X, y, cnn_num_convs, cnn_filters_array):
        y_one_hot = np_utils.to_categorical(y, int(config.class_num))
        y_one_hot = np.array(y_one_hot)

        cnn_model = CNN_Model(config.X_feature_len, config.class_num)
        model = cnn_model.build_model(cnn_num_convs, cnn_filters_array)

        # 训练模型
        model.fit(X, y_one_hot, batch_size=config.batch_size, epochs=config.epochs, verbose=1)
        # 保存模型
        i = datetime.datetime.now()
        time_str = str(i.date()) + '.' + str(i.hour) + '-' + str(i.minute) + '-' + str(i.second)
        model_path = config.project_dir + '/models/' + f'model_' + time_str + '.h5'
        model.save(model_path)
        return model_path


def main():
    showToolName()

    # 1.parse args
    describe_info = '########################## NeuralTE, version ' + str(config.version_num) + ' ##########################'
    parser = argparse.ArgumentParser(description=describe_info)
    parser.add_argument('--data', metavar='data', help='Input data used to train model, fasta format')
    parser.add_argument('--use_terminal', metavar='use_terminal', help='Whether to use LTR, TIR terminal features, 1: true, 0: false. default = [ ' + str(config.use_terminal) + ' ]')
    parser.add_argument('--use_TSD', metavar='use_TSD', help='Whether to use TSD features, 1: true, 0: false. default = [ ' + str(config.use_TSD) + ' ]')
    parser.add_argument('--use_domain', metavar='use_domain', help='Whether to use domain features, 1: true, 0: false. default = [ ' + str(config.use_domain) + ' ]')
    parser.add_argument('--use_ends', metavar='use_ends', help='Whether to use 5-bp terminal ends features, 1: true, 0: false. default = [ ' + str(config.use_ends) + ' ]')
    parser.add_argument('--thread', metavar='thread_num', help='Input thread num, default = [ ' + str(config.threads) + ' ]')
    parser.add_argument('--internal_kmer_sizes', metavar='internal_kmer_sizes', help='The k-mer size used to convert internal sequences to k-mer frequency features, default = [ ' + str(config.internal_kmer_sizes) + ' MB ]')
    parser.add_argument('--terminal_kmer_sizes', metavar='terminal_kmer_sizes', help='The k-mer size used to convert terminal sequences to k-mer frequency features, default = [ ' + str(config.terminal_kmer_sizes) + ' ]')
    parser.add_argument('--cnn_num_convs', metavar='cnn_num_convs', help='The number of CNN convolutional layers. default = [ ' + str(config.cnn_num_convs) + ' ]')
    parser.add_argument('--cnn_filters_array', metavar='cnn_filters_array', help='The number of filters in each CNN convolutional layer. default = [ ' + str(config.cnn_filters_array) + ' ]')
    parser.add_argument('--cnn_kernel_sizes_array', metavar='cnn_kernel_sizes_array', help='The kernel size in each of CNN convolutional layer. default = [ ' + str(config.cnn_kernel_sizes_array) + ' ]')
    parser.add_argument('--cnn_dropout', metavar='cnn_dropout', help='The threshold of CNN Dropout. default = [ ' + str(config.cnn_dropout) + ' ]')
    parser.add_argument('--batch_size', metavar='batch_size', help='The batch size in training model. default = [ ' + str(config.batch_size) + ' ]')
    parser.add_argument('--epochs', metavar='epochs', help='The number of epochs in training model. default = [ ' + str(config.epochs) + ' ]')
    parser.add_argument('--use_checkpoint', metavar='use_checkpoint',  help='Whether to use breakpoint training. 1: true, 0: false. The model will continue training from the last failed parameters to avoid training from head. default = [ ' + str(config.use_checkpoint) + ' ]')


    args = parser.parse_args()

    data_path = args.data
    use_terminal = args.use_terminal
    use_TSD = args.use_TSD
    use_domain = args.use_domain
    use_ends = args.use_ends
    threads = args.thread
    internal_kmer_sizes = args.internal_kmer_sizes
    terminal_kmer_sizes = args.terminal_kmer_sizes
    cnn_num_convs = args.cnn_num_convs
    cnn_filters_array = args.cnn_filters_array
    cnn_kernel_sizes_array = args.cnn_kernel_sizes_array
    cnn_dropout = args.cnn_dropout
    batch_size = args.batch_size
    epochs = args.epochs
    use_checkpoint = args.use_checkpoint


    if use_terminal is not None:
        config.use_terminal = int(use_terminal)
    if use_TSD is not None:
        config.use_TSD = int(use_TSD)
    if use_domain is not None:
        config.use_domain = int(use_domain)
    if use_ends is not None:
        config.use_ends = int(use_ends)
    if threads is not None:
        config.threads = int(threads)
    if internal_kmer_sizes is not None:
        config.internal_kmer_sizes = json.loads(internal_kmer_sizes)
    if terminal_kmer_sizes is not None:
        config.terminal_kmer_sizes = json.loads(terminal_kmer_sizes)
    if cnn_num_convs is not None:
        config.cnn_num_convs = int(cnn_num_convs)
    if cnn_filters_array is not None:
        config.cnn_filters_array = json.loads(cnn_filters_array)
    if cnn_kernel_sizes_array is not None:
        config.cnn_kernel_sizes_array = json.loads(cnn_kernel_sizes_array)
    if cnn_dropout is not None:
        config.cnn_dropout = float(cnn_dropout)
    if batch_size is not None:
        config.batch_size = int(batch_size)
    if epochs is not None:
        config.epochs = int(epochs)
    if use_checkpoint is not None:
        config.use_checkpoint = int(use_checkpoint)

    showTrainParams(data_path)

    # 实例化 DataProcessor 类
    data_processor = DataProcessor()
    # 加载数据
    X, y, seq_names = data_processor.load_data(config.internal_kmer_sizes, config.terminal_kmer_sizes, data_path)
    print(X.shape, y.shape)

    # 实例化 Trainer 类
    trainer = Trainer()

    # 进行训练
    model_path = trainer.train(X, y, config.cnn_num_convs, config.cnn_filters_array)
    print('Trained model is stored in:', model_path)

if __name__ == '__main__':
    main()