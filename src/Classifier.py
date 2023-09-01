import argparse
import os
import sys

current_folder = os.path.dirname(os.path.abspath(__file__))
# 添加 configs 文件夹的路径到 Python 路径
configs_folder = os.path.join(current_folder, "..")  # 需要根据实际目录结构调整
sys.path.append(configs_folder)

from utils.show_util import showToolName, showTestParams
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from configs import config
from CNN_Model import CNN_Model
from DataProcessor import DataProcessor
from utils.evaluate_util import get_metrics



class Classifier:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, X, y, seq_names):
        # 预测概率
        y_pred = self.model.predict(X)
        accuracy, precision, recall, f1 = get_metrics(y_pred, y, seq_names)

        return accuracy, precision, recall, f1


def main():
    showToolName()

    # 1.parse args
    describe_info = '########################## NeuralTE, version ' + str(config.version_num) + ' ##########################'
    parser = argparse.ArgumentParser(description=describe_info)
    parser.add_argument('--data', metavar='data', help='Input data to be classified, fasta format')
    parser.add_argument('--model_path', metavar='model_path',
                        help='Input the path of trained model, absolute path.')
    parser.add_argument('--thread', metavar='thread_num',
                        help='Input thread num, default = [ ' + str(config.threads) + ' ]')

    args = parser.parse_args()

    data_path = args.data
    model_path = args.model_path
    threads = args.thread

    if threads is not None:
        config.threads = int(threads)

    showTestParams(data_path, model_path)

    # 实例化 DataProcessor 类
    data_processor = DataProcessor()
    # 加载数据
    X, y, seq_names = data_processor.load_data(config.internal_kmer_sizes, config.terminal_kmer_sizes, data_path)

    # 实例化 Classifier 类
    classifier = Classifier(model_path=model_path)

    # 进行预测
    accuracy, precision, recall, f1 = classifier.predict(X, y, seq_names)
    print('accuracy, precision, recall, f1:', accuracy, precision, recall, f1)


if __name__ == '__main__':
    main()